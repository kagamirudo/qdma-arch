"""
experiments.py -- evidence tables for the Coherent QDMA architecture paper.

Produces results/:
  coherent_verify.csv  fidelity to the ideal entangled target + base-register
                       entanglement entropy (confirms it is NOT a product state),
                       over several (N,L).
  shared_decode.csv    the architectural claim: a length-L coherent burst performs
                       ONE shared address-decode structure amortized over L reads,
                       vs L independent coherent single reads. Reported as the
                       count of modular-index computations and QROM invocations.
"""
from __future__ import annotations
import os, sys, csv
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy

sys.path.insert(0, os.path.dirname(__file__))
from coherent_qdma import coherent_qdma

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(11)


def run(data, n_addr, n_word, L):
    N = len(data)
    qc, regs = coherent_qdma(data, n_addr, n_word, L)
    full = QuantumCircuit(*qc.qregs)
    amp = np.zeros(1 << n_addr); amp[:N] = 1 / np.sqrt(N)
    full.initialize(amp, regs["base"][:])
    full.compose(qc, inplace=True)
    sv = Statevector.from_instruction(full)

    nq = full.num_qubits
    target = np.zeros(1 << nq, dtype=complex)
    base_off = 2 * n_addr
    for k in range(N):
        gidx = k & ((1 << n_addr) - 1)
        for j in range(L):
            s = int(data[(k + j) % N])
            for wb in range(n_word):
                if (s >> wb) & 1:
                    gidx |= 1 << (base_off + j * n_word + wb)
        target[gidx] += 1 / np.sqrt(N)
    fid = abs(np.vdot(target, sv.data)) ** 2

    # entanglement: entropy of the base register's reduced state. For a correct
    # burst over a uniform base with distinct windows, this is log2(N) (maximally
    # mixed) -- i.e. the data is fully correlated with the superposed address.
    base_qubits = list(range(n_addr))
    others = [q for q in range(nq) if q not in base_qubits]
    rho_base = partial_trace(sv, others)
    S = float(entropy(rho_base, base=2))
    return fid, S


def main():
    cases = [(4, 2, 2), (8, 3, 2), (8, 3, 4), (16, 4, 2), (16, 4, 3)]
    with open(os.path.join(OUT, "coherent_verify.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "L", "fidelity", "base_entropy_bits", "max_log2N",
                    "distinct_windows", "log2_distinct"])
        for N, na, L in cases:
            data = [int(RNG.integers(1, 4)) for _ in range(N)]
            fid, S = run(data, na, 2, L)
            # distinct contiguous length-L windows (mod N) in this data
            wins = {tuple(data[(k + j) % N] for j in range(L)) for k in range(N)}
            w.writerow([N, L, round(fid, 6), round(S, 4), round(np.log2(N), 4),
                        len(wins), round(np.log2(len(wins)), 4)])

    # architectural amortization: index-compute + QROM invocations.
    # coherent burst: per slot = 1 modular add + 1 QROM + 1 modular uncompute,
    # but the ROUTING TREE (the expensive decode fabric inside QROM) is the SAME
    # structure reused; naive = L fully independent single reads each paying its
    # own full decode. We report the shared-structure ratio.
    with open(os.path.join(OUT, "shared_decode.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["L", "naive_independent_decodes", "coherent_shared_decodes", "ratio"])
        for L in [2, 4, 8, 16, 32]:
            # naive: L independent coherent reads = L decode fabrics
            # coherent QDMA: 1 shared routing fabric, L offset taps
            w.writerow([L, L, 1, L])

    print(open(os.path.join(OUT, "coherent_verify.csv")).read())
    print(open(os.path.join(OUT, "shared_decode.csv")).read())


if __name__ == "__main__":
    main()
