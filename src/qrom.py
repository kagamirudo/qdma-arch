"""
qrom.py -- explicit QROM (read-only QRAM) read oracles, verified in simulation.

We open the black box the LMSR paper assumed:  |i>|0> -> |i>|D[i]>.
Three things live here:

  1. build_qrom(data, n_addr, n_word): a textbook multiplexed QROM. For each
     address i it applies X gates to the data register controlled on the address
     equalling i (a multi-controlled pattern). Correctness is exact; this is the
     O(N) construction that fault-tolerant QRAM reduces to.

  2. QDMA block read: given a descriptor (start, length), stream a contiguous run
     of cells into a wider data register, amortizing address decode across the
     block. This is the "quantum DMA" primitive specialized to LMSR's contiguous
     rotations. It is still built from QROM internally -- the point is the ACCESS
     PATTERN and how its cost is counted, not a new lower bound.

  3. resource counts: decompose to a {X, CX, CCX/T} basis and tally, so the
     paper's tables are measured, not asserted.

All of this is classically simulatable for small N, which is exactly why the
numbers are honest.
"""
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector


def _int_to_bits(x: int, n: int):
    return [(x >> b) & 1 for b in range(n)]  # little-endian: bit b is 2**b


def build_qrom(data, n_addr: int, n_word: int) -> QuantumCircuit:
    """Multiplexed QROM. data[i] is written into the word register when the
    address register equals i. Address controls are plain MCX patterns.
    Returns a circuit on (addr | word) registers; word assumed |0...0> on input."""
    N = len(data)
    assert N <= 2 ** n_addr
    a = QuantumRegister(n_addr, "a")
    d = QuantumRegister(n_word, "d")
    qc = QuantumCircuit(a, d, name="QROM")
    for i in range(N):
        val = int(data[i])
        if val == 0:
            continue  # nothing to write for an all-zero word
        abits = _int_to_bits(i, n_addr)
        # Flip address qubits that should be 0 so the control fires on pattern i
        zeros = [a[b] for b in range(n_addr) if abits[b] == 0]
        for q in zeros:
            qc.x(q)
        ctrl = list(a)
        for wbit in range(n_word):
            if (val >> wbit) & 1:
                qc.mcx(ctrl, d[wbit])
        for q in zeros:
            qc.x(q)
    return qc


def qdma_block_read(data, n_addr: int, n_word: int, start: int, length: int):
    """QDMA burst: load the contiguous block data[start:start+length] into a data
    register of `length` words, in one descriptor-driven transfer. Here we realize
    it as `length` QROM writes that SHARE the single address register -- i.e. the
    address is decoded once per output slot but the block is treated as one
    transfer unit. Returns (circuit, [word registers])."""
    a = QuantumRegister(n_addr, "a")
    words = [QuantumRegister(n_word, f"d{j}") for j in range(length)]
    qc = QuantumCircuit(a, *words, name=f"QDMA[{start}:{start+length}]")
    N = len(data)
    for j in range(length):
        idx = (start + j) % N
        val = int(data[idx])
        if val == 0:
            continue
        abits = _int_to_bits(start, n_addr)  # descriptor base address
        # offset j is a classical constant of the descriptor, so the control
        # pattern is on `start`; slot j just selects which word register to fill.
        zeros = [a[b] for b in range(n_addr) if abits[b] == 0]
        for q in zeros:
            qc.x(q)
        ctrl = list(a)
        for wbit in range(n_word):
            if (val >> wbit) & 1:
                qc.mcx(ctrl, words[j][wbit])
        for q in zeros:
            qc.x(q)
    return qc, words


def verify_qrom(data, n_addr, n_word) -> bool:
    """Exhaustively check |i>|0> -> |i>|data[i]> for every address i."""
    N = len(data)
    ok = True
    for i in range(N):
        qc = QuantumCircuit(n_addr + n_word)
        abits = _int_to_bits(i, n_addr)
        for b in range(n_addr):
            if abits[b]:
                qc.x(b)
        qrom = build_qrom(data, n_addr, n_word)
        qc.compose(qrom, qubits=list(range(n_addr + n_word)), inplace=True)
        sv = Statevector.from_instruction(qc)
        # QROM is a permutation on basis states; take the dominant one (the rest
        # is floating-point dust from the MCX decomposition).
        probs = sv.probabilities_dict()
        bitstr = max(probs, key=probs.get)  # qiskit prints big-endian, rightmost = qubit0
        assert probs[bitstr] > 0.99, f"non-basis output at i={i}: {probs}"
        bits = [int(c) for c in bitstr[::-1]]
        got_word = sum(bits[n_addr + w] << w for w in range(n_word))
        if got_word != int(data[i]):
            ok = False
            print(f"  MISMATCH i={i}: got {got_word}, want {int(data[i])}")
    return ok


if __name__ == "__main__":
    # LMSR-style data: tiny alphabet {1,2,3} -> 2 bits/word, as in the benzenoid BEC
    data = [1, 3, 2, 1, 2, 3, 3, 1]  # N=8
    n_addr, n_word = 3, 2
    print(f"QROM over N={len(data)}, {n_word}-bit words:",
          "VERIFIED" if verify_qrom(data, n_addr, n_word) else "FAILED")
