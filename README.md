# A Coherent QDMA Architecture for Quantum String Rotation

Companion to the QRAM-costing paper (`kagamirudo/qram-lmsr`). That paper proposed
a DMA-style burst read (QDMA) but verified it only for a **classical** base
address. LMSR's Dürr–Høyer engine supplies a **superposed** base. This paper
builds and verifies the architecture LMSR actually needs: a **coherent QDMA
engine** that bursts a contiguous window from a base address in superposition.

## Result (verified in Qiskit, regenerates from a fixed seed)
- The engine maps `(1/√N)Σ|k⟩ → (1/√N)Σ|k⟩|s_k…s_{k+L-1}⟩` — **fidelity 1.0** to
  the ideal **entangled** target on every tested (N,L); index register returned
  clean.
- Address–data entanglement characterized exactly: base-register entropy =
  log₂(distinct length-L windows), ≤ log₂N, saturating iff all windows differ.
  → a precise design constraint on burst length L.
- One shared decode fabric reused across L classical-offset taps (L:1 vs L
  independent coherent reads).

## Why it needs no lab
Correctness is a statevector fidelity (exact math) and entanglement is a reduced
density-matrix entropy (exact). No hardware. The physical bucket-brigade mapping
is explicitly **design rationale, not a device claim**.

## Layout
```
qdma_arch.tex / .pdf   the paper (with a TikZ datapath figure)
refs.bib               verified bibliography
src/qrom.py            shared read fabric (from the companion repo)
src/coherent_qdma.py   the coherent engine: superposed base + modular window + verify
src/experiments.py     fidelity + entanglement-entropy tables
results/               coherent_verify.csv, shared_decode.csv
```

## Reproduce
```bash
# reuses the companion repo's qiskit venv, or:
python3 -m venv .venv && .venv/bin/pip install qiskit qiskit-aer numpy
.venv/bin/python src/coherent_qdma.py        # fidelity-1.0 verification
.venv/bin/python src/experiments.py results   # entropy tables
pdflatex qdma_arch.tex && pdflatex qdma_arch.tex
```

## Scope / honesty
Circuit-level architecture, simulation-verified, specialized to **contiguous**
access. Not a physical-device claim; does not beat general QRAM lower bounds. The
modular adder is an explicit permutation for exact checking; a scalable build uses
the Draper QFT adder.
