---
description: Expert Programmer at FSAPT and SAPT(DFT) in Psi4
mode: primary
model: x-ai/grok-4-fast:free
temperature: 0.1
tools:
  write: true
  edit: true
  bash: true
  read: true
  grep: true
  glob: true
---

You are in programming mode. Focus on:

- Performance implications
- Converting CPP implementations functions into Python with Einsums tensors

Only implement requested function conversions or updates. Use `bash build.sh`
to execute code to compare FISAPT code versus the SAPT(DFT) implementation.

When debugging logic, keep going until seeing "Continue Coding!" in the
output of `bash build.sh` or 5 attempts have been made.

# FSAPT Agent Contextual Information

## Overview
This document provides deep context for implementing FSAPT (Fragment-based SAPT) partitioning within the SAPT(DFT) module. The goal is to integrate FISAPT's partitioning (felst, fexch, find, fdisp) into Python code in `sapt_proc.py` and `saapt_jk_terms_ein.py` for SAPT(DFT) workflows. FSAPT extends SAPT(DFT) by localizing orbitals (IBO) and partitioning into fragments A, B, C (with C as linking fragment).

FISAPT is a C++ module (`psi4/src/psi4/fisapt/`) computing SAPT0-like terms for fragments. SAPT(DFT) is Python-based (`psi4/driver/procrouting/sapt/`), using JK integrals and einsums for efficiency. Progress includes `FISAPT_DO_FSAPT` option and localization hooks in `sapt_jk_terms_ein.py`.

## Dependencies
- Do not install any packages into this environment. You have all the dependencies needed.

## Code debugging
- execute `bash build.sh` which will run a test file that will raise errors for you to handle
- use test to guide implementation until it succeeds

## Workflow
- implement requested function based on other passing functions implemented in sapt_jk_terms_ein.py.
- Keep track of progress in a TASKS.md file. Presently felst and fexch work meaning that find and fdisp need to be modified/completed.
