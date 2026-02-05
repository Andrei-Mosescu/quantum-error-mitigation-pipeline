# Quantum Error Mitigation Pipeline (Qiskit)

This project investigates practical quantum error mitigation techniques and implements an automated pipeline that selects the most suitable technique for a given quantum circuit.

## Overview

Near-term quantum computers suffer heavily from noise and hardware imperfections.
Full quantum error correction is still out of reach, making Quantum Error Mitigation (QEM) a crucial tool.

In this project we study and compare three major mitigation techniques:

- Zero Noise Extrapolation (ZNE)
- Probabilistic Error Cancellation (PEC)
- Twirled Readout Error eXtinction (TREX)

We then design a pipeline that automatically switches between them depending on circuit size and depth.

Each mitigation technique has different trade-offs:

Method	Strength	Weakness
PEC	Unbiased expectation values	Exponential sampling overhead
ZNE	Linear overhead	Breaks down at large depths
TREX	Very cheap and robust	Only fixes readout errors

The goal of this project was to determine when each technique should be used and automate the decision.

## Main Contributions

### ZNE cutoff modelling

We experimentally determined the circuit depth at which ZNE stops improving results and fitted a linear model that predicts this cutoff based on qubit count.

### PEC vs ZNE evaluation

We studied how PEC behaves under realistic conditions, including noise-model mismatch and increasing circuit depth.

### Automated mitigation pipeline

We implemented a pipeline that:

- Estimates PEC sampling overhead
- Predicts ZNE usefulness from circuit depth
- Automatically selects the mitigation strategy

The pipeline integrates with IBM Quantum backends and uses calibration data to make hardware-aware decisions.

## Tech Stack

- Python
- Qiskit & Qiskit Runtime
- Mitiq
- NumPy / Matplotlib
- IBM Quantum hardware

## Report

Full technical report:
report/quantum_error_mitigation_pipeline.pdf