import matplotlib.pyplot as plt
import numpy as np
from mitiq import zne
from mitiq.zne.inference import LinearFactory
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import QiskitRuntimeService
import json

service = QiskitRuntimeService()
backend = service.backend("ibm_torino")

# Get IBM hardware properties
props = backend.properties()

run_information = {}

# Extract average 1-qubit gate error
errors_1q = []
for q in range(backend.num_qubits):
    try:
        props = backend.target['sx'][(q,)]
        if props.error is not None:
            errors_1q.append(props.error)
    except KeyError:
        continue

avg_1q_error = np.mean(errors_1q)

# Extract average 2-qubit (CNOT) gate error
props = backend.properties()

# Extract 2-qubit errors by searching for gates with 2 qubits in their description
errors_2q = []

for gate in props.gates:
    # We only want gates that involve exactly two qubits
    if len(gate.qubits) == 2:
        # Pull the 'gate_error' parameter from the gate's parameter list
        for param in gate.parameters:
            if param.name == 'gate_error':
                errors_2q.append(param.value)


if errors_2q:
    avg_2q_error = np.mean(errors_2q)
    print(f"Found {len(errors_2q)} two-qubit gate calibrations.")
else:
    # If it's still empty use a default value
    print("Warning: Could not find 2-qubit gate errors. Falling back to 0.01.")
    avg_2q_error = 0.01

print(f"Real p_1q (Avg): {avg_1q_error:.5f}")
print(f"Real p_2q (Avg): {avg_2q_error:.5f}")
run_information["1q_error"] = avg_1q_error
run_information["2q_error"] = avg_2q_error

# Setup Noisy Simulator
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(avg_1q_error, 1), ['h'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(avg_2q_error, 2), ['cx'])
backend = AerSimulator(noise_model=noise_model)

max_num_qubits = 12
num_qubits = 2

def executor(circuit: QuantumCircuit) -> float:
    meas_circ = circuit.copy()
    meas_circ.measure_all()

    job = backend.run(meas_circ, shots=10000)  # Higher shots for stability
    counts = job.result().get_counts()

    # Calculate expectation value of the |00> state
    return counts.get('0' * num_qubits, 0) / 10000


# Run Experiment
qubit_nrs = range(num_qubits, max_num_qubits + 1, 2)
depths = range(1, 91, 3)
unmitigated_errors = []
all_zne_errors = []
cutoff_depths = {}

for num_qubits in qubit_nrs:
    zne_errors = []
    unmitigated_errors = []
    actual_depths = []
    for d in depths:
        # A simple Bell-state mirror circuit
        print(f"Num qubits: {num_qubits}, d: {d}")
        c = QuantumCircuit(num_qubits)
        for _ in range(d):
            for k in range(num_qubits - 1):
                c.cx(k, k + 1)
            c.barrier()

        # Add the inverse to make the ideal result
        c.compose(c.inverse(), inplace=True)
        actual_depths.append(c.depth())

        # Unmitigated
        val_raw = executor(c)
        unmitigated_errors.append(abs(1.0 - val_raw))

        # ZNE mitigation (Linear)
        fac = LinearFactory(scale_factors=[1.0, 3.0, 5.0]) # Industry standard
        val_zne = zne.execute_with_zne(c, executor, factory=fac)

        zne_errors.append(abs(1.0 - val_zne))


    run_information[f"{num_qubits}_depth"] = actual_depths
    run_information[f"{num_qubits}_unmitigated"] = unmitigated_errors
    run_information[f"{num_qubits}_zne"] = zne_errors

    # Plotting the errors
    plt.figure(figsize=(12, 6))
    plt.plot(actual_depths, unmitigated_errors, 'o-', label='Raw Error', linewidth=2)
    plt.plot(actual_depths, zne_errors, 'x--', label=f'ZNE Error (qubits {num_qubits})', linewidth=2)

    legend_added = False
    for i in range(len(unmitigated_errors)):
        if unmitigated_errors[i] <= zne_errors[i]:
            print(f"{num_qubits} qubit cutoff: {actual_depths[i]}")
            cutoff_depths[num_qubits] = actual_depths[i]

            label_text = f'ZNE cutoff: {actual_depths[i]}' if not legend_added else ""
            plt.axvline(x=actual_depths[i], color='gray', linestyle='--', alpha=0.5, label=label_text)
            legend_added = True
            break

    plt.ylabel("Expectation Value Error", fontsize=20, labelpad=15)
    plt.xlabel("Circuit Depth", fontsize=20, labelpad=15)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.legend(fontsize=14, loc='best')

    plt.tight_layout()
    plt.savefig(f"ZNE {num_qubits}q.png")
    plt.show()

print(unmitigated_errors)
run_information["cutoff_depths"] = cutoff_depths

x = np.array(list(cutoff_depths.keys()))
y = np.array(list(cutoff_depths.values()))

coefficients = np.polyfit(x, y, 1)
polynomial = np.poly1d(coefficients)

line_of_best_fit = polynomial(x)
run_information["polynomial"] = {}
run_information["polynomial"]["m"] = polynomial[0]
run_information["polynomial"]["c"] = polynomial[1]

with open("run_information.json", "w") as f:
    json.dump(run_information, f, indent=4)

print(f"Curve of best fit: {polynomial}")

plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, line_of_best_fit, color='red', label='Line of Best Fit')

plt.xlabel('Number of Qubits')
plt.ylabel('Cutoff Depth')
plt.legend()
plt.savefig("Cutoff depth extrapolation.png")
plt.show()