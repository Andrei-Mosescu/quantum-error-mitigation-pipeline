import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorOptions, EstimatorV2, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import gc
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker

# Create the ZNE cutoff polynomial
a = 10.83  # Slope
b = 99.2  # Intercept
ZNE_line_model = np.poly1d([a, b])


def get_average_2q_error(backend):
    """
    Safely retrieves the mean 2-qubit gate error from the backend target.
    """
    # Identify which entangling gate the backend actually uses
    # Most common are 'cx' or 'ecr'
    available_ops = backend.operation_names

    if 'cx' in available_ops:
        gate_name = 'cx'
    elif 'ecr' in available_ops:
        gate_name = 'ecr'
    elif 'cz' in available_ops:
        gate_name = 'cz'
    else:
        print("Warning: No standard 2-qubit gate found. Defaulting to 1% error.")
        return 0.01

    # Extract the properties from the target
    try:
        instruction_props = backend.target[gate_name]

        # Get errors for all qubit combinations where this gate is defined
        errors = [
            props.error for props in instruction_props.values()
            if props is not None and hasattr(props, 'error') and props.error is not None
        ]

        if not errors:
            return 0.01

        return sum(errors) / len(errors)

    except KeyError:
        return 0.01


def estimate_pec_overhead_dynamic(circuit, backend):
    avg_error = get_average_2q_error(backend)

    ops = circuit.count_ops()
    # Count all possible 2-qubit gates
    total_heavy_gates = ops.get('cx', 0) + ops.get('ecr', 0) + ops.get('cz', 0)

    # Gamma calculation based on actual hardware noise
    gamma_gate = 1 + (2 * avg_error)
    total_overhead = gamma_gate ** (2 * total_heavy_gates)

    return total_overhead, avg_error

def run_smart_mitigation_pipeline(circuit, backend, seed=42):
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa_circuit = pm.run(circuit)

    # Get dynamic hardware stats
    overhead_est, live_error_rate = estimate_pec_overhead_dynamic(isa_circuit, backend)
    depth = isa_circuit.depth()

    print(f"--- Hardware-Aware Analysis ---")
    print(f"Avg 2Q Error: {live_error_rate:.4f}")
    print(f"Estimated PEC Overhead: {overhead_est:.2f}x")

    ZNE_cutoff_depth = ZNE_line_model(circuit.num_qubits)

    options = EstimatorOptions()

    if overhead_est < 5:
        print("Selection: PEC (Explicitly enabled)")
        # In V2, we enable PEC like this:
        options.resilience.pec_mitigation = True
        options.resilience.pec.max_overhead = 100.0
        selection = 3
    elif depth < ZNE_cutoff_depth:
        print("Selection: Custom ZNE (Level 2)")
        options.resilience_level = 2
        options.resilience.zne_mitigation = True
        options.resilience.zne.extrapolator = "polynomial_degree_2"
        options.resilience.zne.noise_factors = [1, 3, 5]
        selection = 2
    else:
        print("Selection: TREX (Default Level 1)")
        options.resilience_level = 1
        selection = 1

    return pm, isa_circuit, options, selection


# --- EXECUTION ---

# Setup Service and Backend
try:
    service = QiskitRuntimeService()
    backend = service.backend("ibm_torino")
except:
    print("No IBM account found. Please ensure you have credentials saved.")
    backend = None

size_depth = 30
size_qubits = 30

algorithms_used = np.zeros((size_qubits, size_depth), dtype=int)

isa_depths = []

if backend:
    for i in range(1, size_qubits+1):
        for j in range(1, size_depth+1):
            # Create Dummy Circuit
            num_qubits = i
            depth_layers = j

            print("Qubits: ", num_qubits)
            print("Depth: ", depth_layers)
            qc = QuantumCircuit(num_qubits)

            # Create a dense circuit of entangling gates
            for _ in range(depth_layers):
                for k in range(num_qubits - 1):
                    qc.cx(k, k + 1)
                qc.barrier()  # Optional: helps visualize the layers

            # Define an observable matching the qubit count
            observable = SparsePauliOp("Z" * num_qubits)

            # Process Pipeline
            pm, isa_circuit, options, selection = run_smart_mitigation_pipeline(qc, backend)
            isa_observable = observable.apply_layout(isa_circuit.layout)
            isa_depths.append(isa_circuit.depth())

            algorithms_used[i-1, j-1] = selection

            # Run Job
            estimator = EstimatorV2(mode=backend, options=options)
            print(f"Submitting job to {backend.name}...")

            del qc
            del isa_circuit
            del pm
            gc.collect()
else:
    print("Backend not accessible.")

algorithms_used = np.array(algorithms_used)

fig, ax = plt.subplots(figsize=(16, 14))

# Custom Discrete Colormap
cmap = ListedColormap(['#e74c3c', '#3498db', '#2ecc71'])

# Create Heatmap
im = ax.imshow(algorithms_used,
               origin='lower',
               cmap=cmap,
               aspect='auto',
               extent=[0.5, size_depth + 0.5, 0.5, size_qubits + 0.5])

# Grid and Axis Formatting
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Set minor ticks for separators
ax.set_xticks(np.arange(0.5, size_depth + 1.5, 1), minor=True)
ax.set_yticks(np.arange(0.5, size_qubits + 1.5, 1), minor=True)

# Grid styling
ax.grid(visible=True, which='minor', color='white', linestyle='-', linewidth=1)
ax.grid(visible=False, which='major')

# Text Formatting
ax.set_xlabel("Circuit Depth (Layers)", fontsize=20, labelpad=15)
ax.set_ylabel("Number of Qubits", fontsize=20, labelpad=15)

# Increase size of the numbers on the axes
ax.tick_params(axis='both', which='major', labelsize=18)

# Creating the colorbar
cbar = plt.colorbar(im, ticks=[1, 2, 3], shrink=0.7)
cbar.ax.set_yticklabels(['TREX', 'ZNE', 'PEC'],
                        fontsize=20)
cbar.set_label('Mitigation Method', fontsize=18, labelpad=15)

plt.tight_layout()

plt.savefig("Error Mitigation Pipeline Colour Map.png")
plt.show()