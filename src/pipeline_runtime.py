import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorOptions, EstimatorV2, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

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

    # Extract properties from target
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
        options.resilience.pec_mitigation = True
        options.resilience.pec.max_overhead = 100.0
    elif depth < ZNE_cutoff_depth:
        print("Selection: Custom ZNE (Level 2)")
        options.resilience_level = 2
        options.resilience.zne_mitigation = True
        options.resilience.zne.extrapolator = "polynomial_degree_2"
        options.resilience.zne.noise_factors = [1, 3, 5]
    else:
        print("Selection: TREX (Default Level 1)")
        options.resilience_level = 1

    return pm, isa_circuit, options


# --- EXECUTION ---

# Setup Service and Backend
try:
    service = QiskitRuntimeService()
    backend = service.backend("ibm_torino")
except:
    print("No IBM account found. Please ensure you have credentials saved.")
    backend = None

if backend:
    # Create Dummy Circuit
    # Dummy Circuit Start -----------------------------------------------------------------
    # Variables of the circuit to change
    num_qubits = 2
    depth_layers = 2

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

    # Dummy Circuit End -------------------------------------------------------------------

    # Process Pipeline
    pm, isa_circuit, options = run_smart_mitigation_pipeline(qc, backend)
    isa_observable = observable.apply_layout(isa_circuit.layout)

    # Run Job
    estimator = EstimatorV2(mode=backend, options=options)
    print(f"Submitting job to {backend.name}...")

    # Run the circuit
    job = estimator.run([(isa_circuit, isa_observable)])
    result = job.result()

    # You can use the result form here on...
else:
    print("Backend not accessible.")