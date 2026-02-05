import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector, Operator, Pauli
from qiskit import transpile

from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error, phase_damping_error


from mitiq import pec, zne
from mitiq import Executor
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_local_depolarizing_noise
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import PolyFactory

SEED = 7
n_qubits = 2
SHOTS = 10000
depths = list(range(1, 4))               
PEC_NUM_SAMPLES = 1000
P_TRUE = 0.01     
MISMATCH = False
TRUE_NOISE_MODEL = False

# Function to execute noisy circuit and return expectation value
def execute_noisy_expectation_true(qc) -> float:
	meas = qc.copy()
	meas.measure_all()
	tqc = transpile(meas, aer_backend, optimization_level=0)
	job = aer_backend.run(tqc, shots=SHOTS)
	counts = job.result().get_counts()

	ev = 0
	for bitstr, c in counts.items():
		b = bitstr.replace(" ", "")[-2:]
		if b in ("00", "11"):
			ev += c
		elif b in ("01", "10"):
			ev -= c

	return float(ev / sum(counts.values()))

# Create the circuits
ZZ = Operator(Pauli("ZZ"))

circuits = []
for reps in depths:
	qc = RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement="full")
	params = np.full(qc.num_parameters, 0.1)
	qc = qc.assign_parameters(params, inplace=False)
	circuits.append(qc)

# Create the noise model of the backend and the backend
p_assumed = P_TRUE 

# Mismatch scenario
if MISMATCH:
	p_assumed = P_TRUE * 1.5

nm = NoiseModel()
dep1 = depolarizing_error(P_TRUE, 1)
dep2 = depolarizing_error(P_TRUE, 2)

# True noise model with phase damping scenario
if TRUE_NOISE_MODEL:
	ph = phase_damping_error(2*P_TRUE)
	for g in ["rx", "ry", "rz", "x", "y", "z", "h", "sx", "id"]:
		nm.add_all_qubit_quantum_error(dep1.compose(ph), g)  

	for g in ["cx", "cz", "ecr"]:
		nm.add_all_qubit_quantum_error(dep2, g)  
		
# True noise model with depolarizing only scenario
else:
	for g in ["rx", "ry", "rz", "x", "y", "z", "h", "sx", "id"]:
		nm.add_all_qubit_quantum_error(dep1, g)

	for g in ["cx", "cz", "ecr"]:
		nm.add_all_qubit_quantum_error(dep2, g)

true_nm = nm
aer_backend = Aer.get_backend("aer_simulator")
aer_backend.set_options(noise_model=true_nm)

# Run the experiments
ideal_vals = []
noisy_vals = []
zne_vals = []
pec_vals = []
pec_errbars = []  

noise_factors = [1, 3, 5]
factory = PolyFactory(scale_factors=noise_factors, order=2) 

for reps, qc in zip(depths, circuits):
	print("The rep we are at:", reps)

	# Get the ideal values
	ideal = float(np.real(Statevector.from_instruction(qc).expectation_value(ZZ)))
	ideal_vals.append(ideal)

	#Get the noisy values
	noisy = float(execute_noisy_expectation_true(qc.copy()))
	noisy_vals.append(noisy)

	# PEC representations
	reps_list = represent_operations_in_circuit_with_local_depolarizing_noise(qc, p_assumed)

	executor = Executor(execute_noisy_expectation_true)

	# Get ZNE values
	zne_val = zne.execute_with_zne(
		qc,
		executor,
		factory=factory,
		scale_noise=fold_global,
	)
	zne_vals.append(float(zne_val))

	# Get PEC values
	pec_value, pec_data = pec.execute_with_pec(
		qc,
		executor,
		observable=None,              
		representations=reps_list,
		num_samples=PEC_NUM_SAMPLES,
		random_state=SEED + reps,
		full_output=True
	)
	pec_vals.append(float(pec_value))
	pec_errbars.append(float(pec_data["pec_error"]))

# Save the results
ideal_vals = np.array(ideal_vals)
noisy_vals = np.array(noisy_vals)
pec_vals = np.array(pec_vals)
pec_errbars = np.array(pec_errbars)
zne_vals = np.array(zne_vals)

noisy_abs_err = np.abs(noisy_vals - ideal_vals)
pec_abs_err = np.abs(pec_vals - ideal_vals)
zne_abs_err = np.abs(zne_vals - ideal_vals)



rows = []
for reps, ideal, noisy, pec_v, pec_e, zne_v in zip(depths, ideal_vals, noisy_vals, pec_vals, pec_errbars, zne_vals):
    rows.append({
        "reps": reps,
        "ideal_<ZZ>": ideal,
        "noisy_<ZZ>": noisy,
        "pec_<ZZ>": pec_v,
        "abs_err_noisy": abs(noisy - ideal),
        "abs_err_pec": abs(pec_v - ideal),
        "improvement": abs(noisy - ideal) - abs(pec_v - ideal),  
        "pec_errorbar": pec_e,
        "zne_<ZZ>": zne_v,
        "abs_err_zne": abs(zne_v - ideal),
        "zne_improvement_over_noisy": abs(noisy - ideal) - abs(zne_v - ideal),
    })

df = pd.DataFrame(rows)
pd.set_option("display.max_columns", None)
print(df.to_string(index=False))

df.to_csv("pec_depth_table.csv", index=False)

# Plot the results
plt.figure()
plt.plot(depths, noisy_abs_err, marker="o", label="No mitigation")
plt.plot(depths, zne_abs_err,  marker="o", label="ZNE")
plt.plot(depths, pec_abs_err, marker="o", label="PEC")
plt.yscale("log")
plt.xlabel("Circuit depth")
plt.ylabel("Absolute error")
plt.title("PEC accuracy vs depth")
plt.legend()
plt.show()

plt.figure()
plt.plot(depths, ideal_vals, marker="o", label="Ideal")
plt.plot(depths, noisy_vals, marker="o", label="Noisy")
plt.plot(depths, zne_vals,  marker="o", label="ZNE")
plt.errorbar(depths, pec_vals, yerr=pec_errbars, marker="o", capsize=3, label="PEC ± pec_error")
plt.xlabel("reps")
plt.ylabel("<ZZ>")
plt.title("Expectation values vs depth")
plt.legend()
plt.show()


