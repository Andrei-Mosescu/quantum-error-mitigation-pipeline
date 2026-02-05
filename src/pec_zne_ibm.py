import time
import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, EstimatorOptions

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform", 
    token="",
    overwrite=True  
)

service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True)
print("Using backend:", backend.name)

DEPTHS = [1, 2, 4, 6, 8, 11]
ZNE_FACTORS = [1, 3, 5]
PEC_MAX_OVERHEAD = 300    
       
OBS = SparsePauliOp.from_list([("ZZ", 1.0)])

pm = generate_preset_pass_manager(optimization_level=0, backend=backend)

pubs = []       
ideal_vals = [] 
depth_after = [] 
twoq_counts = []
oneq_counts = []

def make_identity_depth_circuit(reps: int) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    for _ in range(reps):
        qc.cx(0, 1)
        qc.barrier() 
        qc.cx(0, 1)  
        qc.barrier()
    return qc

for reps in DEPTHS:
	qc = RealAmplitudes(num_qubits=2, reps=reps, entanglement="full")
	params = [0.1] * qc.num_parameters

	qc_ideal = qc.assign_parameters(params, inplace=False)
	ideal = float(np.real(Statevector.from_instruction(qc_ideal).expectation_value(OBS)))
	ideal_vals.append(ideal)

	isa_circuit = pm.run(qc)
	counts = isa_circuit.count_ops()
	twoq_counts.append(counts.get("cx", 0) + counts.get("ecr", 0))
	oneq_counts.append(sum(v for k, v in counts.items() if k in ["rz","sx","x","id"]))
	isa_obs = OBS.apply_layout(isa_circuit.layout)

	depth_after.append(isa_circuit.depth())
	pubs.append((isa_circuit, isa_obs, params))

ideal_vals = np.array(ideal_vals)

def run_job(label: str, options: EstimatorOptions):
        est = EstimatorV2(mode=backend, options=options)
        job = est.run(pubs)
        res = job.result()

        evs = np.array([float(r.data.evs) for r in res], dtype=float)

        stds = None
        try:
                stds = np.array([float(r.data.stds) for r in res], dtype=float)
        except Exception:
                pass

        print(f"{label}: job_id={job.job_id()}")
        return evs, stds, job.job_id()

optP = EstimatorOptions()
optP.resilience_level = 1
optP.resilience.pec_mitigation = True
optP.resilience.pec.max_overhead = PEC_MAX_OVERHEAD
ev_pec, std_pec, id_pec = run_job("PEC", optP)

opt0 = EstimatorOptions()
opt0.resilience_level = 0
ev_unmit, std_unmit, id_unmit = run_job("UNMIT", opt0)

optZ = EstimatorOptions()
optZ.resilience_level = 2
optZ.resilience.zne_mitigation = True
optZ.resilience.zne.noise_factors = ZNE_FACTORS
optZ.resilience.zne.extrapolator = "polynomial_degree_2"
ev_zne, std_zne, id_zne = run_job("ZNE", optZ)

abs_err_unmit = np.abs(ev_unmit - ideal_vals)
abs_err_zne   = np.abs(ev_zne   - ideal_vals)
abs_err_pec   = np.abs(ev_pec   - ideal_vals)

df = pd.DataFrame({
    "reps": DEPTHS,
    "transpiled_depth": depth_after,
    "ideal_<ZZ>": ideal_vals,
    "unmit_<ZZ>": ev_unmit,
    "zne_<ZZ>": ev_zne,
    "pec_<ZZ>": ev_pec,
    "abs_err_unmit": abs_err_unmit,
    "abs_err_zne": abs_err_zne,
    "abs_err_pec": abs_err_pec,
    "zne_improvement_over_unmit": abs_err_unmit - abs_err_zne,
    "pec_improvement_over_unmit": abs_err_unmit - abs_err_pec,
    "n_2q": twoq_counts,
	"n_1q": oneq_counts
})


print("\nResults:")
print(df.to_string(index=False))

print("\nJob IDs:")
print("UNMIT:", id_unmit)
print("ZNE  :", id_zne)
print("PEC  :", id_pec)

df.to_csv("ibm_zne_vs_pec_depths100.csv", index=False)
print("\nSaved: ibm_zne_vs_pec_depths300RA.csv")

plt.figure()
plt.plot(depth_after, abs_err_unmit, marker="o", label="Unmitigated")
plt.plot(depth_after, abs_err_zne,   marker="o", label=f"ZNE factors={ZNE_FACTORS}")
plt.plot(depth_after, abs_err_pec,   marker="o", label=f"PEC max_overhead={PEC_MAX_OVERHEAD}")
plt.yscale("log")
plt.xlabel("Transpiled circuit depth")
plt.ylabel("Absolute error |<ZZ> - <ZZ>_ideal| (log)")
plt.title(f"IBM hardware: ZNE vs PEC vs depth ({backend.name})")
plt.legend()
plt.show()

plt.figure()
plt.axhline(1.0, linestyle="--", label="Ideal = 1")
plt.plot(depth_after, ev_unmit, "o-", label="Unmitigated")
plt.plot(depth_after, ev_zne,   "o-", label="ZNE")
plt.plot(depth_after, ev_pec,   "o-", label="PEC")
plt.xlabel("Transpiled depth")
plt.ylabel("<ZZ>")
plt.legend()
plt.show()

plt.figure()
plt.plot(depth_after, df["zne_improvement_over_unmit"], marker="o", label="ZNE improvement")
plt.plot(depth_after, df["pec_improvement_over_unmit"], marker="o", label="PEC improvement")
plt.axhline(0, linestyle="--")
plt.xlabel("Transpiled circuit depth")
plt.ylabel("Improvement in abs error")
plt.title("Mitigation improvement vs depth")
plt.legend()
plt.show()
