from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from utils.backend_info import max_qubit_num
import numpy as np
import scipy.linalg as la
from qiskit.result import LocalReadoutMitigator

bitstrings_raw = {
    '11111': 100,
    '00000': 100,
}

measure0_fidelity = np.array([0.97535460, 0.97535460, 0.9645634, 0.9907482, 0.96958333])
measure1_fidelity = np.array([0.955646258, 0.97572327, 0.950431034, 0.9629411764, 0.9570833333])

measMats, measMats_inv = [], []
for qubit in range(max_qubit_num):
    measMat = np.array([[
        measure0_fidelity[qubit], 1-measure1_fidelity[qubit]],
        [1-measure0_fidelity[qubit], measure1_fidelity[qubit]]
    ])
    measMat_inv = np.linalg.inv(measMat)
    measMats.append(measMat)
    measMats_inv.append(measMat_inv)

measMats = np.array(measMats)
# print(len(measMats))
# print(qnum)
# print(bitstrings_raw)
lrm_mitigator = LocalReadoutMitigator(measMats, list(range(max_qubit_num)))

def correct_measure(bitstrings_raw):
    mitigated_quasi_probs = lrm_mitigator.quasi_probabilities(bitstrings_raw)
    # mitigated_stddev = mitigated_quasi_probs._stddev_upper_bound
    mitigated_probs = mitigated_quasi_probs.nearest_probability_distribution().binary_probabilities()
    mitigated_probs = {bits.zfill(max_qubit_num): mitigated_probs[bits] for bits in mitigated_probs}
    # print(mitigated_probs)
    return mitigated_probs

ret = correct_measure(bitstrings_raw)
print(ret)