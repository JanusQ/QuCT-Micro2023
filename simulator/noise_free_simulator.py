
from qiskit import QuantumCircuit, execute
from qiskit import IBMQ, Aer
import copy

# 需要暴露电路中比特
qasm_simulator = Aer.get_backend('qasm_simulator')
def simulate_noise_free(circuit, n_samples = 2000):
    
    circuit = copy.deepcopy(circuit)
    n_qubits = circuit.num_qubits
    circuit.measure_all()
    # initial_layout = list(range(n_qubits))
    # initial_layout = circuits.qubits
    # 做噪音实现前就直接必须符合拓扑结构了吧
    # if isinstance(circuit, list):
    #     for circuit in circuit:
    #         match_hardware_constraints(circuit)
    # else:
    #     match_hardware_constraints(circuit)
    # initial_layout = initial_layout, 
    # coupling_map=coupling_map, 
    # basis_gates=basis_gates, 
    # print(circuit)
    result = execute(circuit, qasm_simulator, shots = n_samples, optimization_level = 0).result()
    return result.get_counts()

# def simulate_noise_free(circuit: QuantumCircuit, n_samples = 2000):
#     # n_qubits = circuit.num_qubits
#     n_qubits = max_qubit_num
#     initial_layout = list(range(n_qubits))
#     # initial_layout = circuits.qubits
#     # 做噪音实现前就直接必须符合拓扑结构了吧
#     # if isinstance(circuit, list):
#     #     for circuit in circuit:
#     #         match_hardware_constraints(circuit)
#     # else:
#     match_hardware_constraints(circuit)
#     # initial_layout = initial_layout, 
#     # coupling_map=coupling_map, 
#     # basis_gates=basis_gates, 
#     result = execute(circuit, qasm_simulator, shots = n_samples, optimization_level = 0).result()
#     return result.get_counts()

