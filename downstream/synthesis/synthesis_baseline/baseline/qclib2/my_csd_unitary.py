# pylint: disable=maybe-no-member
# SIWEI: 2022_5_28, 开始加入每次都会permute的代码
from math import log2

import numpy as np
import qiskit
from common_function import random_order, permute
from scipy.linalg import cossin


class CSD():
    def __init__(self, matrix):
        self.matrix = matrix
        return


def cleanMatrix(matrix):
    abs_matrix = np.abs(matrix)
    index1 = abs_matrix > 1 - 1e-5
    index0 = abs_matrix < 1e-5
    matrix[index1] = 1
    matrix[index0] = 0


# 只是作为入口函数的
def decompose_unitary(matrix):
    """
    Implements a generic quantum computation from a
    unitary matrix gate using the cosine sine decomposition.
    """
    size = len(matrix)

    if size > 4:
        n_qubits = int(log2(size))

        qubits = qiskit.QuantumRegister(n_qubits)
        circuit = qiskit.QuantumCircuit(qubits)

        permute_order = random_order(n_qubits)
        matrix = permute(matrix, permute_order)

        right_gates, theta, left_gates = cossin(matrix, size / 2, size / 2, separate=True)

        reconnect_order = [
            qubits[permute_order.index(qubit)]
            for qubit in range(n_qubits)
        ]
        # 起始depth是1
        gate_left = _unitary(list(left_gates), n_qubits, 1)
        gate_right = _unitary(list(right_gates), n_qubits, 1)

        circuit.compose(gate_left, reconnect_order, inplace=True)

        # 未来可以写的舒服点
        ucry_circuit = qiskit.QuantumCircuit(n_qubits)
        # cleanMatrix(theta) #这个是0到pi的用不了
        ucry_circuit.ucry(list(2 * theta), list(range(n_qubits - 1)), n_qubits - 1)
        circuit.compose(ucry_circuit, reconnect_order, inplace=True)
        # print(circuit)

        circuit.compose(gate_right, reconnect_order, inplace=True)

        return circuit

    circuit = qiskit.QuantumCircuit(log2(size))
    circuit.unitary(matrix, circuit.qubits)

    return circuit


# qubit order
def _unitary(gate_list, n_qubits, former_depth=0):
    former_depth += 1

    # 这部分对于csd和qsd是一样的
    if len(gate_list[0]) == 2:
        circuit = qiskit.QuantumCircuit(n_qubits)
        target = 0  # 给树节点的, 只有目标0是这个格式的（对角线上都是2x2的）
        # 是不是可以再拆成1x1的
        control = list(range(0, target)) + list(range(target + 1, n_qubits))
        circuit.uc(gate_list, control, target)
        return circuit
        # https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.uc.html

    return _csd(gate_list, n_qubits, former_depth)


# CSD decomposition
def _csd(matrix_list, n_qubits, depth=1):
    inter_matrix_size = matrix_list[0].shape[0]
    inter_qubit_num = int(log2(inter_matrix_size))
    permute_order = random_order(inter_qubit_num)
    # print(permute_order)

    matrix_list = [permute(elm, permute_order) for elm in matrix_list]
    for elm in matrix_list:
        cleanMatrix(elm)

    left, mid, right = _multiplexed_csd(matrix_list)

    gate_left = _unitary(left, n_qubits, depth)
    gate_right = _unitary(right, n_qubits, depth)

    qubits = qiskit.QuantumRegister(n_qubits)
    circuit = qiskit.QuantumCircuit(qubits)

    reconnect_order = list(range(n_qubits))
    for qubit in range(inter_qubit_num):
        reconnect_order[qubit] = qubits[permute_order.index(qubit)]

    circuit = circuit.compose(gate_left, reconnect_order)

    assert depth == log2(len(left))  # D相加的形状对应吧
    target = int(n_qubits - log2(len(left)))  # int(n_qubits-depth)
    control = list(range(0, target)) + list(range(target + 1, n_qubits))
    ucry_circuit = qiskit.QuantumCircuit(n_qubits)
    # cleanMatrix(mid)
    ucry_circuit.ucry(list(mid), control, target)  # 给中间的
    circuit = circuit.compose(ucry_circuit, reconnect_order)

    circuit = circuit.compose(gate_right, reconnect_order)

    return circuit


def _multiplexed_csd(gate_list):
    left = []
    mid = []
    right = []
    size = len(gate_list[0])
    for gate in gate_list:
        # 必须得是偶数刀?
        right_gates, theta, left_gates = cossin(gate, size / 2, size / 2, separate=True)
        # 为啥right在右边

        left = left + list(left_gates)
        right = right + list(right_gates)
        mid = mid + list(2 * theta)
    return left, mid, right
