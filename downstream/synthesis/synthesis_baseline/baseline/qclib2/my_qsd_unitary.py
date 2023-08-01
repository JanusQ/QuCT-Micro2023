# Copyright 2021 qclib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This module provides access to functions to
implement generic quantum computations.
'''

# pylint: disable=maybe-no-member

from math import log2

import numpy as np
import qiskit
from common_function import random_order, permute
from scipy.linalg import cossin


def qsd(matrix, depth=1):
    """
    Implements a generic quantum computation from a
    unitary matrix gate using the cosine sine decomposition.
    """
    size = len(matrix)
    # print(size)
    if size > 4:
        n_qubits = int(log2(size))

        qubits = qiskit.QuantumRegister(n_qubits)
        circuit = qiskit.QuantumCircuit(qubits)

        permute_order = random_order(n_qubits)
        matrix = permute(matrix, permute_order)
        reconnect_order = [
            qubits[permute_order.index(qubit)]
            for qubit in range(n_qubits)
        ]

        right_gates, theta, left_gates = cossin(matrix, size // 2, size // 2, separate=True)

        # 起始depth是1
        gate_left = _unitary(list(left_gates), n_qubits, depth)
        gate_right = _unitary(list(right_gates), n_qubits, depth)

        circuit.compose(gate_left, reconnect_order, inplace=True)

        ucry_circuit = qiskit.QuantumCircuit(n_qubits)
        ucry_circuit.ucry(list(2 * theta), list(range(n_qubits - 1)), n_qubits - 1)
        circuit.compose(ucry_circuit, reconnect_order, inplace=True)

        circuit.compose(gate_right, reconnect_order, inplace=True)

        return circuit

    circuit = qiskit.QuantumCircuit(log2(size))
    circuit.unitary(matrix, circuit.qubits)

    return circuit


# gate list里面的size得一样
def _unitary(gate_list, n_qubits, former_depth=0):
    former_depth += 1

    # 这部分对于csd和qsd是一样的
    if len(gate_list[0]) == 2:
        circuit = qiskit.QuantumCircuit(n_qubits)
        target = 0  # 给树节点的, 只有目标0是这个格式的（对角线上都是2x2的）
        control = list(range(0, target)) + list(range(target + 1, n_qubits))
        circuit.uc(gate_list, control, target)
        return circuit
        # https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.uc.html

    gate1, gate2 = gate_list
    # return _qsd(*gate_list)

    #  quantum Shannon decomposition (QSD)
    # QSD decomposition
    # def _qsd(gate1, gate2):
    # 应该就是最初输入矩阵的
    # n_qubits = int(log2(len(gate1))) + 1

    assert n_qubits == int(log2(len(gate1))) + 1

    # gate1 应该就是对半的矩阵

    list_d, gate_v, gate_w = _compute_gates(gate1, gate2)

    left_gate = qsd(gate_w)
    right_gate = qsd(gate_v)

    qubits = qiskit.QuantumRegister(n_qubits)
    circuit = qiskit.QuantumCircuit(qubits)

    circuit = circuit.compose(left_gate, qubits[0:-1])
    circuit.ucrz(list(-2 * np.angle(list_d)), qubits[0:-1], qubits[-1])
    circuit = circuit.compose(right_gate, qubits[0:-1])

    return circuit


def _is_unitary(matrix):
    is_identity = np.conj(matrix.T).dot(matrix)
    return np.allclose(is_identity, np.identity(matrix.shape[0]))


def _closest_unitary(matrix):
    svd_u, _, svd_v = np.linalg.svd(matrix)
    return svd_u.dot(svd_v)


def _compute_gates(gate1, gate2):
    d_square, gate_v = np.linalg.eig(gate1 @ gate2.conj().T)
    # egin_value, egin_vector

    list_d = np.sqrt(d_square, dtype=complex)
    gate_d = np.diag(list_d)

    if not _is_unitary(gate_v):
        # degeneracy
        gate_v = _closest_unitary(gate_v)

    gate_w = gate_d @ gate_v.conj().T @ gate2

    return list_d, gate_v, gate_w
