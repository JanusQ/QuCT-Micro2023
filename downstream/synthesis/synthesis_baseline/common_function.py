from functools import lru_cache
from math import log2 as math_log2

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.opflow.primitive_ops.matrix_op import MatrixOp
from qiskit.quantum_info import Operator
from scipy.linalg import block_diag


@lru_cache
def string2binary(binary_string):
    value = 0
    for index, char in enumerate(binary_string):
        if char == '1':
            index = len(binary_string) - index - 1
            value += 2 ** index
    return value


# def is_commute(matirx1, matirx2):
#     return

def is_unitary(matrix):
    is_identity = np.conj(matrix.T).dot(matrix)
    return np.allclose(is_identity, np.identity(matrix.shape[0]))


def closest_unitary(matrix):
    svd_u, _, svd_v = np.linalg.svd(matrix)
    return svd_u.dot(svd_v)


# TODO: 未来换成 >>1
@lru_cache
def log2(value):
    return int(math_log2(value))


def isDiagMat(mat: np.ndarray) -> bool:
    """
    Returns True iff arr is numpy array for diagonal square matrix.

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    bool

    """

    num_rows = mat.shape[0]
    assert mat.shape == (num_rows, num_rows)
    # this extracts diagonal v, then
    # creates a diagonal matrix with v as diagonal
    mat1 = np.diag(np.diag(mat))
    return np.linalg.norm(mat - mat1) < 1e-6


@lru_cache
def identity(qubit_number):
    return np.eye(2 ** qubit_number)


@lru_cache
def binary2string(intger, length=None):
    binary_string = bin(intger).replace('0b', '')
    if length is None or length < len(binary_string):
        length = len(binary_string)
    length = int(length)
    return '0' * (length - len(binary_string)) + binary_string


@lru_cache
def all_strings(qubit_number):
    return [(binary2string(base, qubit_number), base) for base in range(2 ** qubit_number)]


# def is_unitary(m):
#     return np.allclose(np.eye(len(m)), m.dot(m.T.conj()))

def matrix_distance_squared(A, B):
    """
    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    if isinstance(A, Operator):
        A = A.data
    if isinstance(B, Operator):
        B = B.data

    # optimized implementation
    return np.abs(1 - np.abs(np.sum(np.multiply(A, np.conj(B)))) / A.shape[0])


try:
    import tensorflow as tf
    from tensorflow import reduce_sum, transpose, shape, multiply
except:
    # traceback.print_exc()
    print('Warning: TF is not installed')
    tf = reduce_sum = transpose = shape = multiply = None


def conj_transpose_tf(mat):
    return tf.math.conj(transpose(mat))


def matrix_distance_squared_tf(A, B):
    B = tf.math.conj(B)
    abs = tf.abs
    multipe_result = multiply(A, B)

    # np_result = np.multiply(A.numpy(), B.numpy())
    # assert np.allclose(np_result, multipe_result.numpy())
    # assert np.allclose(reduce_sum(multipe_result).numpy(), np.sum(np_result))

    matrix_size = tf.cast(B.shape[0], dtype=tf.float64)
    distance = 1 - abs(reduce_sum(multipe_result)) / matrix_size
    return abs(distance)


def permute(mat, index):
    mat = MatrixOp(mat)
    mat = mat.permute(index).to_matrix()
    return mat


def random_order(num_bits):
    return list(np.random.permutation(num_bits))


def randomPermute(mat):
    num_bits = int(log2(mat.shape[0]))
    rand_index = random_order(num_bits)
    return permute(mat, rand_index)


def permute_qubit_order(qubit_order, permute_indexs):
    new_order = list(qubit_order)
    for i, index in enumerate(permute_indexs):
        new_order[i] = new_order[index]
    return new_order


def cleanMatrix(matrix):
    abs_matrix = np.abs(matrix)
    index1 = abs_matrix > 1 - 1e-5
    index0 = abs_matrix < 1e-5
    matrix[index1] = 1
    matrix[index0] = 0


def printMatrix(mat):
    if isinstance(mat, QuantumCircuit):
        op = Operator(mat)
        printMatrix(op)
    elif isinstance(mat, Operator):
        printMatrix(mat.data)
    else:
        print(np.round(mat, 2))
        print()


# 两个for CSD的算法，可以把separate的变成一个整的
# def recoverUV(uvs):
#     matrix_size = 0
#     for uv in uvs:
#         uv_size = uv.shape
#         assert len(uv_size) == 2 and uv_size[0] == uv_size[1], f'{uv} should be a matrix'
#         matrix_size += uv_size[0]

#     M = np.zeros((matrix_size, matrix_size), dtype=np.complex128)
#     i = 0
#     for uv in uvs:
#         uv_size = uv.shape[0]
#         M[i:i+uv_size,i:i+uv_size] = uv
#         i += uv_size
#     return M

def recoverUV(uvs):
    return block_diag(*uvs)


# 假设q = p
def recoverD(theta, m, p):
    # 假设p<=m-p
    p = int(p)
    assert p < m
    # Construct the middle factor CS
    c = np.diag(np.cos(theta))
    s = np.diag(np.sin(theta))
    r = min(p, m - p)  # 如果p<=m-p, 即在左半边这一部分一定都是0
    n11 = p - r  # p>r
    n12 = n21 = min(p, m - p) - r
    n22 = m - p - r
    Id = np.eye(np.max([n11, n12, n21, n22, r]), dtype=theta.dtype)
    CS = np.zeros((m, m), dtype=theta.dtype)
    # printMatrix(CS)

    CS[:n11, :n11] = Id[:n11, :n11]
    # printMatrix(CS)

    xs = n11 + r
    xe = n11 + r + n12
    ys = n11 + n21 + n22 + 2 * r
    ye = n11 + n21 + n22 + 2 * r + n12
    CS[xs: xe, ys:ye] = -Id[:n12, :n12]
    # printMatrix(CS)

    xs = p + n22 + r
    xe = p + n22 + r + n21
    ys = n11 + r
    ye = n11 + r + n21
    CS[xs:xe, ys:ye] = Id[:n21, :n21]
    # printMatrix(CS)

    CS[p:p + n22, p:p + n22] = Id[:n22, :n22]
    # printMatrix(CS)
    CS[n11:n11 + r, n11:n11 + r] = c  # 如果划在左半边， 左上角,r个
    # printMatrix(CS)
    CS[p + n22:p + n22 + r, r + n21 + n22:2 * r + n21 + n22] = c
    # printMatrix(CS)

    xs = n11
    xe = n11 + r
    ys = n11 + n21 + n22 + r
    ye = n11 + n21 + n22 + 2 * r
    CS[xs:xe, ys:ye] = -s
    # printMatrix(CS)

    CS[p + n22:p + n22 + r, n11:n11 + r] = s
    # printMatrix(CS)

    return CS


def cnot_count(qc):
    qc = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=3)
    count_ops = qc.count_ops()
    if 'cx' in count_ops:
        return count_ops['cx']
    return 0


def permute(mat, orders: list):
    '''
        orders中的j比特会被放到新的矩阵的order[j]的位置
    '''
    mat = MatrixOp(mat)
    mat = mat.permute(orders).to_matrix()
    return mat


# 把大小端的电路换一下
def reverse_unitary(mat, n):
    qubits = list(range(n))
    qubits.reverse()
    return permute(mat, qubits)
