from collections import defaultdict
from operator import index
import random
from upstream.randomwalk_model import RandomwalkModel, add_pattern_error, Step, Path
from simulator.noise_free_simulator import simulate_noise_free
from simulator.noise_simulator import *
from qiskit import QuantumCircuit, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from utils.backend_info import coupling_map, initial_layout, max_qubit_num, default_basis_gates, single_qubit_fidelity, two_qubit_fidelity, readout_error
from qiskit.quantum_info.analysis import hellinger_fidelity
from dataset.random_circuit import one_layer_random_circuit, random_circuit
from downstream.fidelity_predict.other import naive_predict
from upstream.dimensionality_reduction import batch
import numpy as np
from dataset.dataset_loader import load_algorithms, _gen_random_circuits

# 还得传参数进去

from sklearn.utils import shuffle

from jax import numpy as jnp
import jax
from jax import grad, jit, vmap, pmap
import optax

# 给每个比特的每个门一个模型吧

# var1 = tf.Variable(10.0)

model_path = 'rwm_5qubit.pkl'

model = RandomwalkModel.load(model_path)

dataset = model.dataset #[:200]

# for circuit_info in algorithm:
#     qiskit_circuit = circuit_info['qiskit_circuit']
#     # qiskit_circuit.measure_all()
#     error_circuit, n_erroneous_patterns = add_pattern_error(circuit_info, model)
#     error_circuit.measure_all()
#     circuit_info['error_circuit'] = error_circuit
#     circuit_info['n_erroneous_patterns'] = n_erroneous_patterns

# algorithm = [circuit_info for circuit_info in algorithm if circuit_info['n_erroneous_patterns'] != 0]

# print('start simulate')
# results = [simulate_noise_free(circuit_info['error_circuit']) for circuit_info in algorithm]
# true_result = {
#     '0'*max_qubit_num: 2000
# }
# for i, circuit_info in enumerate(algorithm):
#     circuit_info['error_result'] = results[i]
#     circuit_info['ground_truth_fidelity'] = hellinger_fidelity(circuit_info['error_result'], true_result)

# model.save(model_path)

dataset = [circuit_info for circuit_info in dataset if 'error_result' in circuit_info]  # TODO：不要加，正样本也是有用的

# def to_regular(value):
#     return np.array(value.primal)

# def parse_parameter(params):
#     path2error = {}
#     for index, path in enumerate(model.hash_table.keys()):
#         # print(path, params[index])
#         path2error[path] = to_regular(params[index])
#     return path2error

# paths = list(model.hash_table.keys())
# random.shuffle(paths)
# print(paths[:100])

# @jax.jit
def smart_predict(params, reduced_vecs):
    '''预测电路的保真度'''
    errors = vmap(lambda params, reduced_vec: jnp.dot(params/1000, reduced_vec), in_axes=(None, 0), out_axes=0)(params, reduced_vecs)
    return jnp.product(1-errors, axis=0)[0][0] # 不知道为什么是[0][0]
    # return 1-jnp.sum([params[index] for index in sparse_vector])
    # np.array(sum(error).primal)

def loss(params, reduced_vecs, true_fidelity):
    # predict_fidelity = naive_predict(circ) # 对于电路比较浅的预测是准的，大概是因为有可能翻转回去吧，所以电路深了之后会比理论的保真度要高一些
    predict_fidelity = smart_predict(params, reduced_vecs)
    # print(predict_fidelity)
    # print(true_fidelity)
    return optax.l2_loss(true_fidelity - predict_fidelity)*100


def batch_loss(params, X, Y):
    losses = vmap(loss, in_axes=(None, 0, 0), out_axes=0)(params, X, Y)
    return losses.mean()

# def batch_loss(params, X, Y):
#     total_loss = 0
#     for x,y in zip(X, Y):
#         total_loss += loss(params, x, y)
#     return (total_loss/len(X))[0]

reduce_vec_size = 100
params = jax.random.normal(shape=(1, reduce_vec_size), key=jax.random.PRNGKey(0))
params = jnp.zeros(shape=(1, reduce_vec_size))

optimizer = optax.adamw(learning_rate=1e-2)
opt_state = optimizer.init(params)

# TODO：一个好的初始值
# 设置初始值, 也就是本来的
# for qubit1 in range(max_qubit_num):
#     for gate in basis_single_gates:
#         _path = f"loop-{gate}-{qubit1}')"
#         if model.has_path(_path):
#             params = params.at[model.path_index(_path)].set(1-single_qubit_fidelity[qubit1])
#     for qubit2 in range(max_qubit_num):
#         for gate in basis_two_gates:
#             _path = f"loop-{gate}-{qubit1}-{qubit2}')"
#             if model.has_path(_path):
#                 params = params.at[model.path_index(_path)].set(1-two_qubit_fidelity[qubit1])


# 增加一系列的pretrain
pretrain_dataset = []
# 加一些前面是一层单比特门假设保真度是固定的
n_instruction2pretrain_circuit_infos = defaultdict(list)
for n_gates in range(1, 3, ): #(生成一些最多两个比特的)
    n_instruction2pretrain_circuit_infos[n_gates] = _gen_random_circuits(n_qubits = max_qubit_num, n_gates = n_gates, two_qubit_prob = 0.2, n_circuits = 1000, reverse=False)
    pretrain_dataset += n_instruction2pretrain_circuit_infos[n_gates]

for circuit_info in pretrain_dataset:
    qiskit_circuit = circuit_info['qiskit_circuit']
    circuit_info = model.vectorize(circuit_info)

    # one_layer = one_layer_random_circuit(qiskit_circuit.num_qubits)
    # exeucted_circuit = one_layer.compose(qiskit_circuit)
    exeucted_circuit = qiskit_circuit
    exeucted_error_circuit, n_erroneous_patterns = add_pattern_error(circuit_info, model)
    exeucted_circuit.measure_all()
    exeucted_error_circuit.measure_all()

    true_result = simulate_noise_free(exeucted_circuit)
    error_result = simulate_noise_free(exeucted_error_circuit)

    circuit_info['ground_truth_fidelity'] = hellinger_fidelity(error_result, true_result)

    # circuit_info['ground_truth_fidelity'] = .99 *  len(qiskit_circuit.data) # 可以直接用predict的 , 1

for epoch in range(10):
    loss_values = []
    for gate_num in n_instruction2pretrain_circuit_infos:
        for circuit_infos in batch(n_instruction2pretrain_circuit_infos[gate_num], batch_size = 100):
            # X = [circuit_info['qiskit_circuit'] for circuit_info in circuit_infos]
            # X = [np.array(circuit_info['instruction2reduced_vecs'], dtype=np.float32) for circuit_info in circuit_infos]
            X = np.array([ circuit_info['instruction2reduced_vecs'] for circuit_info in circuit_infos], dtype=np.float32) 
            Y = np.array([ [circuit_info['ground_truth_fidelity']] for circuit_info in circuit_infos], dtype=np.float32)

            loss_value, gradient = jax.value_and_grad(batch_loss)(params, X, Y)
            updates, opt_state = optimizer.update(gradient, opt_state, params)
            params = optax.apply_updates(params, updates)

            loss_values.append(loss_value)

            params = params.at[params > 100].set(100)
            params = params.at[params < 0].set(0)

    mean_loss = np.array(loss_values).mean()
    print(epoch, mean_loss)

# for circuit_info in pretrain_circuit_infos:
#     x = circuit_info['instruction2reduced_vecs']
    # print(smart_predict(params, x), circuit_info['ground_truth_fidelity'])

# TODO：不同深度的分开来计算
n_instruction2circuit_infos = defaultdict(list)
for circuit_info in dataset:
    qiskit_circuit = circuit_info['qiskit_circuit']
    gate_num = len(circuit_info['gates'])
    n_instruction2circuit_infos[gate_num].append(circuit_info)

# vgf = jax.value_and_grad(batch_loss)

n_instructions = list(n_instruction2circuit_infos.keys())
n_instructions.sort()
print(n_instructions)
for n_instruction in n_instructions:
    for epoch in range(10):
        loss_values = []
        # for gate_num in n_instruction2circuit_infos:
        for circuit_infos in batch(n_instruction2circuit_infos[gate_num], batch_size = 100):
            # X = [circuit_info['qiskit_circuit'] for circuit_info in circuit_infos]
            # X = [np.array(circuit_info['instruction2reduced_vecs'], dtype=np.float32) for circuit_info in circuit_infos]
            X = np.array([ circuit_info['instruction2reduced_vecs'] for circuit_info in circuit_infos], dtype=np.float32) 
            Y = np.array([ [circuit_info['ground_truth_fidelity']] for circuit_info in circuit_infos], dtype=np.float32)

            # for y in Y:
            #     print(smart_predict(params, y))
            # print(smart_predict(params, X[0]))
            loss_value, gradient = jax.value_and_grad(batch_loss)(params, X, Y)
            updates, opt_state = optimizer.update(gradient, opt_state, params)
            params = optax.apply_updates(params, updates)

            loss_values.append(loss_value)

            params = params.at[params > 100].set(100)
            params = params.at[params < 0].set(0)

        mean_loss = np.array(loss_values).mean()

        print(epoch, mean_loss)
