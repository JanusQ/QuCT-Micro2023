
from collections import defaultdict
from operator import index
from upstream.randomwalk_model import RandomwalkModel, Step
import matplotlib.pyplot as plt
from utils.backend_info import max_qubit_num
from downstream.fidelity_predict.other import naive_predict
from upstream.dimensionality_reduction import batch
import numpy as np

from sklearn.utils import shuffle

from jax import numpy as jnp
import jax
from jax import grad, jit, vmap, pmap
import optax


model_path = 'rwm_5qubit.pkl'

model = RandomwalkModel.load(model_path)
dataset = model.dataset #[:200]

# 整一个naive点的方法
error_param_rescale = 10000


def get_n_instruction2circuit_infos(dataset):
    n_instruction2circuit_infos = defaultdict(list)
    for circuit_info in dataset:
        qiskit_circuit = circuit_info['qiskit_circuit']
        gate_num = len(circuit_info['gates'])
        n_instruction2circuit_infos[gate_num].append(circuit_info)

    # print(n_instruction2circuit_infos[gate_num])
    gate_nums = list(n_instruction2circuit_infos.keys())
    gate_nums.sort()

    return n_instruction2circuit_infos, gate_nums

# qubits = [qubit.index for qubit in instruction.qubits]

@jax.jit
def naive_predict(naive_params, instructions):
    cal_fidelity = lambda qubits: jnp.where(qubits[1] == -1, naive_params['single'][qubits[0]] , naive_params['double'][qubits[0]][qubits[1]]) / error_param_rescale

    # temp = cal_fidelity(instructions[0])
    fidelity = jnp.product(vmap(cal_fidelity, in_axes=(0,), out_axes=0)(instructions), axis=0)
    # for qubits in instructions:
    #     # if qubits[1] == -1:
    #     #     # fidelity *= naive_params['single'][qubits[0]] / error_param_rescale
    #     #     pass
    #     # else:
    #     #     # fidelity *= naive_params['double'][qubits[0]][qubits[1]] / error_param_rescale
    #     _fidelity = naive_params['double'][qubits[0]][qubits[1]] / error_param_rescale
    #     _temp = 0
    return fidelity

def naive_loss(naive_params, instructions, true_fidelity):
    predict_fidelity = naive_predict(naive_params, instructions)
    return optax.l2_loss(true_fidelity - predict_fidelity)*100

def naive_batch_loss(naive_params, X, Y):
    # losses = jnp.array([naive_loss(naive_params, x, y) for x, y in zip(X, Y)])
    losses = vmap(naive_loss, in_axes=(None, 0, 0), out_axes=0)(naive_params, X, Y)
    return losses.mean()

# 在训练中逐渐增加gate num
def naive_epoch_train(circuit_infos, naive_params, naive_opt_state, naive_optimizer):
    # print(circuit_infos[0].keys())

    # print(circuit_infos[0]['qiskit_circuit'])

    X = np.array([[   
            [qubit.index for qubit in instruction.qubits] if len(instruction.qubits) == 2 else [qubit.index for qubit in instruction.qubits ] + [-1]
                for instruction in circuit_info['gates']
                    # if len(instruction.qubits) == 2
        ] for circuit_info in circuit_infos])

    Y = np.array([ [circuit_info['ground_truth_fidelity']] for circuit_info in circuit_infos], dtype=np.float32)

    loss_value, gradient = jax.value_and_grad(naive_batch_loss)(naive_params, X, Y)
    updates, naive_opt_state = naive_optimizer.update(gradient, naive_opt_state, naive_params)
    naive_params = optax.apply_updates(naive_params, updates)

    naive_params['single'] = naive_params['single'].at[naive_params['single'] < 1/error_param_rescale].set(1/error_param_rescale)  # 假设一个特征对error贡献肯定小于0.1
    naive_params['single'] = naive_params['single'].at[naive_params['single'] < 0].set(0)

    naive_params['double'] = naive_params['double'].at[naive_params['double'] < 1/error_param_rescale].set(1/error_param_rescale)  # 假设一个特征对error贡献肯定小于0.1
    naive_params['double'] = naive_params['double'].at[naive_params['double'] < 0].set(0)

    return loss_value, naive_params, naive_opt_state

def naive_train(dataset, naive_params, naive_opt_state, naive_optimizer, epoch_num = 10):
    # 如果同时训练的数组大小不一致没办法使用vmap加速
    n_instruction2circuit_infos, gate_nums = get_n_instruction2circuit_infos(dataset)
    print(gate_nums)
    for gate_num in gate_nums:
        best_loss_value = 1e10
        best_params = None
        for epoch in range(epoch_num):
            loss_values = []
            for circuit_infos in batch(n_instruction2circuit_infos[gate_num], batch_size = 100):
                loss_value, naive_params, opt_state = naive_epoch_train(circuit_infos, naive_params, naive_opt_state, naive_optimizer)
                loss_values.append(loss_value)
                
            mean_loss = np.array(loss_values).mean()
            if mean_loss < best_loss_value:
                best_loss_value = mean_loss
                best_params = naive_params
            
            if epoch % 10 == 0:
                print(f'gate num: {gate_num}, epoch: {epoch}, mean loss: {mean_loss}')
        
        naive_params = best_params
    
    print(f'taining finishs')
    return best_loss_value, naive_params, opt_state

# 每个比特都只有单比特和多比特保真度这一个参数
naive_params = {
    'single': np.ones((max_qubit_num, 1)) * error_param_rescale,
    'double': np.ones((max_qubit_num, max_qubit_num, 1)) * error_param_rescale
}

naive_optimizer = optax.adamw(learning_rate=1e-2)
naive_opt_state = naive_optimizer.init(naive_params)

naive_loss_value, naive_params, naive_opt_state = naive_train(dataset, naive_params, naive_opt_state, naive_optimizer, epoch_num = 10)

print(naive_params)
