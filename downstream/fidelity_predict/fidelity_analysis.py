from collections import defaultdict

from qiskit import QuantumCircuit

from upstream.dimensionality_reduction import batch
import numpy as np
from jax import numpy as jnp
import jax
from jax import vmap
import optax

from upstream.randomwalk_model import extract_device

error_param_rescale = 10000


class FidelityModel():
    def __init__(self):
        self.error_params = None
        return

    def train(self, dataset, device2reverse_path_table_size):
        device2error_params = defaultdict(lambda: jnp.array(0))
        for device, dim in device2reverse_path_table_size.items():
            device2error_params[device] = jnp.zeros(shape=(1, dim))

        optimizer = optax.adamw(learning_rate=1e-2)
        opt_state = optimizer.init(device2error_params)
        train_dataset = dataset
        error_params, opt_state = train_error_params(train_dataset, device2error_params, opt_state, optimizer,
                                                     device2reverse_path_table_size, epoch_num=30)

        _error_params = {}
        for device, error_param in error_params.items():
            _error_params[device] = np.array(error_params[device])
        self.error_params = _error_params

    def predict_fidelity(self, circuit_info):
        error_params = self.error_params
        circuit_predict = smart_predict(error_params, circuit_info['vecs'], circuit_info)
        gate_errors = np.array([
            jnp.dot(error_params[extract_device(circuit_info['gates'][idx])] / error_param_rescale, vec)
            for idx, vec in enumerate(circuit_info['vecs'])
        ])[:, 0]
        circuit_info['gate_errors'] = gate_errors
        circuit_info['circuit_predict'] = circuit_predict
        return circuit_predict, circuit_info, gate_errors


def train_error_params(train_dataset, params, opt_state, optimizer, device2reverse_path_table_size, epoch_num=10, ):
    # 如果同时训练的数组大小不一致没办法使用vmap加速
    n_instruction2circuit_infos, gate_nums = get_n_instruction2circuit_infos(train_dataset)
    print(gate_nums)
    for gate_num in gate_nums:
        best_loss_value = 1e10
        best_params = None
        for epoch in range(epoch_num):
            loss_values = []
            # print(n_instruction2circuit_infos[gate_num])
            n_instruction2circuit_infos[gate_num] = np.array(n_instruction2circuit_infos[gate_num])
            for circuit_infos in batch(n_instruction2circuit_infos[gate_num], batch_size=100):
                loss_value, params, opt_state = epoch_train(circuit_infos, params, opt_state, optimizer,
                                                            device2reverse_path_table_size)
                loss_values.append(loss_value)

            mean_loss = np.array(loss_values).mean()
            if mean_loss < best_loss_value:
                best_loss_value = mean_loss
                best_params = params

            if epoch % 10 == 0:
                print(f'gate num: {gate_num}, epoch: {epoch}, train mean loss: {mean_loss}')

        params = best_params

        # test_mean_losses = []
        # for circuit_info in test_dataset:
        #     test_x = np.array(circuit_info['reduced_vecs'], dtype=np.float32)
        #     test_y = np.array([circuit_info['ground_truth_fidelity']], dtype=np.float32)
        #     test_mean_loss = loss_func(best_params, test_x, test_y)
        #     test_mean_losses.append(test_mean_loss)

        # test_mean_loss = np.array(test_mean_losses).mean()
        # losses_hisotry.append(test_mean_loss)
        # gate_num_loss[gate_num] = test_mean_loss
        # print(gate_num, )
        # print(f'gate num: {gate_num}, test mean loss: {test_mean_loss}')

    print(f'taining error params finishs')
    return params, opt_state


def test(device2params, vec, device):
    error = jnp.dot(device2params[jnp.array(device)[0], jnp.array(device)[1]] / error_param_rescale, vec)
    return error


@jax.jit
def smart_predict(device2params, vecs, devices, device2reverse_path_table_size):
    '''预测电路的保真度'''
    predict = 1.0

    # for idx, vec in enumerate(vecs):
    #     device = devices[idx]
    #     path_table_size = device2reverse_path_table_size[device]
    #     error = jnp.dot(device2params[device] / error_param_rescale, vec[:path_table_size])[0]
    #     predict *= 1 - error
    # predict *=


    # errors = vmap(lambda device2params, vec, device: jnp.dot(device2params[tuple(jnp.array(device)[0],jnp.array(device)[1])] / error_param_rescale, vec),
    #               in_axes=(None, 0, 0), out_axes=0)(device2params, vecs, devices)

    errors = vmap(test, in_axes=(None, 0, 0), out_axes=0)(device2params, vecs, devices)
    return jnp.product(1 - errors, axis=0)  # 不知道为什么是[0][0]


def loss_func(device2params, vecs, true_fidelity, devices, device2reverse_path_table_size):
    # predict_fidelity = naive_predict(circ) # 对于电路比较浅的预测是准的，大概是因为有可能翻转回去吧，所以电路深了之后会比理论的保真度要高一些
    predict_fidelity = smart_predict(device2params, vecs, devices, device2reverse_path_table_size)
    # print(predict_fidelity)
    # print(true_fidelity)
    return optax.l2_loss(true_fidelity - predict_fidelity) * 100


def batch_loss(params, X, Y, devices, device2reverse_path_table_size):
    losses = vmap(loss_func, in_axes=(None, 0, 0, 0, None), out_axes=0)(params, X, Y, devices,
                                                                        device2reverse_path_table_size)

    # losses = []
    # for x, y, circuit_info in zip(X, Y, circuit_infos):
    #     losses.append(loss_func(params, x, y, circuit_info))
    return jnp.array(losses).mean()


# 在训练中逐渐增加gate num
def epoch_train(circuit_infos, params, opt_state, optimizer, device2reverse_path_table_size):
    # print(circuit_infos[0].keys())
    X = np.array([circuit_info['vecs'] for circuit_info in circuit_infos], dtype=np.float32)
    Y = np.array([circuit_info['ground_truth_fidelity'] for circuit_info in circuit_infos], dtype=np.float32)
    devices = np.array([[extract_device(gate) for gate in circuit_info['gates']] for circuit_info in circuit_infos],
                       dtype=np.int32)
    # circuit_infos = np.array(circuit_infos)
    loss_value, gradient = jax.value_and_grad(batch_loss)(params, X, Y, devices, device2reverse_path_table_size)
    updates, opt_state = optimizer.update(gradient, opt_state, params)
    params = optax.apply_updates(params, updates)

    for device, param in params.items():
        param = param.at[param > error_param_rescale / 10].set(error_param_rescale / 10)  # 假设一个特征对error贡献肯定小于0.1
        param = param.at[param < 0].set(0)

    return loss_value, params, opt_state


def get_n_instruction2circuit_infos(dataset):
    n_instruction2circuit_infos = defaultdict(list)
    for circuit_info in dataset:
        # qiskit_circuit = circuit_info['qiskit_circuit']
        gate_num = len(circuit_info['gates'])
        n_instruction2circuit_infos[gate_num].append(circuit_info)

    # print(n_instruction2circuit_infos[gate_num])
    gate_nums = list(n_instruction2circuit_infos.keys())
    gate_nums.sort()

    return n_instruction2circuit_infos, gate_nums
