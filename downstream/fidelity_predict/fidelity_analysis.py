from collections import defaultdict

from qiskit import QuantumCircuit

from upstream.dimensionality_reduction import batch
import numpy as np
from jax import numpy as jnp
import jax
from jax import vmap
import optax

from upstream.randomwalk_model import extract_device
from upstream import RandomwalkModel

error_param_rescale = 10000


class FidelityModel():
    # TODO: 传入backend
    def __init__(self, upstream_model: RandomwalkModel = None):
        self.error_params = None
        self.upstream_model = upstream_model
        return

    # device2reverse_path_table_size去掉，从backend拿
    def train(self, train_dataset, epoch_num = 30):
        upstream_model = self.upstream_model
        # backend = self.upstream_model.backend
        
        params = jnp.zeros(shape=(len(upstream_model.device2path_table), upstream_model.max_table_size))
        # 每个device一行
        
        # defaultdict(lambda: jnp.array(0))
        # for device, dim in device2reverse_path_table_size.items():
        #     device2error_params[device] = jnp.zeros(shape=(1, dim))

        optimizer = optax.adamw(learning_rate=1e-2)
        opt_state = optimizer.init(params)
        train_dataset = train_dataset

        # 如果同时训练的数组大小不一致没办法使用vmap加速
        n_instruction2circuit_infos, gate_nums = get_n_instruction2circuit_infos(train_dataset)
        print(gate_nums)
        for gate_num in gate_nums:
            if(gate_num > 150):
                continue
            best_loss_value = 1e10
            best_params = None
            for epoch in range(epoch_num):
                loss_values = []
                # print(n_instruction2circuit_infos[gate_num])
                n_instruction2circuit_infos[gate_num] = np.array(n_instruction2circuit_infos[gate_num])
                for circuit_infos in batch(n_instruction2circuit_infos[gate_num], batch_size=100, should_shuffle = True):
                    # print(circuit_infos[0].keys())
                    X = np.array([circuit_info['vecs'] for circuit_info in circuit_infos], dtype=np.float32)
                    Y = np.array([circuit_info['ground_truth_fidelity'] for circuit_info in circuit_infos], dtype=np.float32)
                    # devices = np.array([[extract_device(gate) for gate in circuit_info['gates']] for circuit_info in circuit_infos], dtype=np.int32)
                    devices = []
                    for circuit_info in circuit_infos:
                        circuit_devices = []
                        for gate in circuit_info['gates']:
                            device = extract_device(gate)
                            if isinstance(device,tuple):
                                device = (circuit_info['map'][device[0]],circuit_info['map'][device[1]])
                            else:
                                device = circuit_info['map'][device]
                            device_index = list(upstream_model.device2path_table.keys()).index(device)
                            circuit_devices.append(device_index)
                        devices.append(circuit_devices)
                    devices = jnp.array(devices, dtype=jnp.int32)
                         
                    # circuit_infos = np.array(circuit_infos)
                    loss_value, gradient = jax.value_and_grad(batch_loss)(params, X, Y, devices)
                    updates, opt_state = optimizer.update(gradient, opt_state, params)
                    params = optax.apply_updates(params, updates)

                    # for device, param in params.items():
                    params = params.at[params > error_param_rescale / 10].set(error_param_rescale / 10)  # 假设一个特征对error贡献肯定小于0.1
                    params = params.at[params < 0].set(0)

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

        self.error_params = params
        # _error_params = {}
        # for device, error_param in error_params.items():
        #     _error_params[device] = np.array(error_params[device])
        # self.error_params = _error_params

    def predict_fidelity(self, circuit_info):
        error_params = self.error_params


        device_list = list(self.upstream_model.device2path_table.keys())
        circuit_devices = []
        for gate in circuit_info['gates']:
            device = extract_device(gate)
            if isinstance(device,tuple):
                device = (circuit_info['map'][device[0]],circuit_info['map'][device[1]])
            else:
                device = circuit_info['map'][device]
            device_index = device_list.index(device)
            circuit_devices.append(device_index)
        circuit_devices = np.array(circuit_devices)
        vecs = np.array(circuit_info['vecs'])
        circuit_predict = smart_predict(error_params, vecs, circuit_devices)
        # gate_errors = np.array([
        #     jnp.dot(error_params[extract_device(circuit_info['gates'][idx])] / error_param_rescale, vec)
        #     for idx, vec in enumerate(circuit_info['vecs'])
        # ])[:, 0]
        # circuit_info['gate_errors'] = gate_errors
        circuit_info['circuit_predict'] = circuit_predict
        return circuit_predict


def gate_error(device2params, vec, device):
    error = jnp.dot(device2params[device] / error_param_rescale, vec)
    return error


@jax.jit
def smart_predict(device2params, vecs, devices):
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

    errors = vmap(gate_error, in_axes=(None, 0, 0), out_axes=0)(device2params, vecs, devices)
    return jnp.product(1 - errors, axis=0)  # 不知道为什么是[0][0]


def loss_func(device2params, vecs, true_fidelity, devices):
    # predict_fidelity = naive_predict(circ) # 对于电路比较浅的预测是准的，大概是因为有可能翻转回去吧，所以电路深了之后会比理论的保真度要高一些
    predict_fidelity = smart_predict(device2params, vecs, devices)
    # print(predict_fidelity)
    # print(true_fidelity)
    return optax.l2_loss(true_fidelity - predict_fidelity) * 100


def batch_loss(params, X, Y, devices):
    losses = vmap(loss_func, in_axes=(None, 0, 0, 0), out_axes=0)(params, X, Y, devices)

    # losses = []
    # for x, y, circuit_info in zip(X, Y, circuit_infos):
    #     losses.append(loss_func(params, x, y, circuit_info))
    return jnp.array(losses).mean()


# 在训练中逐渐增加gate num
# def epoch_train(circuit_infos, params, opt_state, optimizer, device2reverse_path_table_size):

#     return loss_value, params, opt_state


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
