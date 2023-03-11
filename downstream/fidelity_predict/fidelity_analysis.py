from collections import defaultdict
import random

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
    def train(self, train_dataset, test_dataset = None, epoch_num = 50, ):
        upstream_model = self.upstream_model
        # backend = self.upstream_model.backend
        
        params = jnp.zeros(shape=(len(upstream_model.device2path_table), upstream_model.max_table_size))
        # 每个device一行
        
        optimizer = optax.adamw(learning_rate=1e-2)
        opt_state = optimizer.init(params)

        if test_dataset is not None:
            min_test_loss = 1e10
            best_test_params = None

        self.path_count = defaultdict(int)
        n_instruction2circuit_infos, gate_nums = get_n_instruction2circuit_infos(train_dataset)
        print(gate_nums)
        random_gate_nums = list(gate_nums) + list(gate_nums)
        random.shuffle(random_gate_nums)
        for gate_num in gate_nums + random_gate_nums:
            if(gate_num > 150):
                continue
            best_loss_value = 1e10
            best_params = None
            
            n_iter_no_change=10
            no_change_tolerance=1e-3
            former_loss = 1
            loss_decrease_history = []
            
            for circuit_info in n_instruction2circuit_infos[gate_num]:
                for gate_paths in circuit_info['gate_paths']:
                    for path in gate_paths:
                        self.path_count[path] += 1
                        
            for epoch in range(epoch_num):
                loss_values = []
                # print(n_instruction2circuit_infos[gate_num])
                n_instruction2circuit_infos[gate_num] = np.array(n_instruction2circuit_infos[gate_num])
                            
                for circuit_infos in batch(n_instruction2circuit_infos[gate_num], batch_size=100, should_shuffle = True):
                    # print(circuit_infos[0].keys())
                                
                    X = np.array([circuit_info['vecs'] for circuit_info in circuit_infos], dtype=np.float32)
                    Y = np.array([circuit_info['ground_truth_fidelity'] for circuit_info in circuit_infos], dtype=np.float32)
                    devices = []
                    for circuit_info in circuit_infos:
                        circuit_devices = []
                        for gate in circuit_info['gates']:
                            device = extract_device(gate)
                            device_index = list(upstream_model.device2path_table.keys()).index(device)
                            circuit_devices.append(device_index)
                        devices.append(circuit_devices)
                    devices = jnp.array(devices, dtype=jnp.int32)
                         
                    loss_value, gradient = jax.value_and_grad(batch_loss)(params, X, Y, devices)
                    updates, opt_state = optimizer.update(gradient, opt_state, params)
                    params = optax.apply_updates(params, updates)

                    params = params.at[params > error_param_rescale / 10].set(error_param_rescale / 10)  # 假设一个特征对error贡献肯定小于0.1
                    params = params.at[params < 0].set(0)

                    loss_values.append(loss_value)

                mean_loss = np.array(loss_values).mean()

                loss_decrease_history.append(best_loss_value-mean_loss)
                # former_loss = mean_loss
                if epoch < n_iter_no_change:
                    loss_no_change = False
                else:
                    loss_no_change = True
                    for loss_decrement in loss_decrease_history[-n_iter_no_change:]:
                        if loss_decrement > no_change_tolerance:
                            loss_no_change = False
                if loss_no_change:
                    break
                
                if mean_loss < best_loss_value:
                    best_loss_value = mean_loss
                    best_params = params

                if epoch % 10 == 0:
                    print(f'gate num: {gate_num}, epoch: {epoch}, circuit num: {len(n_instruction2circuit_infos[gate_num])}, train mean loss: {mean_loss}')

            params = best_params
            
            if test_dataset is not None:
                test_loss = 0 
                for circuit_info in test_dataset[:2000]:
                    if circuit_info['ground_truth_fidelity'] < 0.2:
                        continue
                    
                    circuit_devices = []
                    for gate in circuit_info['gates']:
                        device = extract_device(gate)
                        device_index = list(upstream_model.device2path_table.keys()).index(device)
                        circuit_devices.append(device_index)
                    circuit_devices = jnp.array(circuit_devices, dtype=jnp.int32)
                    test_loss += loss_func(best_params, np.array(circuit_info['vecs'], dtype=np.float32), circuit_info['ground_truth_fidelity'], circuit_devices)

                print('test loss:', test_loss)
                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    best_test_params = params
                

        print(f'taining error params finishs')

        if test_dataset is not None:
            self.error_params = best_test_params
        else:
            self.error_params = params


    def predict_fidelity(self, circuit_info):
        error_params = self.error_params


        device_list = list(self.upstream_model.device2path_table.keys())
        circuit_devices = []
        for gate in circuit_info['gates']:
            device = extract_device(gate)
            # if not isinstance(device,tuple):
            #     device = (device, -1)
            device_index = device_list.index(device)
            circuit_devices.append(device_index)
        circuit_devices = np.array(circuit_devices)
        vecs = np.array(circuit_info['vecs'])
        circuit_predict = smart_predict(error_params, vecs, circuit_devices)
        circuit_info['circuit_predict'] = circuit_predict
        return circuit_predict


def gate_error(device2params, vec, device):
    # error = jnp.dot((device2params[device] / error_param_rescale)**2, vec)
    error = jnp.dot((device2params[device] / error_param_rescale), vec)
    return error


@jax.jit
def smart_predict(device2params, vecs, devices):
    '''预测电路的保真度'''
    errors = vmap(gate_error, in_axes=(None, 0, 0), out_axes=0)(device2params, vecs, devices)
    return jnp.product(1 - errors, axis=0)  # 不知道为什么是[0][0]

@jax.jit
def loss_func(device2params, vecs, true_fidelity, devices):
    # predict_fidelity = naive_predict(circ) # 对于电路比较浅的预测是准的，大概是因为有可能翻转回去吧，所以电路深了之后会比理论的保真度要高一些
    predict_fidelity = smart_predict(device2params, vecs, devices)
    return optax.l2_loss(true_fidelity - predict_fidelity) * 100


def batch_loss(params, X, Y, devices):
    losses = vmap(loss_func, in_axes=(None, 0, 0, 0), out_axes=0)(params, X, Y, devices)
    return jnp.array(losses).mean()


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
