import random
from collections import defaultdict

import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import vmap
from sklearn.model_selection import train_test_split

from upstream import RandomwalkModel
from upstream.dimensionality_reduction import batch
from upstream.randomwalk_model import extract_device

error_param_rescale = 10000  # TODO: 再大一些


class FidelityModel():
    # TODO: 传入backend
    def __init__(self, upstream_model: RandomwalkModel = None):
        self.error_params = None
        self.upstream_model = upstream_model
        return

    # device2reverse_path_table_size去掉，从backend拿
    def train(self, train_dataset, validation_dataset = None, epoch_num = 100, verbose = True):
        upstream_model = self.upstream_model
        # backend = self.upstream_model.backend

        params = {
            'gate_params': jnp.zeros(shape=(len(upstream_model.device2path_table), upstream_model.max_table_size)),
            'circuit_bias': jnp.zeros(shape=(1,)),
        }
        # params = jnp.zeros(shape=(len(upstream_model.device2path_table), upstream_model.max_table_size))  #  + 1
        # 每个device一行
        
        optimizer = optax.adamw(learning_rate=1e-2)
        opt_state = optimizer.init(params)


        for circuit_info in train_dataset:
            circuit_devices = []
            for gate in circuit_info['gates']:
                device = extract_device(gate)
                if 'map' in circuit_info:
                    if isinstance(device,tuple):
                        device = (circuit_info['map'][device[0]],circuit_info['map'][device[1]])
                    else:
                        device = circuit_info['map'][device]
                device_index = list(upstream_model.device2path_table.keys()).index(device)
                circuit_devices.append(device_index)
            circuit_devices = jnp.array(circuit_devices, dtype=jnp.int32)
            circuit_info['circuit_devices'] = circuit_devices
    

        if validation_dataset is None:
            train_dataset, validation_dataset  = train_test_split(train_dataset, test_size = 200)
        
        # self.path_count = defaultdict(int)
        n_instruction2circuit_infos, gate_nums = get_n_instruction2circuit_infos(train_dataset)
        # print(gate_nums)

        terminate_num_gate  = 150
        # for gate_num in gate_nums:
        #     if(gate_num >= terminate_num_gate):
        #         break
            
        #     real_fidelities = []
        #     for circuit_info in n_instruction2circuit_infos[gate_num]:
        #         real_fidelities.append(circuit_info['ground_truth_fidelity'])
                
        #         for gate_paths in circuit_info['gate_paths']:
        #             for path in gate_paths:
        #                 self.path_count[path] += 1
            
            # mean_real_fidelity = sum(real_fidelities) / len(real_fidelities)
            
            # if mean_real_fidelity < 0.2:  # 再小的质量太低了，可能没用
            #     terminate_num_gate = gate_num
            #     break
            
            # print(f'n_instruction2circuit_infos[{gate_num}] has', len(n_instruction2circuit_infos[gate_num]))
        if verbose:
            print('n_instruction2circuit_infos = ', {n: len(n_instruction2circuit_infos[n]) for n in n_instruction2circuit_infos})
        
        gate_nums = [gate_num for gate_num in gate_nums if gate_num <= terminate_num_gate]
        for gate_num in gate_nums:
            n_instruction2circuit_infos[gate_num] = np.array(n_instruction2circuit_infos[gate_num])
    
        min_loss = 1e10
        best_params = None
        loss_decrease_history = []
        n_iter_no_change = 10
        no_change_tolerance  = .1  # TODO: 可以改到.1
        
        for epoch in range(epoch_num):


            loss_values = []
            
            random_gate_nums = list(gate_nums) # + list(gate_nums)
            random.shuffle(random_gate_nums)

            for gate_num in gate_nums: # + random_gate_nums:                
                
                for circuit_infos in batch(n_instruction2circuit_infos[gate_num], batch_size=100, should_shuffle = True):
                    X = np.array([circuit_info['vecs'] for circuit_info in circuit_infos], dtype=np.float32)
                    Y = np.array([circuit_info['ground_truth_fidelity'] for circuit_info in circuit_infos], dtype=np.float32)
                    devices = []
                    for circuit_info in circuit_infos:
                        devices.append(circuit_info['circuit_devices'])
                    devices = jnp.array(devices, dtype=jnp.int32)
                        
                    loss_value, gradient = jax.value_and_grad(batch_loss)(params, X, Y, devices)
                    updates, opt_state = optimizer.update(gradient, opt_state, params)
                    params = optax.apply_updates(params, updates)

                    # for key, param in params.items():
                    params['gate_params'] = params['gate_params'].at[params['gate_params'] > error_param_rescale / 10].set(error_param_rescale / 10)  # 假设一个特征对error贡献肯定小于0.1
                    params['gate_params'] = params['gate_params'].at[params['gate_params'] < 0].set(0)

                    params['circuit_bias'] = params['circuit_bias'].at[params['circuit_bias'] > error_param_rescale / 5].set(error_param_rescale / 5)  # 假设一个特征对error贡献肯定小于0.1
                    params['circuit_bias'] = params['circuit_bias'].at[params['circuit_bias'] < -error_param_rescale / 5].set(-error_param_rescale / 5)

                    # params = params.at[params > error_param_rescale / 10].set(error_param_rescale / 10)  # 假设一个特征对error贡献肯定小于0.1
                    # params = params.at[params < 0].set(0)

                    loss_values.append(loss_value)

            test_loss = 0 
            for circuit_info in validation_dataset: #[:2000]:
                circuit_devices = []
                for gate in circuit_info['gates']:
                    device = extract_device(gate)
                    if 'map' in circuit_info:
                        if isinstance(device,tuple):
                            device = (circuit_info['map'][device[0]],circuit_info['map'][device[1]])
                        else:
                            device = circuit_info['map'][device]
                    device_index = list(upstream_model.device2path_table.keys()).index(device)
                    circuit_devices.append(device_index)
                circuit_devices = jnp.array(circuit_devices, dtype=jnp.int32)
                test_loss += loss_func(params, np.array(circuit_info['vecs'], dtype=np.float32), circuit_info['ground_truth_fidelity'], circuit_devices)

            loss_decrease_history.append(min_loss - test_loss)
            if epoch > n_iter_no_change:
                loss_no_change = True
                for loss_decrement in loss_decrease_history[-n_iter_no_change:]:
                    if loss_decrement > no_change_tolerance:
                        loss_no_change = False
                if loss_no_change:
                    break
            
            if test_loss < min_loss:
                min_loss = test_loss
                best_params = params

            if verbose:
                print(f'epoch: {epoch}, \t epoch loss = {sum(loss_values)}, \t test mean loss = {test_loss / len(validation_dataset)}, \t bias = {params["circuit_bias"]}')
                # with open (f'params_{epoch}.pkl', 'wb') as f:
                #     pickle.dump(params,f)
            # params = best_params
        
        self.error_params = best_params
        if verbose:
            print(f'taining error params finishs')
        return best_params
    
        # if validation_dataset is not None:
        #     self.error_params = best_test_params
        # else:
        #     self.error_params = params

    def finetune(self, train_dataset, params = None, epoch_num = 200, verbose = True, test_size =20):
        upstream_model = self.upstream_model
        # backend = self.upstream_model.backend


        # params = jnp.zeros(shape=(len(upstream_model.device2path_table), upstream_model.max_table_size))  #  + 1
        # 每个device一行
        assert params is not None 
        
        optimizer = optax.adamw(learning_rate=1e-2)
        opt_state = optimizer.init(params)


        for circuit_info in train_dataset:
            circuit_devices = []
            for gate in circuit_info['gates']:
                device = extract_device(gate)
                if 'map' in circuit_info:
                    if isinstance(device,tuple):
                        device = (circuit_info['map'][device[0]],circuit_info['map'][device[1]])
                    else:
                        device = circuit_info['map'][device]
                device_index = list(upstream_model.device2path_table.keys()).index(device)
                circuit_devices.append(device_index)
            circuit_devices = jnp.array(circuit_devices, dtype=jnp.int32)
            circuit_info['circuit_devices'] = circuit_devices
    

        
        train_dataset, validation_dataset  = train_test_split(train_dataset, test_size = test_size)
        
        self.path_count = defaultdict(int)
        n_instruction2circuit_infos, gate_nums = get_n_instruction2circuit_infos(train_dataset)
        # print(gate_nums)

        terminate_num_gate  = 150
        for gate_num in gate_nums:
            if(gate_num >= terminate_num_gate):
                break
            
            real_fidelities = []
            for circuit_info in n_instruction2circuit_infos[gate_num]:
                real_fidelities.append(circuit_info['ground_truth_fidelity'])
                
                for gate_paths in circuit_info['gate_paths']:
                    for path in gate_paths:
                        self.path_count[path] += 1
            
            # mean_real_fidelity = sum(real_fidelities) / len(real_fidelities)
            
            # if mean_real_fidelity < 0.2:  # 再小的质量太低了，可能没用
            #     terminate_num_gate = gate_num
            #     break
            
            # print(f'n_instruction2circuit_infos[{gate_num}] has', len(n_instruction2circuit_infos[gate_num]))
        if verbose:
            print('n_instruction2circuit_infos = ', {n: len(n_instruction2circuit_infos[n]) for n in n_instruction2circuit_infos})
        
        gate_nums = [gate_num for gate_num in gate_nums if gate_num <= terminate_num_gate]
        for gate_num in gate_nums:
            n_instruction2circuit_infos[gate_num] = np.array(n_instruction2circuit_infos[gate_num])
    
        min_loss = 1e10
        best_params = None
        loss_decrease_history = []
        n_iter_no_change = 10
        no_change_tolerance  = .1  # TODO: 可以改到.1
        
        for epoch in range(epoch_num):


            loss_values = []
            
            random_gate_nums = list(gate_nums) # + list(gate_nums)
            random.shuffle(random_gate_nums)

            for gate_num in gate_nums + random_gate_nums:                
                
                for circuit_infos in batch(n_instruction2circuit_infos[gate_num], batch_size=100, should_shuffle = True):
                    X = np.array([circuit_info['vecs'] for circuit_info in circuit_infos], dtype=np.float32)
                    Y = np.array([circuit_info['ground_truth_fidelity'] for circuit_info in circuit_infos], dtype=np.float32)
                    devices = []
                    for circuit_info in circuit_infos:
                        devices.append(circuit_info['circuit_devices'])
                    devices = jnp.array(devices, dtype=jnp.int32)
                        
                    loss_value, gradient = jax.value_and_grad(batch_loss)(params, X, Y, devices)
                    updates, opt_state = optimizer.update(gradient, opt_state, params)
                    params = optax.apply_updates(params, updates)

                    # for key, param in params.items():
                    params['gate_params'] = params['gate_params'].at[params['gate_params'] > error_param_rescale / 10].set(error_param_rescale / 10)  # 假设一个特征对error贡献肯定小于0.1
                    params['gate_params'] = params['gate_params'].at[params['gate_params'] < 0].set(0)

                    params['circuit_bias'] = params['circuit_bias'].at[params['circuit_bias'] > error_param_rescale / 5].set(error_param_rescale / 5)  # 假设一个特征对error贡献肯定小于0.1
                    params['circuit_bias'] = params['circuit_bias'].at[params['circuit_bias'] < -error_param_rescale / 5].set(-error_param_rescale / 5)

                    # params = params.at[params > error_param_rescale / 10].set(error_param_rescale / 10)  # 假设一个特征对error贡献肯定小于0.1
                    # params = params.at[params < 0].set(0)

                    loss_values.append(loss_value)

            test_loss = 0 
            for circuit_info in validation_dataset: #[:2000]:
                # circuit_devices = []
                # for gate in circuit_info['gates']:
                #     device = extract_device(gate)
                #     if 'map' in circuit_info:
                #         if isinstance(device,tuple):
                #             device = (circuit_info['map'][device[0]],circuit_info['map'][device[1]])
                #         else:
                #             device = circuit_info['map'][device]
                #     device_index = list(upstream_model.device2path_table.keys()).index(device)
                #     circuit_devices.append(device_index)
                # circuit_devices = jnp.array(circuit_devices, dtype=jnp.int32)
                test_loss += loss_func(params, np.array(circuit_info['vecs'], dtype=np.float32), circuit_info['ground_truth_fidelity'], circuit_info['circuit_devices'])

            loss_decrease_history.append(min_loss - test_loss)
            if epoch > n_iter_no_change:
                loss_no_change = True
                for loss_decrement in loss_decrease_history[-n_iter_no_change:]:
                    if loss_decrement > no_change_tolerance:
                        loss_no_change = False
                if loss_no_change:
                    break
            
            if test_loss < min_loss:
                min_loss = test_loss
                best_params = params

            if verbose:
                print(f'epoch: {epoch}, \t epoch loss = {sum(loss_values)}, \t test loss = {test_loss}')
                # with open (f'params_{epoch}.pkl', 'wb') as f:
                #     pickle.dump(params,f)
            # params = best_params
        
        self.error_params = best_params
        if verbose:
            print(f'taining error params finishs')
        return best_params
    
        # if validation_dataset is not None:
        #     self.error_params = best_test_params
        # else:
        #     self.error_params = params


    def predict_fidelity(self, circuit_info):
        error_params = self.error_params

        device_list = list(self.upstream_model.device2path_table.keys())
        circuit_devices = []
        for gate in circuit_info['gates']:
            device = extract_device(gate)
            if 'map' in circuit_info:
                if isinstance(device,tuple):
                    device = (circuit_info['map'][device[0]],circuit_info['map'][device[1]])
                else:
                    device = circuit_info['map'][device]
            device_index = device_list.index(device)
            circuit_devices.append(device_index)
        circuit_devices = np.array(circuit_devices)
        vecs = np.array(circuit_info['vecs'])
        circuit_predict = smart_predict(error_params, vecs, circuit_devices)
        circuit_predict = float(circuit_predict)
        circuit_info['circuit_predict'] = circuit_predict
        return circuit_predict


def gate_error(params, vec, device):
    # error = jnp.dot((params[device] / error_param_rescale)**2, vec)
    # error = jnp.dot((params[device][:-1] / error_param_rescale), vec) + params[device][-1]
    error = jnp.dot((params['gate_params'][device] / error_param_rescale), vec)
    return error


@jax.jit
def smart_predict(params, vecs, devices):
    '''预测电路的保真度'''
    errors = vmap(gate_error, in_axes=(None, 0, 0), out_axes=0)(params, vecs, devices)
    return jnp.product(1 - errors, axis=0)  + params['circuit_bias'][0]/error_param_rescale # 不知道为什么是[0][0]

@jax.jit
def loss_func(params, vecs, true_fidelity, devices):
    # predict_fidelity = naive_predict(circ) # 对于电路比较浅的预测是准的，大概是因为有可能翻转回去吧，所以电路深了之后会比理论的保真度要高一些
    predict_fidelity = smart_predict(params, vecs, devices)
    return optax.l2_loss(true_fidelity - predict_fidelity) * 100


def batch_loss(params, X, Y, devices):
    losses = vmap(loss_func, in_axes=(None, 0, 0, 0), out_axes=0)(params, X, Y, devices)
    return jnp.array(losses).sum()


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
