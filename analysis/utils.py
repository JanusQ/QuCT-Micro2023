from collections import defaultdict
from circuit.quct.analysis.dimensionality_reduction import batch
import numpy as np
from jax import numpy as jnp
import jax
from jax import vmap
import optax
from circuit.quct.analysis.sparse_dimensionality_reduction import sp_dist

def get_extra_info(dataset):
    
    def get_layer_type_divide(layer2instructions):
        """
        返回layer2instructions的每层的类型，是单比特门层则为1；否则则为2
        """
        return [len(layer[0]['qubits']) for layer in layer2instructions]
    def get_couple_prop(circuit_info):
        couple = 0
        for ins in circuit_info['instructions']:
            if len(ins['qubits']) == 2:
                couple += 1
        return couple / (len(circuit_info['instructions']))
    def get_xeb_fidelity(circuit_info):
        fidelity = 1
        for instruction in circuit_info['instructions']:
            if len(instruction['qubits']) == 2:
                q0_id, q1_id = instruction['qubits'][0], instruction['qubits'][1]
                if q0_id > q1_id:
                    fidelity = fidelity * (1-couple_average_error[q1_id])
                else:
                    fidelity = fidelity * (1-couple_average_error[q0_id])
            else:
                q0_id = instruction['qubits'][0]
                fidelity = fidelity * (1 - single_average_error[q0_id])
        return  fidelity * np.product((measure0_fidelity+measure1_fidelity)/2)

    def get_circuit_duration(layer2instructions):
        single_gate_time = 30
        two_gate_time  = 60
        layer_types = get_layer_type_divide(layer2instructions)
        
        duration = 0
        for layer_type in layer_types:
            if layer_type == 1:
                duration += single_gate_time
            elif layer_type == 2:
                duration += two_gate_time
            else:
                raise Exception(layer_type)
            
        return duration

    qubits = ['q3_15', 'q3_17', 'q3_19',  'q5_19', 'q7_19',]

    single_average_error = np.array([0.084, 0.040, 0.083, 0.025, 0.037, ]) / 100 # 原先是带%的

    couple_average_error = np.array([0.6, 0.459, 0.537, 0.615, ]) / 100  # q1_q2, q2_q3, ..., q9_q10

    t1s = np.array([94.5, 124.1, 117.1, 124.8, 136.3])  # q1, q2, ..., q5

    t2s = np.array([5.04, 7.63, 5.67, 6.97, 4.40, ]) # q1, q2, ..., q10

    measure0_fidelity = np.array([0.97535460, 0.97535460, 0.9645634, 0.9907482, 0.96958333])
    measure1_fidelity = np.array([0.955646258, 0.97572327, 0.950431034, 0.9629411764, 0.9570833333])


    for cir in dataset:
        cir['xeb_fidelity'] = get_xeb_fidelity(cir)
        cir['duration'] = get_circuit_duration(cir['layer2instructions'])
        cir['prop'] = get_couple_prop(cir)
        cir['gate_num'] = len(cir['instructions'])
    return dataset

def func_dist(vec1, vec2):
    return jnp.sqrt(sp_dist(vec1, vec2) / 1000000)

error_param_rescale = 10000
@jax.jit
def smart_predict(params, reduced_vecs):
    '''预测电路的保真度'''
    errors = vmap(lambda params, reduced_vec: jnp.dot(params/error_param_rescale, reduced_vec), in_axes=(None, 0), out_axes=0)(params, reduced_vecs)
    return jnp.product(1-errors, axis=0)[0] # 不知道为什么是[0][0]
    # return 1-jnp.sum([params[index] for index in sparse_vector])
    # np.array(sum(error).primal)

def loss_func(params, reduced_vecs, true_fidelity):
    # predict_fidelity = naive_predict(circ) # 对于电路比较浅的预测是准的，大概是因为有可能翻转回去吧，所以电路深了之后会比理论的保真度要高一些
    predict_fidelity = smart_predict(params, reduced_vecs)
    # print(predict_fidelity)
    # print(true_fidelity)
    return optax.l2_loss(true_fidelity - predict_fidelity)*100

def batch_loss(params, X, Y):
    losses = vmap(loss_func, in_axes=(None, 0, 0), out_axes=0)(params, X, Y)
    return losses.mean()

# 在训练中逐渐增加gate num
def epoch_train(circuit_infos, params, opt_state, optimizer):
    # print(circuit_infos[0].keys())
    X = np.array([ circuit_info['reduced_vecs'] for circuit_info in circuit_infos], dtype=np.float32) 
    Y = np.array([ [circuit_info['ground_truth_fidelity']] for circuit_info in circuit_infos], dtype=np.float32)

    loss_value, gradient = jax.value_and_grad(batch_loss)(params, X, Y)
    updates, opt_state = optimizer.update(gradient, opt_state, params)
    params = optax.apply_updates(params, updates)

    params = params.at[params > error_param_rescale/10].set(error_param_rescale/10)  # 假设一个特征对error贡献肯定小于0.1
    params = params.at[params < 0].set(0)

    return loss_value, params, opt_state


def get_n_instruction2circuit_infos(dataset):
    n_instruction2circuit_infos = defaultdict(list)
    for circuit_info in dataset:
        # qiskit_circuit = circuit_info['qiskit_circuit']
        gate_num = len(circuit_info['instructions'])
        n_instruction2circuit_infos[gate_num].append(circuit_info)

    # print(n_instruction2circuit_infos[gate_num])
    gate_nums = list(n_instruction2circuit_infos.keys())
    gate_nums.sort()

    return n_instruction2circuit_infos, gate_nums

def train_error_params(train_dataset, params, opt_state, optimizer, epoch_num = 10):
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
            for circuit_infos in batch(n_instruction2circuit_infos[gate_num], batch_size = 100):
                loss_value, params, opt_state = epoch_train(circuit_infos, params, opt_state, optimizer)
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


# 整一个naive点的方法

@jax.jit
def naive_predict(naive_params, instructions):
    cal_fidelity = lambda qubits: jnp.where(qubits[1] == -1, naive_params['single'][qubits[0]], naive_params['double'][qubits[0]][qubits[1]]) / error_param_rescale
    fidelity = jnp.product(vmap(cal_fidelity, in_axes=(0,), out_axes=0)(instructions), axis=0)
    # for qubits in instructions:
    #     fidelity *= jnp.where(qubits[1] == -1, naive_params['single'][qubits[0]] / error_param_rescale, naive_params['double'][qubits[0]][qubits[1]] / error_param_rescale ) 
        # if qubits[1] == -1:
        #     fidelity *= naive_params['single'][qubits[0]] / error_param_rescale
        # else:
        #     fidelity *= naive_params['double'][qubits[0]][qubits[1]] / error_param_rescale
    return fidelity

def naive_loss(naive_params, instructions, true_fidelity):
    predict_fidelity = naive_predict(naive_params, instructions)
    return optax.l2_loss(true_fidelity - predict_fidelity)*100

def naive_batch_loss(naive_params, X, Y):
    # losses = jnp.array([naive_loss(naive_params, x, y) for x, y in zip(X, Y)])
    losses = vmap(naive_loss, in_axes=(None, 0, 0), out_axes=0)(naive_params, X, Y)
    return losses.mean()

def naive_prase_circuit(circuit_info):
    return np.array([
        instruction['qubits'] if len(instruction['qubits']) == 2 else instruction['qubits'] + [-1]
        for instruction in circuit_info['instructions']
    ])

# 在训练中逐渐增加gate num
def naive_epoch_train(circuit_infos, naive_params, naive_opt_state, naive_optimizer):
    # print(circuit_infos[0].keys())

    # print(circuit_infos[0]['qiskit_circuit'])

    X = np.array([naive_prase_circuit(circuit_info) for circuit_info in circuit_infos])

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
            n_instruction2circuit_infos[gate_num] = np.array(n_instruction2circuit_infos[gate_num])
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
    
        test_mean_losses = []
        for circuit_info in circuit_infos:
            test_x = naive_prase_circuit(circuit_info)
            test_y = np.array([circuit_info['ground_truth_fidelity']], dtype=np.float32)
            test_mean_loss = naive_loss(best_params, test_x, test_y)
            test_mean_losses.append(test_mean_loss)

        print(f'gate num: {gate_num}, test mean loss: {np.array(test_mean_losses).mean()}')

    print(f'taining finishs')
    return best_loss_value, naive_params, opt_state
