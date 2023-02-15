from dis import Instruction
import pickle
from jax import numpy as jnp
from jax import vmap
import jax
from zoneinfo import available_timezones
import numpy as np
import json
import pprint
from numpy import random
import time
# from to_machine_format_special_edition import conver_circuit
# from direct_access_api_special_edition import direct_sqcg
from qiskit.quantum_info.analysis import hellinger_fidelity
# from new_simulator import get_error_result, count_error_path_num
import ray
import matplotlib.pyplot as plt
from copy import deepcopy
timeout = 3
from simulator.noise_simulator import simulate_noise
from upstream.randomwalk_model import  add_pattern_error_path,RandomwalkModel
from qiskit import QuantumCircuit
timeout = 3
max_qubit_num =10
def count_error_path_num(circuit, n_qubits, model: RandomwalkModel, erroneous_pattern):
    if isinstance(circuit, QuantumCircuit):
        circuit_info = model.vectorize(circuit)
    else:
        circuit_info = circuit
        circuit = circuit_info['qiskit_circuit']

    index2erroneous_pattern = {
        model.path_index(path) : path
        for path in erroneous_pattern
        if model.has_path(path)
    }

    n_erroneous_patterns = []
    
    instruction2sparse_vector = circuit_info['instruction2sparse_vecs']
    instructions = circuit_info['gates']
    for index, instruction in enumerate(instructions):
        error_count = 0 
        sparse_vector = instruction2sparse_vector[index]
        for _index in sparse_vector:
            if _index[0][0] in index2erroneous_pattern:
                # n_erroneous_patterns += 1
                error_count += 1
                break
        
        n_erroneous_patterns.append(error_count)
        
    return n_erroneous_patterns
def get_error_result(circuit_info,model,erroneous_pattern):
    # circuit_info['qiskit_circuit'] = layered_circuits_to_qiskit(max_qubit_num, circuit_info['layer2gates'])
    error_circuit, n_erroneous_patterns = add_pattern_error_path(circuit_info, max_qubit_num, model,erroneous_pattern)
    error_circuit.measure_all()
    noisy_count = simulate_noise(error_circuit, 1000)
    circuit_info['error_result'] = noisy_count
    return circuit_info
@jax.jit
def smart_predict(params, reduced_vecs):
    '''预测电路的保真度'''
    errors = vmap(lambda params, reduced_vec: jnp.dot(params/error_param_rescale, reduced_vec), in_axes=(None, 0), out_axes=0)(params, reduced_vecs)
    return jnp.product(1-errors, axis=0)[0][0] # 不知道为什么是[0][0]
    # return 1-jnp.sum([params[index] for index in sparse_vector])
    # np.array(sum(error).primal)


def getLayerType(circuit,layer):
    return len(circuit['layer2gates'][layer][0]['qubits'])


def getMoveRange(id,circuit):
    available_layer = []
    
    inst = circuit['gates']
    i2l = circuit['gate2layer']
    l2i = circuit['layer2gates']
    
    #accelerate search structure
    loc = 0
    for loc in range(len(inst)):
        if(inst[loc]['id']==id):
            break
    
    layer_size = len(l2i)
    layer = i2l[id]
    # ltype = getLayerType(circuit,layer)

    rows = inst[loc]['qubits']
    rows_set = set(rows)

    ls = layer
    flag = True

    while(flag and ls>=1):
        ls = ls-1
        for index,now_gate in enumerate(l2i[ls]):
            # if ltype != getLayerType(circuit,ls):
            #     flag=False
            #     break
            now_set = set(now_gate['qubits'])
            if(len(rows_set.intersection(now_set))>0):
                flag=False
                break
        if flag == True:
            available_layer.append(ls)

    ls=layer
    flag=True

    while(flag and ls<=layer_size-2):
        ls = ls+1
        # print(ls)
        for index,now_gate in enumerate(l2i[ls]):
            # if ltype != getLayerType(circuit,ls):
            #     flag=False
            #     break
            now_set = set(now_gate['qubits'])
            if(len(rows_set.intersection(now_set))>0):
                flag=False
                break
        if flag == True:
            available_layer.append(ls)

    available_layer = np.sort(available_layer)
    return available_layer
    
    
def moveCircuit(new_layer,id,circuit):
    inst = circuit['gates']
    i2l = circuit['gate2layer']
    l2i = circuit['layer2gates']

    layer = i2l[id]
    i2l[id]=new_layer

    for gate in l2i[layer]:
        if(gate['id']==id):
            l2i[new_layer].append(gate)
            l2i[layer].remove(gate)
            break


multi = 4
def multi_optimize(circuit):
    n_erroneous_patterns = count_error_path_num(circuit, circuit['num_qubits'], model, model.erroneous_pattern)
    error0 = sum(n_erroneous_patterns)
    
    best_circuit = circuit
    best_error = error0 + 1
    
    for i in range(multi):
        circuit_info = optimize(circuit)
        n_erroneous_patterns = count_error_path_num(circuit_info, circuit_info['num_qubits'], model, model.erroneous_pattern)
        error1 = sum(n_erroneous_patterns)
        if error1 < best_error:
            best_error = error1
            best_circuit = circuit_info
    
    print("best_error",best_error)
           
    return best_circuit
        
        

def optimize(circuit):
    # pprint.pprint(circuit)
    # print("-----instructions----------")
    # pprint.pprint(circuit['gates'])
    # print("-------layer2instructions--------")
    # pprint.pprint(circuit['layer2gates'])
    # print("-------instruction2layer--------")
    # print(circuit['gate2layer'])
    circuit_info = deepcopy(circuit)

    time0=time.time()

    gate_num = len(circuit_info['gates'])
    visited = -1
    score0 = smart_predict(error_params, np.array(circuit_info['instruction2reduced_propgation_vecs'], dtype=np.float32))
    n_erroneous_patterns = count_error_path_num(circuit_info, circuit_info['num_qubits'], model, model.erroneous_pattern)
    error0 = sum(n_erroneous_patterns)
    
    
    while True:
        n_erroneous_patterns = count_error_path_num(circuit_info, circuit_info['num_qubits'], model, model.erroneous_pattern)
        error_instructions = [index for index, error_path_num in enumerate(n_erroneous_patterns) if error_path_num > 0]
        
        if len(error_instructions) == 0:
            print('break')
            break
        
        if random.rand()>0.5:
            id = random.randint(0,gate_num)
        else:
            id = random.choice(error_instructions)
        while(id == visited):
            id = random.randint(0, gate_num)
             
        ran = getMoveRange(id, circuit_info)
        # print(id,ran)
        if(len(ran)!=0):
            original_layer = circuit_info['gate2layer'][id]
            # original_score = smart_predict(error_params, np.array(circuit_info['instruction2reduced_propgation_vecs'], dtype=np.float32))
            # 直接算error_path试下
            original_score = -sum(n_erroneous_patterns)
            
            best_score = original_score
            best_pos = original_layer
            
            for lay_pos in ran:
                moveCircuit(lay_pos, id, circuit_info)
                circuit_info = model.vectorize(circuit_info)
                # score = smart_predict(error_params, np.array(circuit_info['instruction2reduced_propgation_vecs'], dtype=np.float32))
                n_erroneous_patterns = count_error_path_num(circuit_info, circuit_info['num_qubits'], model, model.erroneous_pattern)
                score = -sum(n_erroneous_patterns)
                
                if(score > best_score):
                    best_score = score
                    best_pos = lay_pos

            if(best_score > original_score):
                # print("better position found with fidelity rise from %lf to %lf. The gate(id %d)'s layer shifted from %d to %d" %(original_score, best_score, id, original_layer, best_layer))
                # pass
                moveCircuit(best_pos, id, circuit_info)
            else:
                # print("no better position found for gate (id %d, fidelaity %lf). Layer rolled back to %d" %(id, original_score, original_layer))
                moveCircuit(original_layer, id, circuit_info)
                circuit_info = model.vectorize(circuit_info)
        
            visited = id
        # break
        time1=time.time()
        if(time1-time0>timeout):
            break
        
    n_erroneous_patterns = count_error_path_num(circuit_info, circuit_info['num_qubits'], model, model.erroneous_pattern)
    error1 = sum(n_erroneous_patterns)
        
    score1 = smart_predict(error_params, np.array(circuit_info['instruction2reduced_propgation_vecs'], dtype=np.float32))
    print("Optimization completed. The total performance gain is %lf(from %lf to %lf)" %(score1-score0, score0, score1), 'error0', error0, 'error1', error1)

    return circuit_info


@ray.remote
def optimize_remote(circuit):
    return optimize(circuit)

@ray.remote
def multi_optimize_remote(circuit):
    return multi_optimize(circuit)


# erroneous_pattern = ['loop-rz-9,dependency-rz-4', 'loop-cx-2-3,parallel-rx-8', 'loop-rx-9,dependency-cx-7-8', 'loop-cx-8-9,parallel-cx-4-3', 'loop-rz-13,parallel-rz-8', 'loop-cx-6-5,parallel-rz-8', 'loop-rx-8,dependency-rx-7', 'loop-ry-11,parallel-cx-13-14', 'loop-rz-14,dependency-cx-6-5', 'loop-rz-11,dependency-rz-9', 'loop-cx-10-9,dependency-rx-4', 'loop-cx-1-0,parallel-rx-5', 'loop-cx-12-11,dependency-rx-0', 'loop-ry-3,parallel-rx-2', 'loop-ry-14,parallel-cx-10-9', 'loop-rx-4,dependency-ry-13', 'loop-ry-2,dependency-cx-13-12', 'loop-rz-8,parallel-rz-0', 'loop-cx-9-10,dependency-ry-12', 'loop-rz-11,parallel-cx-8-9']

if __name__ == '__main__':
    with open('pattern_extractor/model/10qubits_200gate_fidelity_param','rb')as f:
        model = pickle.load(f)

    # with open('model.pkl','rb')as f:
    #     model = pickle.load(f)

    print(model.erroneous_pattern)
    
    dataset = model.dataset
    
    model.dataset=None
    
    error_params = np.array(model.error_params['gate_parm'])

    error_param_rescale=10000

    opt_res = {}
    index = 0
    
    # ray.init()
    # ray.put(algorithm)
    futures = {}
    
    ## here we submit 'optimize' task
    for index, circuit in enumerate(dataset):
        
        original_circuit = model.vectorize(circuit)

        original_score = sum(count_error_path_num(original_circuit, original_circuit['num_qubits'], model, model.erroneous_pattern))
        # if original_score < 2:
        #     print('passs')
        #     continue
        
        predict_fidelity = smart_predict(error_params, np.array(circuit['instruction2reduced_propgation_vecs'], dtype=np.float32))
        
        # if(predict_fidelity > 0.8):
        #     continue
        # futures[index] = optimize(circuit)
        # futures[index] = optimize_remote.remote(circuit)
        # futures[index] = multi_optimize(circuit)
        futures[index] = multi_optimize_remote.remote(circuit)
        
        
        opt_res[index] = {}
        opt_res[index]['original_predict'] = predict_fidelity
        opt_res[index]['original_ground_truth_fidelity'] = circuit['ground_truth_fidelity']
        opt_res[index]['original_circuit'] = original_circuit
        print(index,"submitted")


    true_result = {'0'*10: 2000}

    ## get all optimized circuit and predict result after optimized    
    for index in futures:    
        future = futures[index]
        circuit = ray.get(future)
        # circuit = future
    
        optimized_predict_fidelity = smart_predict(error_params, np.array(circuit['instruction2reduced_propgation_vecs'], dtype=np.float32))
    
        opt_res[index]['optimized_predict'] = optimized_predict_fidelity
        # opt_res[index]['improvement'] = optimized_predict_fidelity - predict_fidelity
        opt_res[index]['optimized_circuit'] = circuit
        
        # print('\n\n')
        # print(opt_res[index]['original_circuit'])
        # print(opt_res[index]['optimized_circuit'])
        
        # opt_cir = get_error_result(opt_res[index]['optimized_circuit'], model, model.erroneous_pattern)
        # cir = get_error_result(opt_res[index]['original_circuit'], model, model.erroneous_pattern)
        # optimized_ground_truth_fidelity = hellinger_fidelity(opt_cir['error_result'], true_result)
        # original_ground_truth_fidelity = hellinger_fidelity(cir['error_result'], true_result)
        # print(optimized_ground_truth_fidelity, original_ground_truth_fidelity)
        
        
        print("index", index, "ori_predict", opt_res[index]['original_predict'], "opt_predict", optimized_predict_fidelity)
        
    ## run optimized circuit in simulator
    for index in opt_res:
        cir = opt_res[index]['optimized_circuit']
        
        
        cir = get_error_result(cir, model, model.erroneous_pattern)
        optimized_ground_truth_fidelity = hellinger_fidelity(cir['error_result'], true_result)
        opt_res[index]['optimized_ground_truth_fidelity'] =  optimized_ground_truth_fidelity
        print("index",index,"ori_predict",predict_fidelity,"opt_predict",optimized_predict_fidelity,"ori_truth", circuit['ground_truth_fidelity'], "opt_truth",optimized_ground_truth_fidelity)
    
        with open('merged/merged_result_'+str(index)+'.pkl','wb') as f:
            pickle.dump(opt_res[index],f)


    import os
    import traceback
    path = './merged'
    pathList = os.listdir(path)

    opt_res = {}
    count = 0
    for file in pathList:
        if 'pkl' not in file:
            continue
        abs_path = os.path.join(path,file)
        # print(abs_path)
        try:
            with open(abs_path, 'rb') as file:
                result = pickle.load(file)
            opt_res[count] = result
            count += 1
        except Exception as e:
            traceback.print_exc()
        
        # print(result)
            
    ## the following is used to plot
    
    merged_res = opt_res
    
    # with open('merged/save_res.pkl','rb') as f:
    #     merged_res = pickle.load(f)

    opt_plot_data = {}
    ori_plot_data = {}
    data_size = {}
    model_ori_data = {}
    model_opt_data={}


    top15_circuits = []
    for key in merged_res:
        dura = merged_res[key]['original_circuit']['duration']
        imp = merged_res[key]['optimized_ground_truth_fidelity'] - merged_res[key]['original_ground_truth_fidelity']
        if imp < 0:
            continue
        
        # (improve, original_predict, optimized_ground_truth_fidelity, optimized_predict, original_ground_truth_fidelity)
        
        top15_circuits.append((
            imp,
            float(merged_res[key]['original_predict']),
            merged_res[key]['optimized_ground_truth_fidelity'], 
            float(merged_res[key]['optimized_predict']),
            merged_res[key]['original_ground_truth_fidelity'],
            dura
        ))
        # print(merged_res[key]['optimized_ground_truth_fidelity'])
        if(dura in opt_plot_data):
            opt_plot_data[dura] += merged_res[key]['optimized_ground_truth_fidelity']
            ori_plot_data[dura] += merged_res[key]['original_ground_truth_fidelity']
            model_ori_data[dura] += merged_res[key]['original_predict']
            model_opt_data[dura] += merged_res[key]['optimized_predict']
            data_size[dura] += 1
        else:
            opt_plot_data[dura] = merged_res[key]['optimized_ground_truth_fidelity']
            ori_plot_data[dura] = merged_res[key]['original_ground_truth_fidelity']
            model_ori_data[dura] = merged_res[key]['original_predict']
            model_opt_data[dura] = merged_res[key]['optimized_predict']
            data_size[dura] = 1


    for dura in opt_plot_data:
        opt_plot_data[dura] = opt_plot_data[dura] / data_size[dura]
        ori_plot_data[dura] = ori_plot_data[dura] / data_size[dura]
        model_ori_data[dura] = model_ori_data[dura]/ data_size[dura]
        model_opt_data[dura] = model_opt_data[dura]/ data_size[dura]

    top15_circuits.sort(key = lambda elm: elm[0], reverse=True)
    top15_circuits = top15_circuits[:15]
    print(top15_circuits)

    sorted_opt = sorted(opt_plot_data.items()) #,key=lambda x:x[1])#,reverse=True)
    sorted_ori = sorted(ori_plot_data.items())
    sorted_model_ori= sorted(model_ori_data.items())
    sorted_model_opt=sorted(model_opt_data.items())

    data_size = sorted(data_size.items())

    # print(data_size)

    opt_plot_data = {}
    for index in sorted_opt:
        opt_plot_data[index[0]]=index[1]
        
    ori_plot_data = {}
    for index in sorted_ori:
        ori_plot_data[index[0]]=index[1]

    model_ori_data = {}
    model_opt_data={}
    for index in sorted_model_ori:
        model_ori_data[index[0]]=index[1]

    for index in sorted_model_opt:
        model_opt_data[index[0]]=index[1]

    # print(opt_plot_data.keys(),opt_plot_data.values(),ori_plot_data.values())


    fig = plt.figure()

    # plt.plot(opt_plot_data.keys(), opt_plot_data.values(),label='sim_opt')

    plt.plot(ori_plot_data.keys(), ori_plot_data.values(),color='red',label='sim_ori')

    # plt.plot(model_ori_data.keys(), model_ori_data.values(),color='green',label='model_ori')

    plt.plot(model_opt_data.keys(), model_opt_data.values(),color='yellow',label='model_opt')

    plt.legend(loc='best',fontsize=12)

    plt.show()

    fig.savefig("circuit_optimize_2.svg")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # for index in opt_res:
    #     print(index,opt_res[index]['original'],opt_res[index]['optimized'],opt_res[index]['improvement'])

    # sorted_res = sorted(opt_res.items(),key=lambda x:x[1]['improvement'],reverse=True)

    # opt_res = {}
    # for index in sorted_res:
    #     opt_res[index[0]]=index[1]
    # print("----------------------------------")
    # for index in opt_res:
    #     print(index,opt_res[index]['original'],opt_res[index]['optimized'],opt_res[index]['improvement'])
    


    # with open('optimized_result.pkl','wb')as f:
    #     pickle.dump(opt_res,f)
    # with open('optimized_result.pkl','rb')as f:
    #     opt_res = pickle.load(f)
    
    # for index in opt_res:
    #     print(index,opt_res[index]['original'],opt_res[index]['optimized'],opt_res[index]['improvement'])
    
    
    
    
    # seq = conver_circuit(circuit)
    # ret = direct_sqcg(seq, 1000, ['N36U19'])

    # true_result = {
    # '0'*5: 2000
    # }
    # circuit_info = algorithm[0]

    # circuit_info['ground_truth_fidelity'] = hellinger_fidelity(ret['probs'], true_result)
    # print("----------------------------")
    # print(circuit_info['ground_truth_fidelity'])

        # circuit = model.vectorize(circuit)
        # predict_fidelity = smart_predict(error_params, np.array(circuit['instruction2reduced_propgation_vecs'], dtype=np.float32))
        # 
        # optimized_predict_fidelity = smart_predict(error_params, np.array(circuit['instruction2reduced_propgation_vecs'], dtype=np.float32))
        # d[cnt]=optimized_predict_fidelity-predict_fidelity
        # cnt+=1
    
    # sorted(d)
    # print(d)

        # print(predict_fidelity,optimized_predict_fidelity,optimized_predict_fidelity-predict_fidelity)
        # machine_circuit = conver_circuit(circuit) 
        # print(machine_circuit)
    # predict_fidelity = smart_predict(error_params, np.array(circuit['instruction2reduced_propgation_vecs'], dtype=np.float32))
    # print(predict_fidelity)
    # circuit = optimize(circuit)
    # circuit = model.vectorize(circuit)
    # optimized_predict_fidelity = smart_predict(error_params, np.array(circuit['instruction2reduced_propgation_vecs'], dtype=np.float32))

