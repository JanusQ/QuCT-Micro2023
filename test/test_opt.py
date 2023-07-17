import pickle
from jax import numpy as jnp
from jax import vmap
import jax
import numpy as np
from numpy import random
import time
from qiskit.quantum_info.analysis import hellinger_fidelity
import ray
import matplotlib.pyplot as plt
from copy import deepcopy
from downstream.fidelity_predict.fidelity_analysis import smart_predict, error_param_rescale
from upstream.randomwalk_model import RandomwalkModel, extract_device
from qiskit import QuantumCircuit
from collections import defaultdict


def count_error_path_num(circuit_info, model: RandomwalkModel):

    device2erroneous_pattern = model.erroneous_pattern
    device2erroneous_pattern_index = defaultdict(list)
    for device, erroneous_pattern in device2erroneous_pattern.items():
        erroneous_pattern_index = {
            model.path_index(device, path): path
            for path in erroneous_pattern
            if model.has_path(device, path)
        }
        device2erroneous_pattern_index[device] = erroneous_pattern_index

    n_erroneous_patterns = []

    gates = circuit_info['gates']
    for index, gate in enumerate(gates):
        name = gate['name']
        qubits = gate['qubits']
        params = gate['params']
        device = extract_device(gate)
        if 'map' in circuit_info:
            if isinstance(device, tuple):
                device = (circuit_info['map'][device[0]],
                          circuit_info['map'][device[1]])
            else:
                device = circuit_info['map'][device]
        erroneous_pattern_index = device2erroneous_pattern_index[device]
        error_count = 0
        path_index = circuit_info['path_indexs'][index]
        for _index in path_index:
            if _index in erroneous_pattern_index:
                error_count += 1

        n_erroneous_patterns.append(error_count)
       
    return n_erroneous_patterns


def getLayerType(circuit,layer):
    return len(circuit['layer2gates'][layer][0]['qubits'])


def getMoveRange(id,circuit):
    available_layer = []
    
    inst = circuit['gates']
    i2l = circuit['gate2layer']
    l2i = circuit['layer2gates']
    
    # accelerate search structure
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
    n_erroneous_patterns = count_error_path_num(circuit, upstream_model)
    error0 = sum(n_erroneous_patterns)
    best_circuit = circuit
    best_error = error0 + 1
    
    for i in range(multi):
        circuit_info = optimize(circuit)
        n_erroneous_patterns = count_error_path_num(circuit_info, upstream_model)
        error1 = sum(n_erroneous_patterns)
        if error1 < best_error:
            best_error = error1
            best_circuit = circuit_info
    
    print("best_error",best_error)
           
    return best_circuit
        
        

def optimize(circuit):
    circuit_info = deepcopy(circuit)

    time0=time.time()

    gate_num = len(circuit_info['gates'])
    visited = -1
    score0 = downstream_model.predict_fidelity(circuit_info)
    n_erroneous_patterns = count_error_path_num(circuit_info, upstream_model)
    error0 = sum(n_erroneous_patterns)
    
    
    while True:
        n_erroneous_patterns = count_error_path_num(circuit_info, upstream_model)
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
            original_score = -sum(n_erroneous_patterns)
            
            best_score = original_score
            best_pos = original_layer
            
            for lay_pos in ran:
                moveCircuit(lay_pos, id, circuit_info)
                circuit_info = upstream_model.vectorize(circuit_info)
                n_erroneous_patterns = count_error_path_num(circuit_info, upstream_model)
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
                circuit_info = upstream_model.vectorize(circuit_info)
        
            visited = id
        # break
        time1=time.time()
        if(time1-time0>3):
            break
        
    n_erroneous_patterns = count_error_path_num(circuit_info, upstream_model)
    error1 = sum(n_erroneous_patterns)
    score1 = downstream_model.predict_fidelity(circuit_info)
    print("Optimization completed. The total performance gain is %lf(from %lf to %lf)" %(score1-score0, score0, score1), 'error0', error0, 'error1', error1)

    return circuit_info


@ray.remote
def optimize_remote(circuit):
    return optimize(circuit)

@ray.remote
def multi_optimize_remote(circuit):
    return multi_optimize(circuit)



if __name__ == '__main__':
    with open(f"temp_data/error_params_predicts_5.pkl", "rb")as f:
        downstream_model, predicts, reals, durations, test_dataset = pickle.load(f)
    upstream_model = downstream_model.upstream_model

    print(upstream_model.erroneous_pattern)
    error_params = downstream_model.error_params


    opt_res = {}
    futures = {}
    
    # here we submit 'optimize' task
    for index, circuit in enumerate(test_dataset):
        

        n_erroneous_patterns= count_error_path_num(circuit, upstream_model)
        original_score = sum(n_erroneous_patterns)
        predict_fidelity = downstream_model.predict_fidelity(circuit)
        
        futures[index] = multi_optimize_remote.remote(circuit)
        # cir_opt = multi_optimize(circuit)
        
        opt_res[index] = {}
        opt_res[index]['original_predict'] = predict_fidelity
        opt_res[index]['original_ground_truth_fidelity'] = circuit['ground_truth_fidelity']
        opt_res[index]['original_circuit'] = circuit
        print(index,"submitted")


    true_result = {'0'*10: 2000}

    delta = []
    # get all optimized circuit and predict result after optimized    
    for index in futures:    
        future = futures[index]
        circuit = ray.get(future)
    
        optimized_predict_fidelity = downstream_model.predict_fidelity(circuit)
    
        opt_res[index]['optimized_predict'] = optimized_predict_fidelity
        opt_res[index]['optimized_circuit'] = circuit
        
        
        
        print("index", index, "ori_predict", opt_res[index]['original_predict'], "opt_predict", optimized_predict_fidelity)
        delta.append(optimized_predict_fidelity - opt_res[index]['original_predict'])
    delta.sort()[::-1]
    print('delta')
        
    # # run optimized circuit in simulator
    # for index in opt_res:
    #     cir = opt_res[index]['optimized_circuit']
        
        
    #     cir = get_error_result(cir, model, upstream_model.erroneous_pattern)
    #     optimized_ground_truth_fidelity = hellinger_fidelity(cir['error_result'], true_result)
    #     opt_res[index]['optimized_ground_truth_fidelity'] =  optimized_ground_truth_fidelity
    #     print("index",index,"ori_predict",predict_fidelity,"opt_predict",optimized_predict_fidelity,"ori_truth", circuit['ground_truth_fidelity'], "opt_truth",optimized_ground_truth_fidelity)
    
    #     with open('merged/merged_result_'+str(index)+'.pkl','wb') as f:
    #         pickle.dump(opt_res[index],f)


    # import os
    # import traceback
    # path = './merged'
    # pathList = os.listdir(path)

    # opt_res = {}
    # count = 0
    # for file in pathList:
    #     if 'pkl' not in file:
    #         continue
    #     abs_path = os.path.join(path,file)
    #     # print(abs_path)
    #     try:
    #         with open(abs_path, 'rb') as file:
    #             result = pickle.load(file)
    #         opt_res[count] = result
    #         count += 1
    #     except Exception as e:
    #         traceback.print_exc()
        
    #     # print(result)
            
    # # the following is used to plot
    
    # merged_res = opt_res
    
    # # with open('merged/save_res.pkl','rb') as f:
    # #     merged_res = pickle.load(f)

    # opt_plot_data = {}
    # ori_plot_data = {}
    # data_size = {}
    # model_ori_data = {}
    # model_opt_data={}


    # top15_circuits = []
    # for key in merged_res:
    #     dura = merged_res[key]['original_circuit']['duration']
    #     imp = merged_res[key]['optimized_ground_truth_fidelity'] - merged_res[key]['original_ground_truth_fidelity']
    #     if imp < 0:
    #         continue
        
    #     # (improve, original_predict, optimized_ground_truth_fidelity, optimized_predict, original_ground_truth_fidelity)
        
    #     top15_circuits.append((
    #         imp,
    #         float(merged_res[key]['original_predict']),
    #         merged_res[key]['optimized_ground_truth_fidelity'], 
    #         float(merged_res[key]['optimized_predict']),
    #         merged_res[key]['original_ground_truth_fidelity'],
    #         dura
    #     ))
    #     # print(merged_res[key]['optimized_ground_truth_fidelity'])
    #     if(dura in opt_plot_data):
    #         opt_plot_data[dura] += merged_res[key]['optimized_ground_truth_fidelity']
    #         ori_plot_data[dura] += merged_res[key]['original_ground_truth_fidelity']
    #         model_ori_data[dura] += merged_res[key]['original_predict']
    #         model_opt_data[dura] += merged_res[key]['optimized_predict']
    #         data_size[dura] += 1
    #     else:
    #         opt_plot_data[dura] = merged_res[key]['optimized_ground_truth_fidelity']
    #         ori_plot_data[dura] = merged_res[key]['original_ground_truth_fidelity']
    #         model_ori_data[dura] = merged_res[key]['original_predict']
    #         model_opt_data[dura] = merged_res[key]['optimized_predict']
    #         data_size[dura] = 1


    # for dura in opt_plot_data:
    #     opt_plot_data[dura] = opt_plot_data[dura] / data_size[dura]
    #     ori_plot_data[dura] = ori_plot_data[dura] / data_size[dura]
    #     model_ori_data[dura] = model_ori_data[dura]/ data_size[dura]
    #     model_opt_data[dura] = model_opt_data[dura]/ data_size[dura]

    # top15_circuits.sort(key = lambda elm: elm[0], reverse=True)
    # top15_circuits = top15_circuits[:15]
    # print(top15_circuits)

    # sorted_opt = sorted(opt_plot_data.items()) #,key=lambda x:x[1])#,reverse=True)
    # sorted_ori = sorted(ori_plot_data.items())
    # sorted_model_ori= sorted(model_ori_data.items())
    # sorted_model_opt=sorted(model_opt_data.items())

    # data_size = sorted(data_size.items())

    # # print(data_size)

    # opt_plot_data = {}
    # for index in sorted_opt:
    #     opt_plot_data[index[0]]=index[1]
        
    # ori_plot_data = {}
    # for index in sorted_ori:
    #     ori_plot_data[index[0]]=index[1]

    # model_ori_data = {}
    # model_opt_data={}
    # for index in sorted_model_ori:
    #     model_ori_data[index[0]]=index[1]

    # for index in sorted_model_opt:
    #     model_opt_data[index[0]]=index[1]

    # # print(opt_plot_data.keys(),opt_plot_data.values(),ori_plot_data.values())


    # fig = plt.figure()

    # # plt.plot(opt_plot_data.keys(), opt_plot_data.values(),label='sim_opt')

    # plt.plot(ori_plot_data.keys(), ori_plot_data.values(),color='red',label='sim_ori')

    # # plt.plot(model_ori_data.keys(), model_ori_data.values(),color='green',label='model_ori')

    # plt.plot(model_opt_data.keys(), model_opt_data.values(),color='yellow',label='model_opt')

    # plt.legend(loc='best',fontsize=12)

    # plt.show()

    # fig.savefig("circuit_optimize_2.svg")
    
    
    
    
    

