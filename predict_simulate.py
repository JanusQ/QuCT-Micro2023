from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from plot.plot import plot_duration_fidelity
import random
import numpy as np

from circuit import gen_random_circuits, label_ground_truth_fidelity
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import NoiseSimulator
from utils.backend import devide_chip, gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle
def make_circuitlet(dataset):
    result = []
    
    for i, circuit in enumerate(dataset):
        result += cut_circuit(circuit)
        
    return result
        
def cut_circuit(circuit):
    # save_gates = copy.deepcopy(circuit['gates'])
    result = []
    patterns = circuit['devide_qubits']
    for pattern in patterns:
        new_index = list(range(len(pattern)))
        
        qubit_map = {}
        reverse_qubit_map = {}
        
        for i in range(len(pattern)):
            qubit_map[new_index[i]] = pattern[i]
            reverse_qubit_map[pattern[i]] = new_index[i]
        
        circ_let = {}
        
        #hard code
        circ_let['qiskit_circuit']=None
        circ_let['duration']=360
        
        circ_let['num_qubits']=len(pattern)
        circ_let['divide_qubits'] = [pattern]
        circ_let['gate_paths']=[]
        circ_let['path_indexs']=[]
        circ_let['vecs']=[]
        
        circ_let['layer2gates'] = []
        l2g_copy = copy.deepcopy(circuit['layer2gates'])
        for layer in l2g_copy:
            new_layer = []
            for gate in layer:
                if set(gate['qubits']) & set(pattern):
                    new_layer.append(gate)
            circ_let['layer2gates'].append(new_layer)
        
        circ_let['gates'] = []
        circ_let['gate2layer'] = []
        gates_copy = copy.deepcopy(circuit['gates'])
        for gate in gates_copy:
            if set(gate['qubits']) & set(pattern):
                circ_let['gates'].append(gate)
                circ_let['gate2layer'].append(circuit['gate2layer'][gate['id']])
        # save_gates2 = copy.deepcopy(circ_let['gates'])
        #process index
        # print(reverse_qubit_map)
        for gate in circ_let['gates']:
            for i in range(len(gate['qubits'])):
                gate['qubits'][i]=reverse_qubit_map[gate['qubits'][i]]
        
        for layer in circ_let['layer2gates']:
            for gate in layer:
                for i in range(len(gate['qubits'])):
                    gate['qubits'][i]=reverse_qubit_map[gate['qubits'][i]]
                    
        for gate in circ_let['gates']:
            circ_let['gate_paths'].append(circuit['gate_paths'][gate['id']])
            circ_let['path_indexs'].append(circuit['path_indexs'][gate['id']])
            circ_let['vecs'].append(circuit['vecs'][gate['id']])
        
        circ_let['gate_num']=len(circ_let['gates'])
        
        circ_let['map'] = qubit_map
        circ_let['reverse_map'] = reverse_qubit_map
            

        result.append(circ_let)
    
    return result
        
size = 6
n_qubits = 18
topology = gen_grid_topology(size)  # 3x3 9 qubits
new_topology = defaultdict(list)
for qubit in topology.keys():
    if qubit < n_qubits:
        for ele in topology[qubit]:
            if ele < n_qubits:
                new_topology[qubit].append(ele)
topology =  new_topology      
neigh_info = copy.deepcopy(topology)
coupling_map = topology_to_coupling_map(topology)      



backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=default_basis_single_gates,
                  basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

covered_couplng_map = set()
dataset = []
while True:
    ret_backend = devide_chip(backend, max_qubit=5)
    before = len(covered_couplng_map)
    for ele in ret_backend.coupling_map:
        covered_couplng_map.add(tuple(ele))
    if before == len(covered_couplng_map):
        continue
    dataset += gen_random_circuits(min_gate=20, max_gate=160, n_circuits=4,
                                   two_qubit_gate_probs=[4, 8], backend=ret_backend, multi_process=True)
    # dataset_copy = copy.deepcopy(dataset)
    # make_circuitlet(dataset)
    if len(covered_couplng_map) == len(coupling_map):
        break

upstream_model = RandomwalkModel(1, 20, backend=backend)
print(len(dataset), "circuit generated")
upstream_model.train(dataset, multi_process=True)

with open("upstream_model_18.pkl","wb")as f:
    pickle.dump(upstream_model,f)

print("original",len(dataset))
dataset = make_circuitlet(dataset)
print("cutted",len(dataset))


simulator = NoiseSimulator(backend)
erroneous_pattern = simulator.get_error_results(dataset, upstream_model)
upstream_model.erroneous_pattern = erroneous_pattern


    

index = np.arange(len(dataset))
random.shuffle(index)
train_index, test_index = index[:-1500], index[-1500:]
train_dataset, test_dataset = np.array(
    dataset)[train_index], np.array(dataset)[test_index]

with open("split_dataset_18.pkl","wb")as f:
    pickle.dump((train_dataset, test_dataset),f)

def plot_top_ratio(upstream_model, erroneous_pattern_weight):
    x ,y = [],[]
    for top in range(1,100,1):
        top /= 100
        total_find = 0
        for device, pattern_weights in erroneous_pattern_weight.items():
            path_table_size = len(upstream_model.device2path_table[device].keys())
            for pattern_weight in pattern_weights:
                if  pattern_weight[1] < top * path_table_size:
                    total_find += 1

        find_ratio = total_find / len(upstream_model.erroneous_pattern.keys()) * 3
        print(top,find_ratio)
        x.append(top)
        y.append(find_ratio)
        
        
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(figsize=(20,6)) # 创建一个图形对象和一个子图对象
    axes.plot(x, y ,markersize = 12,linewidth = 2, label='ratio',marker = '^' )
    axes.set_xlabel('top ')
    axes.set_ylabel('find_ratio')
    axes.legend() # 添加图例
    fig.show()
    fig.savefig("find_ratio.svg")

def find_error_path(upstream_model, error_params):
    error_params = np.array(error_params)
    erroneous_pattern = upstream_model.erroneous_pattern
    
    device_index2device = {} #两比特门与但单比特门映射为一维下标
    for device  in upstream_model.device2path_table.keys():
        device_index = list(upstream_model.device2path_table.keys()).index(device)
        device_index2device[device_index] = device
        
    error_params_path_weight = {} #训练好的参数对应的path及其权重
    error_params_path = {}
    for idx, device_error_param in enumerate(error_params):
        device = device_index2device[idx]
        sort = np.argsort(device_error_param)
        sort = sort[::-1]
        device_error_params_path_weight = []
        device_error_params_path = []
        for i in sort:
            if int(i) in upstream_model.device2reverse_path_table[device].keys():
                path = upstream_model.device2reverse_path_table[device][int(i)]
                if isinstance(path,str):
                    device_error_params_path_weight.append((path,device_error_param[i]))
                    device_error_params_path.append(path)
        error_params_path_weight[device] = device_error_params_path_weight
        error_params_path[device] = device_error_params_path
        
    erroneous_pattern_weight = {} #手动添加的error_path在训练完参数中的排位
    for device, patterns in erroneous_pattern.items():
        device_error_params_path = error_params_path[device]
        device_erroneous_pattern_weight = []
        for pattern in patterns:
            if pattern in device_error_params_path:
                k = device_error_params_path.index(pattern)
                device_erroneous_pattern_weight.append((pattern,k))
        erroneous_pattern_weight[device] = device_erroneous_pattern_weight
        
    plot_top_ratio(upstream_model, erroneous_pattern_weight)

# with open("split_dataset.pkl","rb")as f:
#     train_dataset, test_dataset = pickle.load(f)
# with open("upstream_model.pkl","rb")as f:
#     upstream_model = pickle.load(f)
    
downstream_model = FidelityModel(upstream_model)
downstream_model.train(train_dataset)


for idx, cir in enumerate(test_dataset):
    cir = upstream_model.vectorize(cir)
    if idx % 100 == 0:
        print(idx, "predict finished!")
    predict = downstream_model.predict_fidelity(cir)
    print(predict)
with open("downstream_model_18.pkl","wb")as f:
    pickle.dump(downstream_model,f)
    
find_error_path(upstream_model, downstream_model.error_params)


fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
duration_X, duration2circuit_index = plot_duration_fidelity(fig, axes, test_dataset)
fig.savefig("duration_fidelity_step2.svg")  # step
