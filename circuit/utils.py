import copy
import numpy as np

from circuit.parser import get_circuit_duration


def get_extra_info(dataset):
    def get_layer_type_divide(layer2instructions):
        """
        返回layer2instructions的每层的类型，是单比特门层则为1；否则则为2
        """
        return [len(layer[0]['qubits']) for layer in layer2instructions]

    def get_couple_prop(circuit_info):
        couple = 0
        for ins in circuit_info['gates']:
            if len(ins['qubits']) == 2:
                couple += 1
        return couple / (len(circuit_info['gates']))

    def get_xeb_fidelity(circuit_info):
        fidelity = 1
        for instruction in circuit_info['instructions']:
            if len(instruction['qubits']) == 2:
                q0_id, q1_id = instruction['qubits'][0], instruction['qubits'][1]
                if q0_id > q1_id:
                    fidelity = fidelity * (1 - couple_average_error[q1_id])
                else:
                    fidelity = fidelity * (1 - couple_average_error[q0_id])
            else:
                q0_id = instruction['qubits'][0]
                fidelity = fidelity * (1 - single_average_error[q0_id])
        return fidelity * np.product((measure0_fidelity + measure1_fidelity) / 2)

    def get_circuit_duration(layer2instructions):
        single_gate_time = 30
        two_gate_time = 60
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

    qubits = ['q3_15', 'q3_17', 'q3_19', 'q5_19', 'q7_19', ]

    single_average_error = np.array([0.084, 0.040, 0.083, 0.025, 0.037, ]) / 100  # 原先是带%的

    couple_average_error = np.array([0.6, 0.459, 0.537, 0.615, ]) / 100  # q1_q2, q2_q3, ..., q9_q10

    t1s = np.array([94.5, 124.1, 117.1, 124.8, 136.3])  # q1, q2, ..., q5

    t2s = np.array([5.04, 7.63, 5.67, 6.97, 4.40, ])  # q1, q2, ..., q10

    measure0_fidelity = np.array([0.97535460, 0.97535460, 0.9645634, 0.9907482, 0.96958333])
    measure1_fidelity = np.array([0.955646258, 0.97572327, 0.950431034, 0.9629411764, 0.9570833333])

    for cir in dataset:
        # cir['xeb_fidelity'] = get_xeb_fidelity(cir)
        cir['duration'] = get_circuit_duration(cir['layer2gates'])
        cir['prop'] = get_couple_prop(cir)

    return dataset

def label_ground_truth_fidelity(dataset, labels):
    assert len(dataset) == len(labels)
    assert isinstance(dataset[0], dict)
    for idx, cir in enumerate(dataset):
        cir['ground_truth_fidelity'] = labels[idx]

def get_xeb_fidelity(dataset):
    single_average_error = np.array([0.084, 0.040, 0.083, 0.025, 0.037, ]) / 100  # 原先是带%的
    couple_average_error = np.array([0.6, 0.459, 0.537, 0.615, ]) / 100  # q1_q2, q2_q3, ..., q9_q10

    measure0_fidelity = np.array([0.97535460, 0.97535460, 0.9645634, 0.9907482, 0.96958333])
    measure1_fidelity = np.array([0.955646258, 0.97572327, 0.950431034, 0.9629411764, 0.9570833333])
    
    
    xebs = []
    for circuit_info in dataset:
        fidelity = 1
        for instruction in circuit_info['gates']:
            if len(instruction['qubits']) == 2:
                q0_id, q1_id = instruction['qubits'][0], instruction['qubits'][1]
                if q0_id > q1_id:
                    fidelity = fidelity * (1 - couple_average_error[q1_id])
                else:
                    fidelity = fidelity * (1 - couple_average_error[q0_id])
            else:
                q0_id = instruction['qubits'][0]
                fidelity = fidelity * (1 - single_average_error[q0_id])
        xebs.append(fidelity * np.product((measure0_fidelity + measure1_fidelity) / 2))
    return np.array(xebs)
    
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
            if len(new_layer) != 0:
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
        
        circ_let['duration'] = get_circuit_duration(circ_let['layer2gates'])
        
        
        for idx, gate in enumerate(circ_let['gates']):
            gate['id'] = idx
            
        layer2gates = []
        idx = 0
        for layer, gates in enumerate(circ_let['layer2gates']):
            new_layer = []
            for _ in range(len(circ_let['layer2gates'][layer])):
                circ_let['gate2layer'][idx] = layer
                new_layer.append(circ_let['gates'][idx])
                idx += 1
            layer2gates.append(new_layer)
        circ_let['layer2gates'] = layer2gates
        

        result.append(circ_let)
    
    return result
    
def stitching(n_qubits, cut_datasets, devide_qubits, maps, reverse_maps,align = False):
    stitching_dataset = [] 
    
    for cir_idx in range(len(cut_datasets[0])):
        circuit_info = {}
        circuit_info['gates'], circuit_info['layer2gates'], circuit_info['gate2layer']= [],[],[]
        
        layer = 0
        gate_id = 0
        while True:
            new_layer = []
            for cut_dataset, map, reverse_map in zip(cut_datasets, maps, reverse_maps):
                cut_dataset[cir_idx]['map'] = map
                cut_dataset[cir_idx]['reverse_map'] = reverse_map
                
                if layer < len(cut_dataset[cir_idx]['layer2gates']):
                    for gate in cut_dataset[cir_idx]['layer2gates'][layer]:
                        new_gate = copy.deepcopy(gate)
                        new_gate['id'] = gate_id  
                        gate_id += 1
                        new_gate['qubits'] = [map[qubit] for qubit in new_gate['qubits'] ]
                        new_layer.append(new_gate)
                        circuit_info['gates'].append(new_gate)
                        circuit_info['gate2layer'].append(layer)
            if len(new_layer) != 0:
                circuit_info['layer2gates'].append(new_layer)
                layer += 1
            else:
                break
            
        circuit_info['devide_qubits'] = devide_qubits
        circuit_info['gate_num'] = len(circuit_info['gates'])
        circuit_info['duration'] = get_circuit_duration(circuit_info['layer2gates'])
        circuit_info['num_qubits'] = n_qubits
        
        if align:
            circuit_info['layer2gates'].reverse()
            max_layer = len(circuit_info['layer2gates']) -1 
            circuit_info['gate2layer'] = [max_layer - layer for layer in circuit_info['gate2layer'] ] 
            circuit_info['max_layer'] = max_layer
        
        stitching_dataset.append(circuit_info)
        # qc = layered_circuits_to_qiskit(n_qubits, circuit_info['layer2gates'])
        # fig = qc.draw('mpl')
        # fig.savefig("stitching_figure/"+str(devide_qubits)+'_'+str(cir_idx)+".png")
        # print(len(cut_datasets[0][cir_idx]['layer2gates']), len(cut_datasets[1][cir_idx]['layer2gates']), len(cut_datasets[2][cir_idx]['layer2gates']),len(cut_datasets[3][cir_idx]['layer2gates']))
    return stitching_dataset
  
