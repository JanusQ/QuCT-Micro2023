import copy
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import ray
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import CrosstalkAdaptiveSchedule

from circuit.dataset_loader import gen_algorithms
from circuit.formatter import layered_circuits_to_qiskit, get_layered_instructions, qiskit_to_my_format_circuit
from circuit.parser import qiskit_to_layered_circuits
from downstream.fidelity_predict.baseline.rb import get_errors as get_errors_rb
from downstream.fidelity_predict.fidelity_analysis import FidelityModel
from simulator.noise_simulator import NoiseSimulator
from upstream.randomwalk_model import RandomwalkModel, extract_device
from utils.backend import Backend, gen_linear_topology, topology_to_coupling_map
from utils.ray_func import wait


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
    
def opt_move(circuit_info, downstream_model, threshold):
    upstream_model = downstream_model.upstream_model 
    
    new_circuit = {}         
    new_circuit['gates'] = []
    new_circuit['layer2gates'] = []
    new_circuit['gate2layer'] = []
    new_circuit['id'] = circuit_info['id']
    new_circuit['num_qubits'] = circuit_info['num_qubits']
    new_circuit['gate_num'] = len(circuit_info['gates'])
    # new_circuit['ground_truth_fidelity'] = circuit_info['ground_truth_fidelity']

    cur_layer = [0 for i in range(circuit_info['num_qubits'])]
    
    pre_fidelity = 1
    id = 0
    cnt = 0
    for layergates in circuit_info['layer2gates']:
        for _gate in layergates:
            
            gate = copy.deepcopy(_gate)
            new_circuit['gates'].append(gate)
            new_circuit['gate2layer'].append(-1)
            gate['id'] = id
            id += 1
            qubits = gate['qubits']
            
            offset = 0
            while True:
                if len(qubits) == 1:
                    qubit = qubits[0]
                    insert_layer = cur_layer[qubit] + offset
                    new_circuit['gate2layer'][-1] = insert_layer
                    if insert_layer >= len(new_circuit['layer2gates']):
                        assert insert_layer == len(new_circuit['layer2gates'])
                        new_circuit['layer2gates'].append([gate])
                    else:
                        new_circuit['layer2gates'][insert_layer].append(gate)
                        
                else:
                    qubit0 = qubits[0]
                    qubit1 = qubits[1]
                    insert_layer = max(cur_layer[qubit0],cur_layer[qubit1]) + offset
                    new_circuit['gate2layer'][-1] = insert_layer
                    if insert_layer >= len(new_circuit['layer2gates']):
                        assert insert_layer == len(new_circuit['layer2gates'])
                        new_circuit['layer2gates'].append([gate])
                    else:
                        new_circuit['layer2gates'][insert_layer].append(gate)
                
                
                
                new_circuit = upstream_model.vectorize(new_circuit)
                new_circuit['n_erroneous_patterns'] = count_error_path_num(new_circuit,upstream_model)
                # new_circuit['duration'] = get_circuit_duration(new_circuit['layer2gates'])
                new_fidelity = downstream_model.predict_fidelity(new_circuit)
                new_fidelity = new_fidelity if new_fidelity < 1 else 1
                # print(pre_fidelity,new_fidelity)
                
                if offset > 5 or pre_fidelity - new_fidelity < threshold:
                    if offset > 5:
                        print('threshold too small')
                    pre_fidelity = new_fidelity
                    if len(qubits) == 1:
                        cur_layer[qubit] = insert_layer + 1
                    else:
                        cur_layer[qubit0] = insert_layer+ 1
                        cur_layer[qubit1] = insert_layer+ 1
                    break
                else:
                    new_circuit['layer2gates'][insert_layer].remove(gate)
                    offset += 1
                    cnt += 1
    
    upstream_model.vectorize(circuit_info)
    print(new_circuit['id'], 'predict:', downstream_model.predict_fidelity(circuit_info), '--->', new_fidelity)
    print(cnt)
    return new_circuit


@ray.remote
def opt_move_remote(circuit_info, downstream_model, threshold):
    return opt_move(circuit_info, downstream_model, threshold)


def schedule_crosstalk(backend: Backend, rb_error: list, crosstalk_prop: list, dataset):
    class QiskitBackendProperty():
        def __init__(self, backend: Backend) -> None:
            self.backend = backend
            self.qubits = list(range(backend.n_qubits))

            class _Gate():
                def __init__(self, gate, qubits) -> None:
                    self.gate = gate
                    self.qubits = list(qubits)

            self.gates = [
                _Gate(gate, qubits)
                for gate in backend.basis_two_gates
                for qubits in backend.coupling_map
            ] + [
                _Gate(gate, [qubit])
                for gate in backend.basis_single_gates
                for qubit in self.qubits
            ]
            return

        def t1(self, qubit):
            return self.backend.qubit2T1[qubit]

        def t2(self, qubit):
            return self.backend.qubit2T2[qubit]

        def gate_length(self, gate, qubits):
            if gate in self.backend.basis_single_gates or gate in ('u1', 'u2', 'u3'):
                return self.backend.single_qubit_gate_time
            elif gate in self.backend.basis_two_gates:
                return self.backend.two_qubit_gate_time

        def gate_error(self, gate, qubits):
            if gate in self.backend.basis_single_gates or gate in ('u1', 'u2', 'u3'):
                return self.backend.rb_error[0][qubits]
            elif gate in self.backend.basis_two_gates:
                return self.backend.rb_error[1][tuple(qubits)]
            else:
                raise Exception('known', gate, qubits)

    # 门都还得转成 'u1', 'u2', 'u3'
    def to_crosstalk_structure(layer2gates):
        layer2gates = copy.deepcopy(layer2gates)
        for layer in layer2gates:
            for gate in layer:
                if gate['name'] == 'rz':
                    gate['name'] = 'u1'
                if gate['name'] == 'rx':
                    gate['name'] = 'u2'
                    gate['params'] = gate['params'] + [0]
                if gate['name'] == 'ry':
                    gate['name'] = 'u3'
                    gate['params'] = gate['params'] + [0, 0]
                if gate['name'] == 'cz':
                    gate['name'] = 'cx'
        return layer2gates

    def from_crosstalk_structure(layer2gates):
        layer2gates = copy.deepcopy(layer2gates)
        for layer in layer2gates:
            for gate in layer:
                if gate['name'] == 'u1':
                    gate['name'] = 'rz'
                    gate['params'] = gate['params'][:1]
                if gate['name'] == 'u2':
                    gate['name'] = 'rx'
                    gate['params'] = gate['params'][:1]
                if gate['name'] == 'u3':
                    gate['name'] = 'ry'
                    gate['params'] = gate['params'][:1]
                if gate['name'] == 'cx':
                    gate['name'] = 'cz'
        return layer2gates

    '''这个使用rb跑出来的'''
    two_qubit_error = rb_error[1]
    backend.rb_error = (rb_error[0],
                        {tuple(coupler): two_qubit_error[index]
                         for index, coupler in enumerate(backend.coupling_map)})

    crosstalk_bakcend = copy.deepcopy(backend)
    crosstalk_bakcend.basis_single_gates = ['u1', 'u2', 'u3']
    crosstalk_bakcend.basis_two_gates = ['cx']
    crosstalk_bakcend.basis_gates = crosstalk_bakcend.basis_single_gates + crosstalk_bakcend.basis_two_gates
    
    backend_property = QiskitBackendProperty(crosstalk_bakcend)
    crosstalk_scheduler = CrosstalkAdaptiveSchedule(
        backend_property, crosstalk_prop=crosstalk_prop)

    optimized_circuits = []
    for circuit_info in dataset:
        
        qiskit_circuit = layered_circuits_to_qiskit(
            n_qubits, to_crosstalk_structure(circuit_info['layer2gates']), barrier=False)
        # transpiled_circuit = transpile()
        transpiled_circuit = dag_to_circuit(crosstalk_scheduler.run(
            circuit_to_dag(qiskit_circuit)))

        transpiled_circuit = qiskit_to_my_format_circuit(
            get_layered_instructions(transpiled_circuit)[0])
        
        
        transpiled_circuit = from_crosstalk_structure(transpiled_circuit[0])
        transpiled_circuit = layered_circuits_to_qiskit(n_qubits, transpiled_circuit)

        optimized_circuit_info = qiskit_to_layered_circuits(transpiled_circuit, False, False)
        
        # optimized_circuit_info['layer2gates'] =  transpiled_circuit
        # optimized_circuit_info['gate2layer'] = transpiled_circuit[1]
        # optimized_circuit_info['gates'] = transpiled_circuit[2]
        
        optimized_circuits.append(optimized_circuit_info)
        '''check一下对不对'''
        # print(transpiled_circuit)
        # print(simulate_noise_free(
        #     layered_circuits_to_qiskit(n_qubits, transpiled_circuit)))

    return optimized_circuits
def run_simulate(dataset):
    vecs = []
    for cir in dataset:
        veced_cir = upstream_model.vectorize(cir)
        vecs.append(veced_cir)
        downstream_model.predict_fidelity(veced_cir)
    simulator.get_error_results(vecs, upstream_model, multi_process=True, erroneous_pattern=upstream_model.erroneous_pattern)
    reals = []
    for cir in vecs:
        reals.append(cir['ground_truth_fidelity'])
        
    return np.array(reals)
        
if __name__ == '__main__':
    size = 3
    n_qubits = 5
    n_steps = 1

    topology = gen_linear_topology(n_qubits)
    coupling_map = topology_to_coupling_map(topology)
    neighbor_info = copy.deepcopy(topology)
    
    
    backend = Backend(n_qubits=5, topology=topology, coupling_map=coupling_map, neighbor_info=neighbor_info)
    
    '''要跑的'''
    backend.routing = 'basic' #'lookahead'
    backend.optimzation_level = 1
    algos_baseline = gen_algorithms(n_qubits, backend, mirror = True, trans= True)
    backend.routing = 'sabre'
    backend.optimzation_level = 3

        

    with open(f"quct_model.pkl", "rb")as f:
        downstream_model, predicts, reals, durations, _ = pickle.load(f)
        downstream_model: FidelityModel = downstream_model
        upstream_model: RandomwalkModel = downstream_model.upstream_model
        # backend: Backend = upstream_model.backend
        
    alg2best_fidelity = defaultdict(float)
    alg2best_circuitinfo = {}
    
    simulator = NoiseSimulator(backend)
    
    retrain = False
    if retrain:
        rb_error = get_errors_rb(backend, simulator, upstream_model = upstream_model, multi_process=True)
        all_errors = rb_error
        print(all_errors)
        single_average_error = {}
        couple_average_error = {}
        for q, e in enumerate(all_errors[0]):
            single_average_error[q] = e
        for c, e in zip(list(backend.coupling_map), all_errors[1]):
            couple_average_error[tuple(c)] = e
            
        print(single_average_error, couple_average_error)
        with open(f"rb_model.pkl", "wb")as f:
            pickle.dump((single_average_error, couple_average_error,rb_error), f)
    else:
        with open(f"rb_model.pkl", "rb")as f:
            single_average_error, couple_average_error, rb_error = pickle.load(f)
    
    
    # fig, axes = plt.subplots(figsize=(15, 10))  
    # x = [i for i in range(len(algos_baseline))]
    # x = np.array(x) * 10
    # axes.bar(x,predicts,width =2,label='predicts')
    # axes.bar(x+2,rbs,width =2,label='rbs')
    # axes.bar(x+4,reals,width =2,label='reals')
    # fig.legend()
    # fig.savefig('opt_5bit/rb_error.svg')

    # fig, axes = plt.subplots(figsize=(15, 10))  
    # x = [i for i in range(len(algos_baseline))]
    # x = np.array(x) * 10
    # axes.bar(x,predicts,width =2,label='predicts')
    # axes.bar(x+2,rbs,width =2,label='rbs')
    # axes.bar(x+4,reals,width =2,label='reals')
    # fig.legend()
    # fig.savefig('opt_5bit/rb_error.svg')

    
    for _ in range(10):
        backend.routing = 'sabre'
        for circuit_info in gen_algorithms(n_qubits, backend, mirror = True, trans= True):
            circuit_info = upstream_model.vectorize(circuit_info)
            circuit_id = circuit_info['id']
            
            fidelity_predict = downstream_model.predict_fidelity(circuit_info)

            if fidelity_predict > alg2best_fidelity[circuit_id]:
                alg2best_fidelity[circuit_id] = fidelity_predict
                alg2best_circuitinfo[circuit_id] = circuit_info
                
    
    for _ in range(100):
        backend.routing = 'stochastic'
        for circuit_info in gen_algorithms(n_qubits, backend, mirror = True, trans= True):
            circuit_info = upstream_model.vectorize(circuit_info)
            circuit_id = circuit_info['id']
            
            fidelity_predict = downstream_model.predict_fidelity(circuit_info)

            if fidelity_predict > alg2best_fidelity[circuit_id]:
                alg2best_fidelity[circuit_id] = fidelity_predict
                alg2best_circuitinfo[circuit_id] = circuit_info
        backend.routing = 'sabre'

    '''要跑的'''
    
    names = [circuit_info['alg_id'] for circuit_info in algos_baseline]
    
    algos_routing = [alg2best_circuitinfo[circuit_info['id']] for circuit_info in algos_baseline]

    '''要跑的'''
    algos_schedule = wait([opt_move_remote.remote(circuit_info, downstream_model, threshold = 0.005) for circuit_info in algos_baseline])

    '''要跑的'''
    algos_routing_schedule = wait([opt_move_remote.remote(circuit_info, downstream_model, threshold = 0.005)  for circuit_info in algos_routing]) #TODO: 0.01


    algos_satmap = gen_algorithms(n_qubits, backend, mirror = True, trans= True)

    # {(0, 1): {(2, 3): 0.2, (2): 0.15}, (2, 3): {(0, 1): 0.05, }}
    crosstalk_prop = defaultdict(lambda: {})
    for paths in upstream_model.erroneous_pattern.values():
        for path in paths:
            path = path.split('-')
            if path[1] != 'parallel':
                continue
            path = [RandomwalkModel.parse_gate_info(elm)['qubits'] for elm in path]
            crosstalk_prop[tuple(path[0])][tuple(path[2])] = 0.1
    
    '''要跑的'''
    algos_crosstalk = schedule_crosstalk(backend ,rb_error, crosstalk_prop, algos_baseline) # eval_5qubits.py
    algos_satmap_crosstalk = schedule_crosstalk(backend , rb_error, crosstalk_prop, algos_routing) # eval_5qubits.py

    reals_baseline = run_simulate(algos_baseline)
    reals_routing = run_simulate(algos_routing)
    reals_routing_schedule = run_simulate(algos_routing_schedule)
    reals_schedule = run_simulate(algos_schedule)

    reals_satmap = run_simulate(algos_satmap)
    reals_crosstalk = run_simulate(algos_crosstalk)
    reals_satmap_crosstalk = run_simulate(algos_satmap_crosstalk)
    

    fig, axes = plt.subplots(figsize=(15, 10))  
    x = [i for i in range(len(algos_baseline))]
    x = np.array(x) * 20
    axes.bar(x,reals_baseline,width =2,label='reals_baseline')

    axes.bar(x+2,reals_routing,width =2,label='reals_routing')
    axes.bar(x+4,reals_schedule,width =2,label='reals_schedule')
    axes.bar(x+6,reals_routing_schedule,width =2,label='reals_routing_schedule')

    axes.bar(x+8,reals_satmap,width =2,label='reals_satmap')
    axes.bar(x+10,reals_crosstalk,width =2,label='reals_crosstalk')
    axes.bar(x+12,reals_satmap_crosstalk,width =2,label='reals_satmap_crosstalk')
    axes.set_xticks(x)
    axes.set_xticklabels(names)
    fig.legend()
    fig.savefig('reals.svg')

    print(reals_baseline)

    print(reals_routing)
    print(reals_schedule)
    print(reals_routing_schedule)
    

    print(reals_satmap)
    print(reals_crosstalk)
    print(reals_satmap_crosstalk)

    reals_baseline = np.array(reals_baseline) 
    reals_routing = np.array(reals_routing)
    reals_schedule = np.array(reals_schedule)
    reals_routing_schedule = np.array(reals_routing_schedule)
    reals_satmap = np.array(reals_satmap)
    reals_crosstalk = np.array(reals_crosstalk)
    reals_satmap_crosstalk = np.array(reals_satmap_crosstalk)


    fig, axes = plt.subplots(figsize=(15, 10))  
    x = [i for i in range(len(algos_baseline))]
    x = np.array(x) * 20
    # axes.bar(x,reals_baseline,width =2,label='reals_baseline')

    axes.bar(x+2,reals_routing/reals_baseline,width =2,label='reals_routing')
    axes.bar(x+4,reals_schedule/reals_baseline,width =2,label='reals_schedule')
    axes.bar(x+6,reals_routing_schedule/reals_baseline,width =2,label='reals_routing_schedule')

    axes.bar(x+8,reals_satmap/reals_baseline,width =2,label='reals_satmap')
    axes.bar(x+10,reals_crosstalk/reals_baseline,width =2,label='reals_crosstalk')
    axes.bar(x+12,reals_satmap_crosstalk/reals_baseline,width =2,label='reals_satmap_crosstalk')
    axes.set_xticks(x)
    axes.set_xticklabels(names)
    fig.legend()
    fig.savefig('improve.svg')

    # 去掉top10中的error path的 random circuits duration-fidelity
    
# [0.9375218946689947, 0.8678619379287392, 0.7413771249603797, 0.8158809962067515, 0.5067941627664263, 0.9252444215001001, 0.8471615675468609, 0.8422288126058711, 0.8450226378314304, 0.7363917910037793, 0.6873243968068411, 0.6043078849218952, 0.7455148896935385] 
# [0.8624759316444397, 0.755416989326477, 0.5409650206565857, 0.6314520835876465, 0.23517946898937225, 0.8179912567138672, 0.6598278284072876, 0.6634188890457153, 0.6558599472045898, 0.5382810235023499, 0.5378336906433105, 0.3895255923271179, 0.5392568111419678] 
# [0.9138181818181818, 0.8163636363636364, 0.34127272727272734, 0.3790909090909091, 0.10954545454545463, 0.8218181818181818, 0.7432727272727273, 0.8311818181818181, 0.318909090909091, 0.47427272727272735, 0.3651818181818181, 0.21018181818181816, 0.41300000000000003]