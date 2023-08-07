import copy
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from circuit.dataset_loader import gen_algorithms
from downstream.fidelity_predict.baseline.rb import get_errors as get_errors_rb
from downstream.fidelity_predict.fidelity_analysis import FidelityModel
from simulator.noise_simulator import NoiseSimulator
from upstream.randomwalk_model import RandomwalkModel, extract_device
from utils.backend import Backend, gen_linear_topology, topology_to_coupling_map


def get_rb_fidelity(circuit_info):
        fidelity = 1
        for gate in circuit_info['gates']:
            device = extract_device(gate)
            if isinstance(device, tuple):
                fidelity = fidelity * (1 - couple_average_error[device])
            else:
                fidelity = fidelity * (1 - single_average_error[device])
        return fidelity
    


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

    algos_baseline = gen_algorithms(n_qubits, backend, mirror = True, trans= True)
    backend.routing = 'sabre'
    backend.optimzation_level = 3


        
    with open(f"quct_model.pkl", "rb")as f:
        downstream_model, predicts, reals, durations, _ = pickle.load(f)
        downstream_model: FidelityModel = downstream_model
        upstream_model: RandomwalkModel = downstream_model.upstream_model
        
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

    reals =run_simulate(algos_baseline)
    predicts , rbs, names = [],[],[]
    for cir in algos_baseline:
        veced_cir = upstream_model.vectorize(cir)
        predicts.append(downstream_model.predict_fidelity(veced_cir))
        rbs.append(get_rb_fidelity(cir))
        names.append(cir['alg_id'])
    
    rbs = np.array(rbs)
    predicts = np.array(predicts)
    reals = np.array(reals)
    print(rbs, predicts , reals )
    print('inacc: rb', np.abs(reals-rbs).mean())
    print('inacc: quct', np.abs(reals-predicts).mean())
    
    fig, axes = plt.subplots(figsize=(15, 10))  
    x = [i for i in range(len(algos_baseline))]
    x = np.array(x) * 10
    axes.bar(x,predicts,width =2,label='quct')
    axes.bar(x+2,rbs,width =2,label='rbs')
    axes.bar(x+4,reals,width =2,label='reals')
    axes.set_xticks(x)
    axes.set_xticklabels(names)
    fig.legend()
    fig.savefig('test_predict_alg.svg')    
    
    
    fig, axes = plt.subplots(figsize=(15, 10))  
    x = [i for i in range(len(algos_baseline))]
    x = np.array(x) * 6
    axes.bar(x, np.abs(predicts - reals), width =2,label='quct')
    axes.bar(x+2, np.abs(rbs - reals), width =2,label='rbs')
    axes.set_xticks(x)
    axes.set_xticklabels(names)
    fig.legend()
    fig.savefig('test_predict_alg_delta.svg')
    
    