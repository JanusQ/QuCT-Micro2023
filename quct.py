import statistics

import numpy as np
from circuit.quct.analysis.utils import get_extra_info, smart_predict, error_param_rescale, func_dist
from circuit.quct.dataset.dataset_loader import load_randomcircuits
from circuit.quct.pattern_extractor.randomwalk_model import RandomwalkModel
from jax import numpy as jnp, vmap
from qiskit import QuantumCircuit
import pickle

model = RandomwalkModel.load('model.pkl')
model_bug = RandomwalkModel.load('model_bug_step1.pkl')
with open('circuit/quct/pattern_extractor/model/all_ins_vec_step1_fix_sparse.pkl', 'rb')as f:
    num_qubits2instructions, num_qubits2positive_vecs = pickle.load(f)


def generate_circtuit(max_qubit_num, min_gate, max_gate, n_circuits, devide, require_decoupling):
    dataset = []
    for n_gates in range(min_gate, max_gate, 5):
        for prob in range(4, 8):
            prob *= .1
            dataset += load_randomcircuits(n_qubits=max_qubit_num, n_gates=n_gates, two_qubit_prob=prob,
                                           n_circuits=n_circuits, devide=devide, require_decoupling=require_decoupling)

    dataset = get_extra_info(dataset)
    return dataset


def train_model(dataset):

    print(f'load {len(dataset)} circuits')

    model = RandomwalkModel(1, 15)
    model.batch_train(dataset)
    print(
        f'succeed in training, and generate {len(model.hash_table)} path type.')
    model.load_reduced_vecs()
    model.load_error_params()
    return model


def predict_fidelity(qc: QuantumCircuit):
    veced = model.vectorize(qc)
    error_params = model.error_params
    circuit_predict = smart_predict(error_params, veced['reduced_vecs'])
    gate_errors = np.array([
        jnp.dot(error_params/error_param_rescale, vec)
        for vec in veced['reduced_vecs']
    ])[:, 0]
    veced['gate_errors'] = gate_errors
    veced['circuit_predict'] = circuit_predict
    return veced, circuit_predict, gate_errors


def find_bug(circuit):  # bug_instructions是手动添加的bug的id

    circuit_info = model_bug.vectorize(circuit)
    gate_vecs = circuit_info['sparse_vecs']  # .reshape(-1, 100)
    num_qubits = circuit_info['num_qubits']
    instruction2nearest_circuits = []

    for analyzed_vec_index, analyzed_vec in enumerate(gate_vecs):

        dists = np.array(vmap(func_dist, in_axes=(0, None), out_axes=0)(
            num_qubits2positive_vecs[num_qubits], analyzed_vec))

        dist_indexs = np.argsort(dists)[:3]  # 小的在前面
        nearest_dists = dists[dist_indexs]

        dist_indexs = dist_indexs[nearest_dists < 2]

        nearest_positive_instructions = [
            num_qubits2instructions[num_qubits][_index]
            for _index in dist_indexs
        ]
        nearest_circuits = [
            elm[2]['id']
            for elm in nearest_positive_instructions
        ]

        nearest_circuits = set(nearest_circuits)

        instruction2nearest_circuits.append(nearest_circuits)

    bug_positions = []
    for index, nearest_circuits in enumerate(instruction2nearest_circuits):

        neighbor_nearest_circuits = []
        pre = 0 if (index - 6) < 0 else index - 6
        for nearest_circuit_set in instruction2nearest_circuits[pre:index] + instruction2nearest_circuits[index + 1: index + 6]:
            neighbor_nearest_circuits += list(nearest_circuit_set)

        # isbug = True
        # for nearest_circuit in nearest_circuits:
        #     if nearest_circuit in neighbor_nearest_circuits:
        #         isbug = False
        #         break
        # if isbug:
        #     bug_indexes.append(index)
        
        neighbor_mode_nearest_circuit1 = statistics.mode(neighbor_nearest_circuits)
        neighbor_nearest_circuits = [elm for elm in neighbor_nearest_circuits if elm != neighbor_mode_nearest_circuit1]
        
        if len(neighbor_nearest_circuits) > 0:
            neighbor_mode_nearest_circuit2 = statistics.mode(neighbor_nearest_circuits)
        else:
            neighbor_mode_nearest_circuit2 = None

        if neighbor_mode_nearest_circuit1 not in nearest_circuits:
            if neighbor_mode_nearest_circuit2 is None:
                bug_positions.append((circuit_info['instructions'][index]['qubits'][1] if len(circuit_info['instructions'][index]['qubits']) == 2 
                                      else circuit_info['instructions'][index]['qubits'][0]
                                      ,circuit_info['instruction2layer'][index]))
            elif neighbor_mode_nearest_circuit2 not in nearest_circuits:
                bug_positions.append((circuit_info['instructions'][index]['qubits'][1] if len(circuit_info['instructions'][index]['qubits']) == 2 
                                      else circuit_info['instructions'][index]['qubits'][0]
                                      ,circuit_info['instruction2layer'][index]))


    return bug_positions
