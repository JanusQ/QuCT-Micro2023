import json
import os
import cloudpickle as pickle

import numpy as np
from qiskit import QuantumCircuit

from downstream.synthesis.synthesis_baseline.experiment_tool.utils import get_depth, get_cnot_cnt


def load_quct(n_qubits, type, verbose=False):
    path = os.path.join('ae/synthesis/quct/', str(n_qubits), type)

    all_res = []
    for filename in os.listdir(path):
        # print(filename)
        with open(os.path.join(path, filename), 'rb') as f:
            all_res.append(pickle.load(f))

    gate_nums, cnot_nums, depths, times = [], [], [], []
    for ele in all_res:
        qc = ele['qiskit circuit']
        gate_num = len(qc)
        cnot_num = get_cnot_cnt(qc)
        depth = get_depth(qc)
        if 'cpu time' in ele:
            time = ele['cpu time']
        else:
            time = ele['synthesis_time']

        gate_nums.append(gate_num)
        cnot_nums.append(cnot_num)
        depths.append(depth)
        times.append(time)

        if verbose:
            print(' gete_num:', gate_num, ' cnot_num:', cnot_num, 'depth:', depth, ' time:', time, )
    gate_nums = np.array(gate_nums)
    cnot_nums = np.array(cnot_nums)
    depths = np.array(depths)
    times = np.array(times)
    print('avg', ' gete_num:', gate_nums.mean(), ' cnot_num:', cnot_nums.mean(), ' depth:', depths.mean(), ' time:',
          times.mean(), )

    return gate_nums.mean(), cnot_nums.mean(), depths.mean(), times.mean()


def laod_qgd(n_qubits, type, verbose=False):
    path = f'ae/synthesis/qgd/{n_qubits}_qubit_res_{type}.pkl'

    with open(path, 'rb') as f:
        all_res = pickle.load(f)

    gate_nums, cnot_nums, depths, times = [], [], [], []
    for name, v in all_res.items():
        qc, time, _ = v

        gate_num = len(qc)
        cnot_num = get_cnot_cnt(qc)
        depth = get_depth(qc)

        gate_nums.append(gate_num)
        cnot_nums.append(cnot_num)
        depths.append(depth)
        times.append(time)

        if verbose:
            print(name, ' gete_num:', gate_num, ' cnot_num:', cnot_num, ' depth:', depth, ' time:', time)
    gate_nums = np.array(gate_nums)
    cnot_nums = np.array(cnot_nums)
    depths = np.array(depths)
    times = np.array(times)
    print('avg', ' gete_num:', gate_nums.mean(), ' cnot_num:', cnot_nums.mean(), ' depth:', depths.mean(), ' time:',
          times.mean())
    return gate_nums.mean(), cnot_nums.mean(), depths.mean(), times.mean()


def load_baseline(n_qubits, type, synthesiser, verbose=False):
    path = os.path.join('ae/synthesis/baseline/', str(n_qubits), type)
    all_res = []
    for filename in os.listdir(path):
        # print(filename)
        with open(os.path.join(path, filename), 'rb') as f:
            all_res.append(json.load(f))

    gate_nums, cnot_nums, depths, times = [], [], [], []
    for ele in all_res:
        for Metric in ele['Metrics']:
            if Metric['Synthesiser'] != synthesiser:
                continue
            qc = QuantumCircuit.from_qasm_str(Metric['Circuit'])
            gate_num = len(qc)
            cnot_num = get_cnot_cnt(qc)
            depth = get_depth(qc)
            if 'CPU time' in Metric:
                time = Metric['CPU time']
            else:
                time = Metric['Execution Time']

            gate_nums.append(gate_num)
            cnot_nums.append(cnot_num)
            depths.append(depth)
            times.append(time)

            if verbose:
                print(' gete_num:', gate_num, ' cnot_num:', cnot_num, ' depth:', depth, ' time:', time, )
    gate_nums = np.array(gate_nums)
    cnot_nums = np.array(cnot_nums)
    depths = np.array(depths)
    times = np.array(times)
    print('avg', ' gete_num:', gate_nums.mean(), ' cnot_num:', cnot_nums.mean(), ' depth:', depths.mean(), ' time:',
          times.mean())

    return gate_nums.mean(), cnot_nums.mean(), depths.mean(), times.mean()


def load_baseline_Unitary(n_qubits, type):
    path = os.path.join('ae/synthesis/baseline/', str(n_qubits), type)
    unitaries = []
    filenames = []
    for filename in os.listdir(path):
        print(filename)
        filenames.append(filename)
        with open(os.path.join(path, filename), 'rb') as f:
            result_dict = json.load(f)

        # Deserialize fields
        picked_unitary = json.loads(result_dict['Unitary']).encode('latin-1')
        unitary = pickle.loads(picked_unitary)
        # print(unitary)
        unitaries.append(unitary)

    return filenames, unitaries


if __name__ == '__main__':
    n_qubits = 3
    type = 'random'
    synthesiser = 'QFast Synthesiser'
    load_baseline(n_qubits, type, synthesiser)
    load_quct(n_qubits, type, )
    # laod_qgd(n_qubits, type, )
    filenames, unitaries = load_baseline_Unitary(n_qubits, type)
    print(filenames)


# avg  gete_num: 45.6  cnot_num: 14.2  depth: 29.4  time: 651.219592165947
# avg  gete_num: 57.0  cnot_num: 18.0  depth: 37.0  time: 364.21331002665966
# avg  gete_num: 56.1  cnot_num: 17.7  depth: 36.4  time: 339.6948023623105