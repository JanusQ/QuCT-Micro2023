import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import cloudpickle as pickle
import numpy as np
import ray
from downstream.synthesis.quct_synthesis.synthesis_model_pca_unitary_jax_random_forest import SynthesisModelNN, matrix_distance_squared, \
    SynthesisModel, synthesize
import os
from circuit import gen_random_circuits
from circuit.formatter import layered_circuits_to_qiskit
from downstream.synthesis.synthesis_baseline.experiment_tool.utils import get_cnot_cnt
from ae.synthesis.load_syn_result import load_baseline_Unitary
from upstream import RandomwalkModel
from utils.backend import Backend, gen_fulllyconnected_topology


def eval(n_qubits, type, filename, U, synthesis_model: SynthesisModel, n_unitary_candidates, n_neighbors):
    use_heuristic = True

    synthesis_log = {}

    assert matrix_distance_squared(U @ U.T.conj(), np.eye(2 ** n_qubits)) < 1e-4

    synthesized_circuit, cpu_time = synthesize(U, backend=target_backend, allowed_dist=1e-2,
                                               multi_process=True,
                                               heuristic_model=synthesis_model if use_heuristic else None,
                                               verbose=True, lagre_block_penalty=4, synthesis_log=synthesis_log,
                                               n_unitary_candidates=n_unitary_candidates, timeout=3 * 3600)

    qiskit_circuit = layered_circuits_to_qiskit(n_qubits, synthesized_circuit, barrier=False)

    result = {
        'n_qubits': n_qubits,
        'U': U,
        'qiskit circuit': qiskit_circuit,
        '#gate': len(qiskit_circuit),
        '#two-qubit gate': get_cnot_cnt(qiskit_circuit),
        'depth': qiskit_circuit.depth(),
        'cpu time': cpu_time,
        'use heuristic': use_heuristic,
        'n_unitary_candidates': n_unitary_candidates,
        'n_neighbors': n_neighbors,
        'baseline_name': filename
    }
    result.update(synthesis_log)

    print('RESULT')
    print({key: item for key, item in result.items() if key not in ('print', 'qiskit circuit', 'U')})

    dir_path = os.path.join('ae/synthesis/quct', str(n_qubits), type)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    with open(os.path.join(dir_path, filename+'.pkl'), 'wb') as f:
        pickle.dump(result, f)


@ray.remote
def eval_remote(*args):
    return eval(*args)


n_qubits = 3
type = 'random'
n_step = 5
n_unitary_candidates = 5
n_neighbors = 7
if n_qubits <= 4:
    n_neighbors = 10

synthesis_model_name = f'synthesis_{n_qubits}_{n_step}_{n_neighbors}NN'

try:
    synthesis_model: SynthesisModel = SynthesisModel.load(n_qubits, synthesis_model_name)
    target_backend: Backend = synthesis_model.backend
except Exception as e:
    print(e)
    topology = gen_fulllyconnected_topology(n_qubits)
    neigh_info = gen_fulllyconnected_topology(n_qubits)
    target_backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
                             basis_two_gates=['cz'], divide=False, decoupling=False)

    min_gate, max_gate = max([4 ** n_qubits - 100, 10]), 4 ** n_qubits
    dataset = gen_random_circuits(min_gate=min_gate, max_gate=max_gate, gate_num_step=max_gate // 20, n_circuits=10,
                                  two_qubit_gate_probs=[4, 7], backend=target_backend, reverse=False, optimize=True,
                                  multi_process=True, circuit_type='random')

    upstream_model = RandomwalkModel(
        n_step, 4 ** n_step, target_backend, travel_directions=('parallel', 'next'))
    upstream_model.train(dataset, multi_process=True,
                         remove_redundancy=False)

    synthesis_model = SynthesisModelNN(upstream_model, synthesis_model_name)
    data = synthesis_model.construct_data(dataset, multi_process=True)
    print(f'data size of {synthesis_model_name} is {len(data[0])}')
    synthesis_model.construct_model(data, n_neighbors)
    synthesis_model.save(n_qubits)


filenames, unitaries = load_baseline_Unitary(n_qubits, type)
for filename, U in zip(filenames, unitaries):
    eval_remote.remote(n_qubits, type, filename, U, synthesis_model, n_unitary_candidates, n_neighbors)
    # eval(n_qubits, type, filename, U, synthesis_model, n_unitary_candidates, n_neighbors)


