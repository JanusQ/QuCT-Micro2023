from downstream.synthesis.synthesis_baseline.experiment_tool.experiment_runner import ExperimentRunner

from circuit.algorithm.get_data import get_dataset_alg_component

from utils.backend import Backend, gen_fulllyconnected_topology

from downstream.synthesis.quct_synthesis.tensor_network_op_jax import layer_circuit_to_matrix

from utils.ray_func import wait

import ray


@ray.remote
def remote_run(name, unitary):
    synthesiser_list = ['CSDSynthesis', 'QSDSynthesis', 'QiskitSynthesis', 'QFastSynthesis' ]
    runner = ExperimentRunner(name, unitary, synthesiser_list,'alg')
    runner.run()


futures = []
n_qubits = 3
topology = gen_fulllyconnected_topology(n_qubits)
neigh_info = gen_fulllyconnected_topology(n_qubits)
target_backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
                         basis_two_gates=['cz'], divide=False, decoupling=False)
dataset = get_dataset_alg_component(n_qubits, target_backend)

for circuit_info in dataset:
    unitary = layer_circuit_to_matrix(circuit_info['layer2gates'], n_qubits)
    name = circuit_info['id']

    futures.append(remote_run.remote(name, unitary))

wait(futures)
