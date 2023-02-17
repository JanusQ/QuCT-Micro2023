import numpy.random

from circuit import gen_random_circuits, label_ground_truth_fidelity
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import NoiseSimulator
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates

topology = gen_grid_topology(3) # 3x3 9 qubits
neigh_info = get_grid_neighbor_info(3, 1)
coupling_map = topology_to_coupling_map(topology)
n_qubits = 9

backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates = default_basis_single_gates,
                  basis_two_gates = default_basis_two_gates, divide = False, decoupling=False)
upstream_model = RandomwalkModel(1, 20, backend = backend)

train_dataset = gen_random_circuits(min_gate = 10, max_gate = 40, n_circuits = 6, two_qubit_gate_probs=[4, 8],backend = backend)
upstream_model.train(train_dataset, multi_process = True)


simulator = NoiseSimulator(backend)
simulator.get_error_results(train_dataset, upstream_model)
# label_ground_truth_fidelity(train_dataset,numpy.random.rand(64))
import pickle
with open("upstream_model.pkl","wb") as f:
     pickle.dump(upstream_model, f)


# import pickle
#
# with open("upstream_model.pkl", "rb") as f:
#     upstream_model = pickle.load(f)

downstream_model = FidelityModel()
downstream_model.train(upstream_model.dataset, upstream_model.device2reverse_path_table_size)
with open("downstream_model.pkl", "wb") as f:
    pickle.dump(downstream_model, f)



test_dataset = gen_random_circuits(min_gate=10, max_gate=40, n_circuits=1, two_qubit_gate_probs=[4, 8],
                                   backend=upstream_model.backend)
for cir in test_dataset:
    cir = upstream_model.vectorize(cir)
    predict, circuit_info, gate_errors = downstream_model.predict_fidelity(cir)
    print(predict)
