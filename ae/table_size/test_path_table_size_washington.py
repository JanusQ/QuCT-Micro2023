import numpy as np
import matplotlib.pyplot as plt
from circuit import gen_random_circuits
from upstream import RandomwalkModel
from utils.backend import *

for step in range(1, 2):
    fig, ax = plt.subplots(figsize=(7, 5))
    X_qubits, Y_table_size = [], []
    for n_qubits in range(5, 128, 1):
        topology = gen_washington_topology(n_qubits)  # 3x3 9 qubits
        neighbor_info = get_washington_neighbor_info(topology, 1)
        coupling_map = topology_to_coupling_map(topology)
        # print(neigh_info)

        backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info,
                          basis_single_gates=default_basis_single_gates, basis_two_gates=default_basis_two_gates,
                          divide=False, decoupling=False)

        dataset = gen_random_circuits(min_gate=n_qubits * 10, max_gate=n_qubits * 20, n_circuits=1000 // n_qubits,
                                      two_qubit_gate_probs=[2, 3], gate_num_step=10, backend=backend,
                                      multi_process=True)

        upstream_model = RandomwalkModel(step, 20, backend=backend, travel_directions=('parallel', 'former'))

        upstream_model.train(dataset, multi_process=True, remove_redundancy=True)

        counts = [list(table.keys())[-1] for device, table in upstream_model.device2reverse_path_table.items()]
        count = np.array(counts).mean()

        print((n_qubits, step), count)
        X_qubits.append(n_qubits)
        Y_table_size.append(count)
    ax.plot(X_qubits, Y_table_size)
    ax.set_xlabel('qubits')
    ax.set_ylabel('table_size')
    fig.show()
    fig.savefig(f'washington_step{step}.svg')
