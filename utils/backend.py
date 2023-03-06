from utils.backend_info import *
import math
import copy
import itertools as it


def gen_grid_topology(size):
    '''
    Example:
    0   1   2
    3   4   5
    6   7   8
    '''
    topology = defaultdict(list)

    for x in range(size):
        for y in range(size):
            qubit = x * size + y
            for neigh_x in range(x - 1, x + 2):
                neigh_y = y
                if neigh_x < 0 or neigh_x >= size or x == neigh_x:
                    continue
                neigh_qubit = neigh_x * size + neigh_y
                topology[qubit].append(neigh_qubit)

            for neigh_y in range(y - 1, y + 2):
                neigh_x = x
                if neigh_y < 0 or neigh_y >= size or y == neigh_y:
                    continue
                neigh_qubit = neigh_x * size + neigh_y
                topology[qubit].append(neigh_qubit)

    for qubit, coupling in topology.items():
        coupling.sort()

    return topology


def get_grid_neighbor_info(size, max_distance):
    neigh_info = defaultdict(list)

    for x in range(size):
        for y in range(size):
            qubit = x * size + y
            for neigh_x in range(x - 1, x + 2):
                for neigh_y in range(y - 1, y + 2):
                    if neigh_x < 0 or neigh_x >= size or neigh_y < 0 or neigh_y >= size or (
                            x == neigh_x and y == neigh_y):
                        continue
                    if ((x - neigh_x) ** 2 + (y - neigh_y) ** 2) <= max_distance ** 2:
                        neigh_qubit = neigh_x * size + neigh_y
                        neigh_info[qubit].append(neigh_qubit)

    return neigh_info


def gen_linear_topology(n_qubits):
    return {
        q1: [q2 for q2 in [q1-1, q1+1] if q2 >= 0 and q2 < n_qubits]
        for q1 in range(n_qubits)
    }


def get_linear_neighbor_info(n_qubits, max_distance):
    return {
        q1: [
            q2 for q2 in range(n_qubits)
            if q1 != q2 and (q1-q2)**2 <= max_distance**2
        ]
        for q1 in range(n_qubits)
    }


def topology_to_coupling_map(topology: dict) -> list:
    coupling_map = set()
    for qubit, coupling in topology.items():
        for neighbor_qubit in coupling:
            coupling = [qubit, neighbor_qubit]
            coupling.sort()
            coupling_map.add(tuple(coupling))
    return [
        list(coupling)
        for coupling in coupling_map
    ]


class Backend():
    '''
    Example:
    0-1-2
    | | |
    3-4-5
    | | |
    6-7-8

    topology: {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]})
    coupling_map: [[0, 1], [1, 2], [3, 4], [5, 8], [0, 3], [1, 4], [6, 7], [4, 5], [3, 6], [2, 5], [4, 7], [7, 8]]
    neigh_info: {0: [1, 3], 1: [0, 3], 2: [1], 3: [1, 4, 7], 4: [1, 3, 5, 7], 5: [1, 4, 7], 6: [7], 7: [5, 8], 8: [5, 7]})  
    '''

    def __init__(self, n_qubits, topology=None, neighbor_info=None, basis_single_gates: list = None,
                 basis_two_gates: list = None,
                 divide: bool = True, decoupling: bool = True, single_qubit_gate_time=30, two_qubit_gate_time=60):
        self.n_qubits = n_qubits

        if topology is None:
            topology = {
                q1: [q1 for q2 in range(n_qubits) if q1 != q2]
                for q1 in range(n_qubits)
            }

        self.topology = topology
        self.coupling_map = topology_to_coupling_map(topology)
        self.true_coupling_map = list(self.coupling_map)
        # describe qubits that have mutual interactions
        self.neighbor_info = neighbor_info  # TODO: rename to 'adjlist'

        if basis_single_gates is None:
            basis_single_gates = default_basis_single_gates

        if basis_two_gates is None:
            basis_two_gates = default_basis_two_gates

        self.basis_single_gates = basis_single_gates
        self.basis_two_gates = basis_two_gates
        self.basis_gates = self.basis_single_gates + self.basis_two_gates

        self.divide = divide  # whether a layer only has single-qubit gates or two-qubit gates
        self.decoupling = decoupling  # whether dynamic decoupling is applied

        self.single_qubit_gate_time = single_qubit_gate_time  # ns
        self.two_qubit_gate_time = two_qubit_gate_time  # ns
        
        # 随机了一个噪音
        self.single_qubit_fidelity = [
            1 - random.random() / 1000
            for q in range(n_qubits)
        ]

        self.two_qubit_fidelity = [
            1 - random.random() / 500
            for i, coupler in enumerate(self.coupling_map)
        ]

        self.qubit2T1 = [
            110 - random.random() * 20
            for q in range(n_qubits)
        ]

        self.qubit2T2 = [
            7 - random.random() * 3
            for q in range(n_qubits)
        ]

        self.cache = {}


    def get_subgraph(self, location):
        """Returns the sub_coupling_graph with qubits in location."""
        subgraph = []
        for q0, q1 in self.coupling_graph:
            if q0 in location and q1 in location:
                subgraph.append((q0, q1))
        return subgraph

    def get_sub_backend(self, sub_qubits):
        sub_backend = copy.deepcopy(self)
        sub_backend.topology = {
            qubit: [] if qubit not in sub_qubits else [
                connect_qubit for connect_qubit in connect_qubits if connect_qubit in sub_qubits]
            for qubit, connect_qubits in self.topology.items()
        }
        sub_backend.coupling_map = topology_to_coupling_map(
            sub_backend.topology)
        return sub_backend

    def get_connected_qubit_sets(self, n_qubit_set):
        """
        Returns a list of qubit sets that complies with the topology.
        """

        assert n_qubit_set < self.n_qubits and n_qubit_set > 0, (n_qubit_set, self.n_qubits)
        
        if n_qubit_set in self.cache:
            return self.cache[n_qubit_set]

        locations = []

        for group in it.combinations(range(self.n_qubits), n_qubit_set):
            # Depth First Search
            seen = set([group[0]])
            frontier = [group[0]]

            while len(frontier) > 0 and len(seen) < len(group):
                for q in group:
                    if frontier[0] in self.topology[q] and q not in seen:
                        seen.add(q)
                        frontier.append(q)

                frontier = frontier[1:]

            if len(seen) == len(group):
                locations.append(group)

        self.cache[n_qubit_set] = locations
        return locations
    
    '''TODO: 拓扑结构也得相等'''
    def __eq__(self, other):
        return self.n_qubits == other.n_qubits


'''TODO: 还没有写完'''
class FullyConnectedBackend(Backend):
    def __init__(self, n_qubits):
        topology = {}
        Backend.__init__(self, n_qubits)
        return