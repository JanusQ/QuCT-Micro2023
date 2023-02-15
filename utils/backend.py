from utils.backend_info import *
import math

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
            for neigh_x in range(x-1, x+2):
                neigh_y = y
                if neigh_x < 0 or neigh_x >= size or x == neigh_x:
                    continue
                neigh_qubit = neigh_x * size + neigh_y
                topology[qubit].append(neigh_qubit)

            for neigh_y in range(y-1, y+2):
                neigh_x = x
                if neigh_y < 0 or neigh_y >= size or y == neigh_y:
                    continue
                neigh_qubit = neigh_x * size + neigh_y
                topology[qubit].append(neigh_qubit)

    for qubit, coupling in topology.items():
        coupling.sort()

    return topology


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


def get_grid_neighbor_info(size, max_distance):
    neigh_info = defaultdict(list)
    for x in range(size):
        for y in range(size):
            qubit = x * size + y
            for neigh_x in range(x-1, x+2):
                for neigh_y in range(y-1, y+2):
                    if neigh_x < 0 or neigh_x >= size or neigh_y < 0 or neigh_y >= size or (x == neigh_x and y == neigh_y):
                        continue
                    if math.sqrt((x-neigh_x)**2 + (x-neigh_y)**2) <= max_distance:
                        neigh_qubit = neigh_x * size + neigh_y
                        neigh_info[qubit].append(neigh_qubit)

    return neigh_info


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
    def __init__(self, n_qubits, topology = None, neighbor_info = None, basic_single_gates: list = None, basis_two_gates: list = None, 
                 divide: bool = True, decoupling: bool = True, single_qubit_gate_time = 30, two_qubit_gate_time = 60):
        self.n_qubits = n_qubits
        
        if topology is None:
            topology = {
                q1: [q1 for q2 in range(n_qubits) if q1 != q2]
                for q1 in range(n_qubits)
            }
        
        
        self.topology = topology 
        self.coupling_map = topology_to_coupling_map(topology)
        self.neighbor_info = neighbor_info # describe qubits that have mutual interactions


        if basic_single_gates is None:
            basic_single_gates = default_basis_single_gates
            
        if basis_two_gates is None:
            basis_two_gates = default_basis_two_gates
            
        self.basic_single_gates = basic_single_gates
        self. basis_two_gates = basis_two_gates
        
        self.divide = divide  # whether a layer only has single-qubit gates or two-qubit gates
        self.decoupling = decoupling # whether dynamic decoupling is applied
        
        self.single_qubit_gate_time = single_qubit_gate_time  # ns
        self.two_qubit_gate_time = two_qubit_gate_time # ns