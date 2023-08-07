import copy
import itertools as it
import time
import random

from utils.backend_info import *


def gen_washington_topology(n_qubit):
    topology = {0: [1, 14], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5, 15], 5: [4, 6], 6: [5, 7], 7: [6, 8], 8: [7, 9, 16], 9: [10],
                10: [9, 11], 11: [10, 12], 12: [11, 13, 17], 13: [12], 14: [0, 18], 15: [4, 22], 16: [8, 26], 17: [12, 30], 18: [14, 19], 19: [18, 20],
                20: [19, 21, 33], 21: [20, 22], 22: [15, 21, 23], 23: [22, 24], 24: [23, 25, 34], 25: [24, 26], 26: [16, 25, 27], 27: [26, 28], 28: [27, 29, 35], 29: [28, 30],
                30: [17, 29, 31], 31: [30, 32], 32: [31, 36], 33: [20, 39], 34: [24, 43], 35: [28, 47], 36: [32, 51], 37: [38, 52], 38: [37, 39], 39: [33, 38, 40],
                40: [39, 41], 41: [40, 42, 53], 42: [41, 43], 43: [34, 42, 44], 44: [43, 45], 45: [44, 46, 54], 46: [45, 47], 47: [35, 46, 48], 48: [47, 49], 49: [48, 50, 55],
                50: [49, 51], 51: [36, 50], 52: [37, 56], 53: [41, 60], 54: [45, 64], 55: [49, 68], 56: [52, 57], 57: [56, 58], 58: [57, 59, 71], 59: [58, 60],
                60: [53, 59, 61], 61: [60, 62], 62: [61, 63, 72], 63: [62, 64], 64: [54, 63, 65], 65: [64, 66], 66: [65, 67, 73], 67: [66, 68], 68: [55, 67, 69], 69: [68, 70],
                70: [69, 74], 71: [58, 77], 72: [62, 81], 73: [66, 85], 74: [70, 89], 75: [76, 90], 76: [75, 77], 77: [71, 76, 78], 78: [77, 79], 79: [78, 80, 91],
                80: [79, 81], 81: [72, 80, 82], 82: [81, 83], 83: [82, 84, 92], 84: [83, 85], 85: [73, 84, 86], 86: [85, 87], 87: [86, 88, 93], 88: [87, 89], 89: [74, 88],
                90: [75, 94], 91: [79, 98], 92: [83, 102], 93: [87, 106], 94: [90, 95], 95: [94, 96], 96: [95, 97, 109], 97: [96, 98], 98: [91, 97, 99], 99: [98, 100],
                100: [99, 101, 110], 101: [100, 102], 102: [92, 101, 103], 103: [102, 104], 104: [103, 105, 111], 105: [104, 106], 106: [93, 105, 107], 107: [106, 108], 108: [107, 112], 109: [96, 114],
                110: [100, 118], 111: [104, 122], 112: [108, 126], 113: [114], 114: [109, 113, 115], 115: [114, 116], 116: [115, 117], 117: [116, 118], 118: [110, 117, 119], 119: [118, 120],
                120: [119, 121], 121: [120, 122], 122: [111, 121, 123], 123: [122, 124], 124: [123, 125], 125: [124, 126], 126: [112, 125]}
    assert n_qubit <= 127

    new_topology = defaultdict(list)
    for qubit in topology.keys():
        if qubit < n_qubit:
            for ele in topology[qubit]:
                if ele < n_qubit:
                    new_topology[qubit].append(ele)
    return new_topology


def get_washington_neighbor_info(topology, max_distance):

    neigh_info = copy.deepcopy(topology)
    if max_distance == 1:
        return neigh_info
    for qubit in topology.keys():
        fommer_step = topology[qubit]
        t = max_distance - 1
        while t > 0:
            new_fommer_step = []
            for fommer_qubit in fommer_step:
                neigh_info[qubit] += topology[fommer_qubit]
                new_fommer_step += topology[fommer_qubit]
            fommer_step = new_fommer_step
            t -= 1
        neigh_info[qubit] = list(set(neigh_info[qubit]))
        neigh_info[qubit].remove(qubit)
    return neigh_info


def gen_sycamore_topology(size):
    '''
    Example:
    0   1   2 
      3   4   5
    6   7   8


     0   1   2   3
       4   5   6   7
     8   9   10  11
       12  13  14  15
    '''
    topology = defaultdict(list)

    for x in range(size):
        for y in range(size):
            qubit = x * size + y
            if x == 0:
                up = []
            else:
                up = [(x-1) * size + i for i in range(size)]
            if x == size - 1:
                down = []
            else:
                down = [(x+1) * size + i for i in range(size)]

            up_down = up + down
            if x % 2 == 1:
                candidates = [qubit - size, qubit -
                              size + 1, qubit + size, qubit + size+1]
            else:
                candidates = [qubit - size, qubit -
                              size - 1, qubit + size, qubit + size-1]

            for candidate in candidates:
                if candidate in up_down:
                    topology[qubit].append(candidate)

    for qubit, coupling in topology.items():
        coupling.sort()

    return topology


def get_sycamore_neighbor_info(topology, max_distance):

    neigh_info = copy.deepcopy(topology)
    if max_distance == 1:
        return neigh_info
    for qubit in topology.keys():
        fommer_step = topology[qubit]
        t = max_distance - 1
        while t > 0:
            new_fommer_step = []
            for fommer_qubit in fommer_step:
                neigh_info[qubit] += topology[fommer_qubit]
                new_fommer_step += topology[fommer_qubit]
            fommer_step = new_fommer_step
            t -= 1
        neigh_info[qubit] = list(set(neigh_info[qubit]))
        neigh_info[qubit].remove(qubit)
    return neigh_info


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



def gen_fulllyconnected_topology(n_qubits):
    return {
        q1: [q2  for q2 in range(n_qubits) if q1 != q2]
        for q1 in range(n_qubits)
    }


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
        # tuple(coupling)
        for coupling in coupling_map
    ]


def get_devide_qubit(topology, max_qubit):
    qubits = topology.keys()
    trevel_node, devide_qubits = [], []

    random.seed(time.time())
    while len(trevel_node) != len(qubits):
        sub_qubits = []
        head = random.choice(list(qubits-trevel_node))
        
        fommer_step = topology[head]
        trevel_node.append(head)
        sub_qubits.append(head)
        t = 1
        while t < max_qubit:
            new_fommer_step = []
            for fommer_qubit in fommer_step:
                if t == max_qubit:
                    break
                if fommer_qubit in trevel_node:
                    continue
                sub_qubits.append(fommer_qubit)
                trevel_node.append(fommer_qubit)
                new_fommer_step+= topology[fommer_qubit]
                t += 1
            if len(new_fommer_step) == 0:
                break
            fommer_step = list(set(new_fommer_step))
            if head in fommer_step:
                fommer_step.remove(head)
            
        sub_qubits.sort()
        devide_qubits.append(sub_qubits)

    return devide_qubits


def devide_chip(backend, max_qubit, devide_qubits = None):
    ret_backend = copy.deepcopy(backend)

    # n_qubits = backend.n_qubits
    # devide_qubits = [i for i in range(n_qubits)]
    # devide_qubits = devide_qubits[offset:] + devide_qubits[:offset]
    # devide_qubits = [devide_qubits[i:i+max_qubit] for i in range(0,n_qubits,max_qubit)]
    if not devide_qubits:
        devide_qubits = get_devide_qubit(ret_backend.topology,  max_qubit)
    ret_backend.devide_qubits = devide_qubits
    coupling_map = copy.deepcopy(ret_backend.coupling_map)
    for e1, e2 in coupling_map:
        for i in range(len(devide_qubits)):
            if (e1 in devide_qubits[i] and e2 not in devide_qubits[i]) or (e1 not in devide_qubits[i] and e2 in devide_qubits[i]):
                ret_backend.coupling_map.remove([e1, e2])
                break
    return ret_backend


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

    def __init__(self, n_qubits, topology=None, neighbor_info=None, coupling_map = None, basis_single_gates: list = None,
                 basis_two_gates: list = None,
                 divide: bool = False, decoupling: bool = False, single_qubit_gate_time=30, two_qubit_gate_time=60):
        self.n_qubits = n_qubits

        if topology is None:
            topology = {
                q1: [q2 for q2 in range(n_qubits) if q1 != q2]
                for q1 in range(n_qubits)
            }

        self.topology = topology
        if coupling_map is None:
            self.coupling_map = topology_to_coupling_map(topology)
        else:
            self.coupling_map = coupling_map
        # self._coupling_map = [tuple(elm) for elm in coupling_map]
            
        self._true_coupling_map = list(self.coupling_map)
        # describe qubits that have mutual interactions

        if neighbor_info is None:
            self.neighbor_info = copy.deepcopy(topology)
        else:
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
            1 - 1 / 2000  #1 - random.random() / 10000
            for q in range(n_qubits)
        ]

        self.two_qubit_fidelity = [
            1 - 1 / 1000  #1 - random.random() / 5000
            for i, coupler in enumerate(self.coupling_map)
        ]

        self.qubit2T1 = [
            110 - 5 #random.random() * 10
            for q in range(n_qubits)
        ]

        self.qubit2T2 = [
            110 - 3 #random.random() * 3
            for q in range(n_qubits)
        ]

        self.rb_error = None # rb测得的error
        self.cache = {}
        self.devide_qubits = None

        self.routing = 'sabre'
        self.optimzation_level = 3
        
    def get_subgraph(self, location):
        """Returns the sub_coupling_graph with qubits in location."""
        subgraph = []
        for q0, q1 in self.coupling_map:
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


if __name__ == "__main__":
    topology = gen_grid_topology(4)
    neigh_info = get_grid_neighbor_info(4, 1)
from utils.backend import Backend

