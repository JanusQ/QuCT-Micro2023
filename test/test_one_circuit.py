'''相梁做实验用的'''
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import matplotlib as mpl
import copy
import os
import pathlib
import sys
import scipy
import re
import time
from functools import reduce, partial
from imp import reload
from colorama import Fore
import math
from qiskit import QuantumRegister, QuantumCircuit, transpile  # ClassicalRegister,
from qiskit.tools.visualization import circuit_drawer
from qiskit.circuit import Gate, Instruction
import os
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
# import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

from qiskit.dagcircuit import DAGCircuit
import random

# 这里的代码都是基于circuit的结构不会变化的，看下有没有circuit的immutable的版本
RAW_5x3 = [ \
    ["X/2",        "-X/2",       "X/2",        "-X/2",       "X/2",        "VZ",         "X",          "VZ",         "X",          "X",          "-X/2",       "X/2",        "-X/2",       "X/2",        "I"           ],
    ["I",          "dcz1_8",     "dcz2_7",     "dcz3_6",     "dcz4_5",     "dcz4_5",     "dcz3_6",     "dcz2_7",     "dcz1_8",     "dcz9_10",    "dcz9_10",    "I",          "I",          "I",          "I"           ],
    ["I",          "X/2",        "I",          "X/2",        "I",          "I",          "X",          "I",          "X",          "X",          "X/2",        "I",          "I",          "I",          "I"           ],
    ["I",          "dcz1_2",     "dcz1_2",     "dcz3_4",     "dcz3_4",     "I",          "dcz6_13",    "I",          "dcz8_11",    "I",          "I",          "dcz8_11",    "I",          "dcz6_13",    "I"           ],
    ["I",          "I",          "X/2",        "I",          "X/2",        "I",          "VZ",         "I",          "VZ",         "I",          "dcz10_11",   "dcz10_11",   "I",          "I",          "I"           ],
    ["I",          "I",          "VZ jszzzz2", "I",          "VZ jszzzz4", "I",          "I",          "I",          "I",          "I",          "I",          "X/2",        "I",          "I",          "I"           ],
    ["I",          "I",          "X/2",        "I",          "X/2",        "I",          "I",          "I",          "I",          "I",          "I",          "VZ jszzzz1", "I",          "I",          "I"           ],
    ["I",          "dcz1_2",     "dcz1_2",     "dcz3_4",     "dcz3_4",     "I",          "I",          "I",          "I",          "I",          "I",          "X/2",        "I",          "I",          "I"           ],
    ["I",          "X/2",        "dcz2_7",     "X/2",        "dcz4_5",     "dcz4_5",     "I",          "dcz2_7",     "dcz8_11",    "I",          "I",          "dcz8_11",    "I",          "I",          "I"           ],
    ["I",          "dcz1_8",     "X",          "dcz3_6",     "-VZ/2",      "Y/2",        "dcz3_6",     "X",          "dcz1_8",     "I",          "dcz10_11",   "dcz10_11",   "I",          "I",          "I"           ],
    ["I",          "X/2",        "I",          "X/2",        "I",          "X",          "I",          "dcz7_12",    "Y/2",        "I",          "X/2",        "X",          "dcz7_12",    "I",          "I"           ],
    ["dcz0_1",     "dcz0_1",     "dcz2_3",     "dcz2_3",     "dcz4_5",     "dcz4_5",     "I",          "X",          "X",          "dcz9_10",    "dcz9_10",    "I",          "X/2",        "I",          "I"           ],
    ["Y -jszz1",   "I",          "Y -jszz2",   "I",          "Y -jsxx2",   "VZ",         "I",          "I",          "VZ",         "Y/2",        "X",          "I",          "dcz12_13",   "dcz12_13",   "I"           ],
    ["dcz0_1",     "dcz0_1",     "dcz2_3",     "dcz2_3",     "dcz4_5",     "dcz4_5",     "I",          "I",          "I",          "I",          "VZ/2",       "I",          "I",          "X/2",        "I"           ],
    ["VZ/2",       "Y/2",        "VZ/2",       "Y/2",        "VZ/2",       "I",          "I",          "I",          "I",          "I",          "X/2",        "I",          "I",          "VZ jszzzz3", "I"           ],
    ["dcz0_9",     "-X/2",       "I",          "-X/2",       "X/2",        "I",          "I",          "I",          "I",          "dcz0_9",     "I",          "I",          "I",          "X/2",        "I"           ],
    ["X/2",        "dcz1_8",     "I",          "I",          "I",          "I",          "dcz6_13",    "I",          "dcz1_8",     "X",          "I",          "I",          "I",          "dcz6_13",    "I"           ],
    ["dcz0_1",     "dcz0_1",     "I",          "I",          "I",          "I",          "Y/2",        "I",          "I",          "I",          "I",          "I",          "dcz12_13",   "dcz12_13",   "I"           ],
    ["I",          "X/2",        "I",          "I",          "I",          "I",          "X",          "I",          "I",          "I",          "I",          "I",          "X/2",        "X",          "I"           ],
    ["I",          "VZ jsxxxx1", "I",          "dcz3_6",     "I",          "I",          "dcz3_6",     "dcz7_12",    "I",          "I",          "I",          "I",          "dcz7_12",    "dcz13_14",   "dcz13_14"    ],
    ["I",          "X/2",        "I",          "I",          "I",          "I",          "VZ",         "Y/2",        "I",          "I",          "I",          "I",          "X/2",        "Y -jszz4",   "I"           ],
    ["dcz0_1",     "dcz0_1",     "dcz2_7",     "I",          "I",          "I",          "I",          "dcz2_7",     "I",          "I",          "I",          "dcz11_12",   "dcz11_12",   "dcz13_14",   "dcz13_14"    ],
    ["X/2",        "dcz1_8",     "X/2",        "I",          "I",          "I",          "I",          "X",          "dcz1_8",     "I",          "I",          "Y -jszz3",   "I",          "VZ/2",       "Y/2"         ],
    ["dcz0_9",     "X",          "dcz2_3",     "dcz2_3",     "I",          "I",          "I",          "VZ",         "X",          "dcz0_9",     "I",          "dcz11_12",   "dcz11_12",   "I",          "-X/2"        ],
    ["X",          "VZ/2",       "I",          "X/2",        "I",          "dcz5_14",    "I",          "I",          "dcz8_11",    "X/2",        "I",          "dcz8_11",    "Y/2",        "I",          "dcz5_14"     ],
    ["VZ/2",       "X/2",        "I",          "VZ jsxxxx3", "I",          "I",          "I",          "I",          "X",          "dcz9_10",    "dcz9_10",    "VZ/2",       "-X/2",       "I",          "I"           ],
    ["X/2",        "I",          "I",          "X/2",        "I",          "I",          "I",          "dcz7_12",    "I",          "Y -jsxx1",   "I",          "X/2",        "dcz7_12",    "I",          "I"           ],
    ["I",          "I",          "dcz2_3",     "dcz2_3",     "I",          "I",          "I",          "I",          "I",          "dcz9_10",    "dcz9_10",    "dcz11_12",   "dcz11_12",   "I",          "I"           ],
    ["I",          "I",          "X/2",        "dcz3_6",     "I",          "I",          "dcz3_6",     "I",          "I",          "VZ/2",       "Y/2",        "I",          "X/2",        "I",          "I"           ],
    ["I",          "I",          "dcz2_7",     "X",          "I",          "I",          "X",          "dcz2_7",     "I",          "X/2",        "X",          "I",          "VZ jsxxxx2", "I",          "I"           ],
    ["I",          "I",          "X",          "VZ/2",       "I",          "I",          "dcz6_13",    "I",          "I",          "I",          "I",          "I",          "X/2",        "dcz6_13",    "I"           ],
    ["I",          "I",          "VZ/2",       "X/2",        "I",          "I",          "X",          "dcz7_12",    "I",          "I",          "I",          "I",          "dcz7_12",    "X/2",        "I"           ],
    ["I",          "I",          "X/2",        "I",          "I",          "I",          "I",          "Y/2",        "I",          "I",          "I",          "dcz11_12",   "dcz11_12",   "dcz13_14",   "dcz13_14"    ],
    ["I",          "I",          "I",          "I",          "I",          "I",          "I",          "X",          "I",          "I",          "I",          "X/2",        "X",          "I",          "X/2"         ],
    ["I",          "I",          "I",          "I",          "I",          "I",          "I",          "I",          "dcz8_11",    "I",          "I",          "dcz8_11",    "VZ/2",       "I",          "VZ jsxxxx4"  ],
    ["I",          "I",          "I",          "I",          "I",          "I",          "I",          "I",          "Y/2",        "I",          "I",          "X",          "X/2",        "I",          "X/2"         ],
    ["I",          "I",          "I",          "I",          "I",          "dcz5_14",    "I",          "I",          "X",          "I",          "I",          "VZ/2",       "I",          "I",          "dcz5_14"     ],
    ["I",          "I",          "I",          "I",          "I",          "Y/2",        "I",          "I",          "I",          "I",          "I",          "X/2",        "I",          "dcz13_14",   "dcz13_14"    ],
    ["I",          "I",          "I",          "I",          "I",          "X",          "I",          "I",          "I",          "I",          "I",          "I",          "I",          "X/2",        "X"           ],
    ["I",          "I",          "I",          "I",          "I",          "I",          "dcz6_13",    "I",          "I",          "I",          "I",          "I",          "I",          "dcz6_13",    "VZ/2"        ],
    ["I",          "I",          "I",          "I",          "I",          "I",          "Y/2",        "I",          "I",          "I",          "I",          "I",          "I",          "X",          "X/2"         ],
    ["I",          "I",          "I",          "I",          "I",          "I",          "X",          "I",          "I",          "I",          "I",          "I",          "I",          "VZ/2",       "I"           ],
    ["I",          "I",          "I",          "I",          "I",          "I",          "I",          "I",          "I",          "I",          "I",          "I",          "I",          "X/2",        "I"           ],
       ]

max_qubit_num = 14
basis_gates = ['rx', 'ry', 'rz', 'cx']
basis_single_gates = ['rx', 'ry', 'rz']
basis_two_gates = ['cx']
opss = RAW_5x3
output='mpl'
scale=0.9
t_xy=40
t_cz=50

qs = QuantumRegister(len(opss[0]), 'q')
circuit = QuantumCircuit(qs)
t_all = 0
d_all = 0
for ops in opss:
    cz_gate_checked = []
    t_step = 0
    for i, gate in enumerate(ops):
        if len(ops) == 1:
            custom_gate = Gate(gate, len(qs), [])
            circuit.append(custom_gate, qs)
            continue
        if 'dcz' not in gate:
            if '(' not in gate:
                custom_gate = Gate(gate, 1, [])
            else:
                gate, param = eval(gate)
                print(gate, param)
                custom_gate = Gate(gate, 1, [param * np.pi])
            t_step = max(t_step, t_xy) if gate != 'VZ' else max(t_step, 0)
            circuit.append(custom_gate, [i])
        else:
            if gate not in cz_gate_checked:
                qi, qj = gate[3:].split('_')
                circuit.cz(int(qi), int(qj))
                cz_gate_checked.append(gate)
                t_step = max(t_step, t_cz)
    t_all += t_step
    if gate != 'VZ':
        d_all = d_all + 1
        circuit.barrier()

def my_circuit_to_dag(circuit):
    instructions = []
    dagnodes = []

    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name
    dagcircuit.global_phase = circuit.global_phase
    dagcircuit.calibrations = circuit.calibrations
    dagcircuit.metadata = circuit.metadata

    dagcircuit.add_qubits(circuit.qubits)
    dagcircuit.add_clbits(circuit.clbits)

    for register in circuit.qregs:
        dagcircuit.add_qreg(register)

    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    for instruction in circuit:
        if instruction.operation.name == 'barrier':
            continue
        dag_node = dagcircuit.apply_operation_back(
            instruction.operation.copy(), instruction.qubits, instruction.clbits
        )
        instructions.append(instruction)  # TODO: 这个里面会不会有barrier
        dagnodes.append(dag_node)
        assert instruction.operation.name == dag_node.op.name

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    return dagcircuit, instructions, dagnodes

# 之后还可以单独写一个获得每个比特前面的和后面的门的代码
def get_layered_instructions(circuit):
    '''这个layer可能不是最好的，应为这个还考虑了画图的时候不要交错'''
    dagcircuit, instructions, nodes = my_circuit_to_dag(circuit)
    graph_layers = dagcircuit.multigraph_layers()

    layer2operations = []# Remove input and output nodes
    for layer in graph_layers:
        layer = [node for node in layer if isinstance(node, DAGOpNode) and node.op.name != 'barrier']
        if len(layer) != 0:
            layer2operations.append(layer)
    
    for _index, instruction in enumerate(instructions):
        assert instruction.operation.name != 'barrier'
        assert nodes[_index].op.name != 'barrier'

    layer2instructions = []
    instruction2layer = [None] * len(nodes)
    for layer, operations in enumerate(layer2operations):
        layer_instructions = []
        for node in operations:
            assert node.op.name != 'barrier'
            # print(node.op.name)
            index = nodes.index(node)
            layer_instructions.append(instructions[index])
            instruction2layer[index] = layer
        layer2instructions.append(layer_instructions)

    return layer2instructions, instruction2layer

# circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=0) # coupling_map=coupling_map, initial_layout=initial_layout, 
print(circuit)
layer2instructions, instruction2layer = get_layered_instructions(circuit)

def parse_instruction(instruction):
    qubits = [qubit.index for qubit in instruction.qubits]
    op_name = instruction.operation.name
    return f'{op_name},{qubits}'

class Step(): 
    '''A step contains (source gate, edge_type, target gate)'''
    def __init__(self, source, edge, target):
        parse_instruction(source)
        self.source = parse_instruction(source) # it should not be changed in the future
        self.edge = edge
        self.target = parse_instruction(target)

        self._hash_id = str(self)
        return

    def __hash__(self): return hash(self._hash_id)

    # def __str__(self): return str((self.source, self.edge, self.target))
    def __str__(self): return str((self.edge, self.target))

# TODO: there should be a Step factory to save memory

class Path():
    '''A path consists of a list of step'''
    def __init__(self, steps):
        self.steps = steps
        self._hash_id = str(self)
    
    def add(self, step):
        steps = list(self.steps)
        steps.append(step)
        return Path(steps)
    
    def __hash__(self): return hash(self._hash_id)

    def __str__(self): return ' '.join([str(step) for step in self.steps])

def randomwalk(graph: QuantumCircuit, head_instruction: Gate, max_step):
    # self-loop

    now_instruction = head_instruction
    traveled_nodes = [head_instruction]

    now_path = Path([Step(head_instruction, 'loop', head_instruction)])  # 初始化一个指向自己的
    paths = [now_path]
    for step_index in range(max_step):
        former_node = now_instruction
        now_node_index = instructions.index(now_instruction)
        now_layer = instruction2layer[now_node_index]
        parallel_gates = layer2instructions[now_layer]
        former_gates = [] if now_layer == 0 else layer2instructions[now_layer - 1]  #TODO: 暂时只管空间尺度的

        candidates = parallel_gates + former_gates
        candidates = [candidate for candidate in candidates if candidate not in traveled_nodes]
        if len(candidates) == 0:
            break

        now_instruction = random.choice(candidates)
        traveled_nodes.append(now_instruction)

        now_path = now_path.add(Step(former_node, 'parallel' if now_instruction in parallel_gates else 'dependency', now_instruction))
        paths.append(now_path)

    return paths

instructions = [instrcution for instrcution in circuit.data if instrcution.operation.name != 'barrier']
sparse_vecs = [None] * len(instructions)

path_per_instruction = 20
max_step = 2
path_count = defaultdict(int)
gate_paths = []
for head_instruction in instructions:
    traveled_paths = set()
    for _ in range(path_per_instruction):
        paths = randomwalk(circuit, head_instruction, max_step)
        for path in paths:
            # print(path)
            if path in traveled_paths:
                continue
            traveled_paths.add(path)
            path_count[path] += 1
    gate_paths.append(traveled_paths)

path_count = defaultdict(int)
for _paths in gate_paths:
    for _path in _paths:
        path_count[str(_path)] += 1

path_index = [
    _path
    for _path, _count in path_count.items()
    if _count > 6 # a threshold
]
# print(path_index)

instruction2vec = [None] * len(instructions)
# defaultdict(lambda : )
for _index, _paths in enumerate(gate_paths):
    instruction = instructions[_index]
    vec = np.zeros((len(path_index,)))
    for _path in _paths:
        if str(_path) not in path_index: continue
        _path_index = path_index.index(str(_path))
        vec[_path_index] = 1
    assert instruction2vec[_index] is None
    instruction2vec[_index] = vec
    # print(vec)

# print('')
# circuit.draw()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=20).fit(instruction2vec)
labels = kmeans.labels_
print(labels)


draw_circuit = QuantumCircuit(max_qubit_num)
for instruction in circuit:
    op_name = instruction.operation.name
    if op_name != 'barrier':
        instruction.operation.label = str(labels[instructions.index(instruction)])
        # custom_gate = Gate(op_name, len(qs), [])
        # draw_circuit.add_instruction(instruction)


import seaborn as sns
sns.color_palette("hls", 8)


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple([int(elm*255) for elm in rgb])

colors = [rgb_to_hex(rgb) for rgb in sns.color_palette("hls", 20)]

fig = circuit.draw(output='mpl',
                scale=0.9,
                vertical_compression='high',
                fold=100,
                idle_wires=False,
                style={
                    "linecolor": "#1e1e1e",
                    "gatefacecolor": "#25a4e8",
                    "displaycolor": {
                        # "_Y_": ["#539f18", "#FFFFFF"],
                        # "_I_": ["#d7d7d7", "#FFFFFF"],
                        # "I": ["#d9d2e9", "#FFFFFF"],
                        # "VZ": ["#cfd0f3", "#FFFFFF"]
                        str(_label): [colors[_label], "#FFFFFF"]
                        for _label in range(20)
                    }
                })
# ax = fig.get_axes()[0]
fig.tight_layout()

# from qiskit.visualization.qcstyle
# circuit.draw('mpl')
plt.show()