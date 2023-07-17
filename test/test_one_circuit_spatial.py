'''相梁做实验用的，先输出一个空间尺度上的关系'''
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
from qiskit.circuit import Gate
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
    nodes = []

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
        nodes.append(dag_node)
        assert instruction.operation.name == dag_node.op.name

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    return dagcircuit, instructions, nodes

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
    if len(qubits) == 1:
        op_name = 'single'
    else:
        op_name = instruction.operation.name
    return f'{op_name},{qubits}'

import itertools
def get_subset(mylist):
    sub_set = []
    n =len(mylist)
    for num in range(n):
        for elm in itertools.combinations(mylist, num+1):
            # if len(elm) >= 2:
            elm = list(elm)
            elm.sort()
            sub_set.append(tuple(elm))
    return sub_set

path_count = defaultdict(int)
instruction_set2layer = defaultdict(list)
for layer, layer_instructions in enumerate(layer2instructions):
    layer_instructions = [_instruction for _instruction in layer_instructions if 'I' not in _instruction.operation.name]
    layer_instruction_labels = [parse_instruction(_instruction) for _instruction in layer_instructions]
    subset_instructions = get_subset(layer_instruction_labels)
    for elm in subset_instructions:
        path_count[elm] += 1
        instruction_set2layer[elm].append(layer)

path_index = [
    (_path, _count)
    for _path, _count in path_count.items()
    # if _count > 6 # a threshold
]

path_index.sort(key=lambda x: x[1], reverse=True)

# print(path_index)
file = open('parallel_gates.txt', 'w')
for path in path_index:
    file.write(f'{path[0]},\t\t{path[1]}\n')
    # print(path[0], path[1])
file.close()

def showLayer(elm, circuit):
    for layer in instruction_set2layer[elm]:
        for instruction in layer2instructions[layer]:
            if parse_instruction(instruction) in elm:
                instruction.operation.label = "mark"

    fig = circuit.draw(output='mpl',
                    scale=0.9,
                    vertical_compression='high',
                    fold=100,
                    idle_wires=False,
                    style={
                        "linecolor": "#1e1e1e",
                        "gatefacecolor": "#25a4e8",
                        "displaycolor": {
                            "mark": "#FF0000",
                        }
                    })
    fig.tight_layout()
    plt.show()
    return

showLayer(('single,[1]', 'single,[8]'), circuit)
