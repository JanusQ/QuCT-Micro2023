from qiskit.dagcircuit.dagnode import DAGOpNode

from circuit.formatter import my_circuit_to_dag


def get_layered_instructions(circuit):
    '''
    这个layer可能不是最好的，应为这个还考虑了画图的时候不要交错
    '''
    dagcircuit, instructions, nodes = my_circuit_to_dag(circuit)
    # dagcircuit.draw(filename = 'dag.svg')
    graph_layers = dagcircuit.multigraph_layers()

    layer2operations = []  
    layer_opnames = []
    for layer in graph_layers:
        opnames = [node.op.name for node in layer if isinstance(node, DAGOpNode) and node.op.name != 'barrier']
        layer = [node for node in layer if isinstance(node, DAGOpNode) and node.op.name != 'barrier']
        if len(layer) != 0:
            layer2operations.append(layer)
        if len(opnames) != 0:
            layer_opnames.append(opnames)

    for _index, instruction in enumerate(instructions):
        assert instruction.operation.name != 'barrier'
        assert nodes[_index].op.name != 'barrier'

    layer2instructions = []
    instruction2layer = [None] * len(nodes)
    for layer, operations in enumerate(layer2operations):  # 层号，该层操作
        layer_instructions = []
        for node in operations:  # 该层的一个操作
            assert node.op.name != 'barrier'
            # print(node.op.name)
            index = nodes.index(node)
            layer_instructions.append(instructions[index])
            instruction2layer[index] = layer  # instruction在第几层
        layer2instructions.append(layer_instructions)

    return layer_opnames, layer2instructions, instruction2layer, instructions, dagcircuit, nodes

### pattern match
def is_identical(pattern1, pattern2):
    for layer1, layer2 in zip(pattern1, pattern2):
        if len(layer1) != len(layer2):
            return False
        for op1, op2 in zip(layer1, layer2):
            if op1 != op2:
                return False
    return True

### find a sub-pattern for a circuit, return the start and end index of the sub-pattern
def find_subpattern(layered_opname, min_len = 2, max_len = 6, pattern_repeat_times = 2):
    '''
    min_len: the min length of the pattern
    max_len: the max length of the pattern
    pattern_repeat_times: the min occurance of this patten appears in this circuit
    return: the start and end index of the sub-pattern
    '''
    re_start = -1
    re_end = -1
    for pattern_len in range(min_len, max_len + 1):
        for _index in range(0, len(layered_opname) - pattern_len + 1):
            pattern = layered_opname[_index : _index + pattern_len]
            repeat_time = 1
            for _index2 in range(_index + pattern_len, len(layered_opname) - pattern_len + 1):
                if is_identical(pattern, layered_opname[_index2 : _index2 + pattern_len]):
                    repeat_time += 1
                    if repeat_time >= pattern_repeat_times:
                        re_start = _index
                        re_end = _index + pattern_len
    return re_start, re_end

### add bug to the circuit
def add_bug(circuit):
    layered_opname, layer2instructions, instruction2layer, instructions, dagcircuit, nodes = get_layered_instructions(circuit)
    print(circuit)
    # bug_circuit = qiskit.QuantumCircuit()

