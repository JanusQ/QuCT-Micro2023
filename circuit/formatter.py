from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.dagcircuit import DAGCircuit, DAGOpNode


def get_layered_instructions(circuit):
    '''
    这个layer可能不是最好的，应为这个还考虑了画图的时候不要交错
    '''
    dagcircuit, instructions, nodes = my_circuit_to_dag(circuit)
    # dagcircuit.draw(filename = 'dag.svg')
    graph_layers = dagcircuit.multigraph_layers()

    layer2operations = []  # Remove input and output nodes
    for layer in graph_layers:
        layer = [node for node in layer if isinstance(node, DAGOpNode) and node.op.name != 'barrier']
        if len(layer) != 0:
            layer2operations.append(layer)

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

    return layer2instructions, instruction2layer, instructions, dagcircuit, nodes

def my_format_circuit_to_qiskit(qubit_num, layer2instructions):
    circuit = QuantumCircuit(qubit_num)

    for layer, layer_instructions in enumerate(layer2instructions):
        for instruction in layer_instructions:
            name = instruction['name']
            qubits = instruction['qubits']
            params = instruction['params']
        if name in ('rx', 'ry', 'rz'):
            assert len(params) == 1 and len(qubits) == 1
            circuit.__getattribute__(name)(params[0], qubits[0])
        elif name in ('cz', 'cx'):
            assert len(params) == 0 and len(qubits) == 2
            circuit.__getattribute__(name)(qubits[0], qubits[1])
        elif name in ('h'):
            circuit.__getattribute__(name)(qubits[0])
        else:
            circuit.__getattribute__(name)(*(params + qubits))

        circuit.barrier()

    return circuit

def qiskit_to_my_format_instruction(instruction):
    return {
        'name': instruction.operation.name,
        'qubits': [qubit.index for qubit in instruction.qubits],
        'params': instruction.operation.params,
    }


def qiskit_to_my_format_circuit(layer2qiskit_instructions):
    instructions = []
    instruction2layer = []
    layer2instructions = []
    for layer, layer_instructions in enumerate(layer2qiskit_instructions):
        layer_instructions = [qiskit_to_my_format_instruction(instruction) for instruction in layer_instructions]
        instructions += layer_instructions
        layer2instructions.append(layer_instructions)

        for layer_instructions in layer_instructions:
            instruction2layer.append(layer)

    for index, instruction in enumerate(instructions):
        instruction['id'] = index

    return layer2instructions, instruction2layer, instructions

def layer2circuit(layer2instructions, n_qubits):
    new_circuit = QuantumCircuit(n_qubits)
    for layer, instructions in enumerate(layer2instructions):
        involved_qubits = []
        for instruction in instructions:
            involved_qubits += [qubit.index for qubit in instruction.qubits]
            new_circuit.append(instruction)
        new_circuit.barrier()
    return new_circuit

def my_circuit_to_dag(circuit: QuantumCircuit):
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

    # for instruction, qargs, cargs in circuit.data:
    # for instruction in circuit:
    for instruction in circuit.data:
        operation = instruction.operation


        dag_node = dagcircuit.apply_operation_back(
            operation, instruction.qubits, instruction.clbits
        )
        if operation.name == 'barrier':
            continue
        instructions.append(instruction)  # TODO: 这个里面会不会有barrier
        dagnodes.append(dag_node)
        operation._index = len(dagnodes) - 1
        assert instruction.operation.name == dag_node.op.name

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    return dagcircuit, instructions, dagnodes


def assign_barrier(qiskit_circuit):
    layer2instructions, instruction2layer, instructions, dagcircuit, nodes = get_layered_instructions(
        qiskit_circuit)  # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的

    new_circuit = QuantumCircuit(qiskit_circuit.num_qubits)
    for layer, instructions in enumerate(layer2instructions):
        for instruction in instructions:
            new_circuit.append(instruction)
        new_circuit.barrier()

    return new_circuit

def instruction2str(instruction):
    if isinstance(instruction, CircuitInstruction):
        qubits = [qubit.index for qubit in instruction.qubits]
        op_name = instruction.operation.name
    else:
        qubits = instruction['qubits']
        op_name = instruction['name']
    return f'{op_name}-{"-".join([str(_) for _ in qubits])}'

def layered_circuits_to_executable_code(layer2instructions):
    req_data = [{
        "seq": convert_circuit(layer2instructions),
        "stats": 1000,
    }]
    return req_data

def convert_circuit(circuit_info):
    # qc = circuit_info['qiskit_circuit']
    # match_hardware_constraints(qc)
    # print(circuit_info['id'])
    return [convert_layer(level) for level in circuit_info['layer2instructions']]

BASIC_GATE_SET = ['h', 'rx', 'ry', 'rz', 'cz']

GATE_NAME_MAP = {
    'h': 'H',
    'rx': 'X',
    'ry': 'Y',
    'rz': 'Z',
    'cz': 'CZ'
}
QUBIT_NAME_OFFLINE = ['q3_15',
                      'q3_17',
                      'q3_19',
                      'q5_19',
                      'q7_19',
                      'q7_17',
                      'q9_17',
                      'q11_17',
                      'q13_17',
                      'q13_15']

def convert_layer(level):
    # Filter warnings for back-reference index of a qubit
    import warnings

    QUBIT_NAME = QUBIT_NAME_OFFLINE

    level_result = {
        'TwoQ': [],
        'SingleQ': [],
        'TwoQType': [],
        'SingleQType': [],
        'TwoAngle': [],
        'SingleAngle': []
    }

    for instruction in level:
        if instruction.operation.num_qubits == 2:
            q0_id, q1_id = instruction.qubits[0].index, instruction.qubits[1].index
            level_result['TwoQ'].append((QUBIT_NAME[q0_id], QUBIT_NAME[q1_id]))
            level_result['TwoQType'].append(
                GATE_NAME_MAP[instruction.operation.name])
            level_result['TwoAngle'].append(0)
        else:
            q0_id = instruction.qubits[0].index
            if instruction.operation.name == 'h':
                angle = 0.0
                assert len(instruction.operation.params) == 0
            else:
                angle = instruction.operation.params[0]
                assert len(instruction.operation.params) == 1

            if not isinstance(angle, float):
                angle = float(angle)

            level_result['SingleQ'].append(QUBIT_NAME[q0_id])
            level_result['SingleQType'].append(
                GATE_NAME_MAP[instruction.operation.name])
            level_result['SingleAngle'].append(angle)

    return level_result
