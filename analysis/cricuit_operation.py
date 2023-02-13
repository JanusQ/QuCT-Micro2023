from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumCircuit
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
import math
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit import QuantumCircuit, QuantumRegister


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


# TODO: 需要按照层架barrier，为了方便真机执行
def assign_barrier(qiskit_circuit):
    layer2instructions, instruction2layer, instructions, dagcircuit, nodes = get_layered_instructions(
        qiskit_circuit)  # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的

    new_circuit = QuantumCircuit(qiskit_circuit.num_qubits)
    for layer, instructions in enumerate(layer2instructions):
        for instruction in instructions:
            new_circuit.append(instruction)
        new_circuit.barrier()

    return new_circuit


def layer2circuit(layer2instructions, n_qubits):
    new_circuit = QuantumCircuit(n_qubits)
    for layer, instructions in enumerate(layer2instructions):
        involved_qubits = []
        for instruction in instructions:
            involved_qubits += [qubit.index for qubit in instruction.qubits]
            new_circuit.append(instruction)
        new_circuit.barrier()
    return new_circuit


def get_x_instruction(n_qubits, index):
    """
    生成一个x门的instruction，输入电路的比特数和，x门插在哪个比特上
    """
    x_circuit = QuantumCircuit(1)
    x_circuit.rx(math.pi, 0)
    x_operation = x_circuit.data[0].operation
    register = QuantumRegister(size=n_qubits, name='q')
    return CircuitInstruction(x_operation,
                              [Qubit(register=register, index=index)]
                              )


def get_layer_type(layer2instructions, n_qubits):
    num_layer = len(layer2instructions)
    layer_type = [[0 for _ in range(num_layer)] for _ in range(n_qubits)]
    for index, layer in enumerate(layer2instructions):
        for ins in layer:
            for qubit in ins.qubits:
                layer_type[qubit.index][index] = 1
    return layer_type


def _get_ideal_section_in_one_qubit_dynamic_coupling(start, one_layer_type):
    num_layer = len(one_layer_type)
    end = start
    for i in range(start, num_layer):
        if one_layer_type[i] == 0:
            start = i
            break
    fin_loop = 0
    for i in range(start, num_layer):
        if one_layer_type[i] == 1:
            end = i
            break
        fin_loop = i
    if fin_loop == num_layer - 1:
        end = num_layer
    return start, end


def _dynamic_decoupling_in_one_ideal_section(start, end, qubit, t_t, t_dd, one_layer_type, layer2instructions):
    n_qubits = layer2instructions[0][0].qubits[0].register.size
    if end - start < 2:
        return layer2instructions
    while (end - start) > 1:
        start_gate_id = start
        jump = t_dd // 2 // t_t
        if (t_dd / 2) % t_t > (t_dd / 4):
            jump += 1
        end_gate_id = start_gate_id + jump
        if end_gate_id >= end:
            end_gate_id -= 1
        layer2instructions[start_gate_id].append(
            get_x_instruction(n_qubits, qubit)
        )
        layer2instructions[end_gate_id].append(
            get_x_instruction(n_qubits, qubit)
        )
        start = end_gate_id + 1
    return layer2instructions


def dynamic_decoupling(layer2instructions, t_t, t_dd):
    n_qubits = layer2instructions[0][0].qubits[0].register.size
    num_layer = len(layer2instructions)
    layer_type = get_layer_type(layer2instructions, n_qubits)

    for qubit in range(n_qubits):
        start = 0
        while start < num_layer:
            start, end = _get_ideal_section_in_one_qubit_dynamic_coupling_divide(start, layer_type[qubit])
            layer2instructions = _dynamic_decoupling_in_one_ideal_section(
                start, end, qubit, t_t, t_dd, layer_type[qubit], layer2instructions
            )
            start = end + 1
    return layer2instructions


def get_layer_type_divide(layer2instructions):
    """
    返回layer2instructions的每层的类型，是单比特门层则为1；否则则为2
    """
    return [len(layer[0].qubits) for layer in layer2instructions]


def get_layer2qubits_divide(layer2instructions, n_qubits):
    """
    返回一个列表，列表中的每个元素是 每个比特对应的每层门类型表，0为单比特层无门，1为单比特门，2为双比特门，3为双比特层无门
    """
    layer2qubits = []
    num_layer = len(layer2instructions)
    layer_type = get_layer_type_divide(layer2instructions)
    for qubit in range(n_qubits):
        layer2qubits.append([0 for _ in range(num_layer)])
    for index, layer in enumerate(layer2instructions):
        if layer_type[index] == 1:
            for ins in layer:
                layer2qubits[ins.qubits[0].index][index] = 1
        else:
            for ins in layer:
                for qubit in ins.qubits:
                    layer2qubits[qubit.index][index] = 2
                for qubit in range(n_qubits):
                    if layer2qubits[qubit][index] == 0:
                        layer2qubits[qubit][index] = 3
    return layer2qubits


def _get_ideal_section_in_one_qubit_dynamic_coupling_divide(start, one_layer2qubits):
    num_layer = len(one_layer2qubits)
    end = start
    for i in range(start, num_layer):
        if one_layer2qubits[i] == 0 or one_layer2qubits[i] == 3:
            start = i
            break
    fin_loop = 0
    for i in range(start, num_layer):
        if one_layer2qubits[i] == 2 or one_layer2qubits[i] == 1:
            end = i
            break
        fin_loop = i
    if fin_loop == num_layer - 1:
        end = num_layer
    return start, end


def _dynamic_decoupling_in_one_ideal_section_divide(start, end, qubit, t_s, t_t, t_dd, one_layer2qubits, layer2instructions):
    n_qubits = layer2instructions[0][0].qubits[0].register.size
    list0 = []
    for i in range(start, end):
        if one_layer2qubits[i] == 0:
            list0.append(i)
    if len(list0) < 2:
        return layer2instructions

    while len(list0) > 1:
        start_gate_id = list0.pop(0)
        listd = []
        for i in list0:
            t = 0
            for j in range(start_gate_id, i):
                if one_layer2qubits[j] == 0:
                    t += t_s
                else:
                    t += t_t
            listd.append({
                'id': i,
                't': abs(t - t_dd / 2)
            })
        min_t = listd[0]['t']
        end_gate_id = listd[0]['id']
        for d in listd:
            if min_t > d['t']:
                min_t = d['t']
                end_gate_id = d['id']
        layer2instructions[start_gate_id].append(
            get_x_instruction(n_qubits, qubit)
        )
        layer2instructions[end_gate_id].append(
            get_x_instruction(n_qubits, qubit)
        )
        list0 = [i for i in list0 if i > end_gate_id]
    return layer2instructions


def dynamic_decoupling_divide(layer2instructions, t_s, t_t, t_dd):
    n_qubits = layer2instructions[0][0].qubits[0].register.size
    num_layer = len(layer2instructions)
    layer2qubits = get_layer2qubits_divide(layer2instructions, n_qubits)
    for qubit in range(n_qubits):
        start = 0
        while start < num_layer:
            # 分段找完整的03段
            start, end = _get_ideal_section_in_one_qubit_dynamic_coupling_divide(start, layer2qubits[qubit])
            # 在这一段进行门保护
            layer2instructions = _dynamic_decoupling_in_one_ideal_section_divide(
                start, end, qubit, t_s, t_t, t_dd, layer2qubits[qubit], layer2instructions
            )
            start = end + 1
    return layer2instructions


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

if __name__ == '__main__':
    pass
