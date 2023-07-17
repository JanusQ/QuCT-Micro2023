from qiskit import QuantumCircuit, execute
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode

max_qubit_num = 5
circ = QuantumCircuit(max_qubit_num)
circ.h(0)
for q in range(1, max_qubit_num):
    circ.cx(0, q)
for q in range(1, max_qubit_num):
    circ.cx(0, q)
print(circ)

from qiskit.dagcircuit import DAGCircuit
def _circuit_to_dag(circuit):
    """Build a ``DAGCircuit`` object from a ``QuantumCircuit``.

    Args:
        circuit (QuantumCircuit): the input circuit.

    Return:
        DAGCircuit: the DAG representing the input circuit.

    Example:
        .. jupyter-execute::

            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.dagcircuit import DAGCircuit
            from qiskit.converters import circuit_to_dag
            from qiskit.visualization import dag_drawer
            %matplotlib inline

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            circ = QuantumCircuit(q, c)
            circ.h(q[0])
            circ.cx(q[0], q[1])
            circ.measure(q[0], c[0])
            circ.rz(0.5, q[1]).c_if(c, 2)
            dag = circuit_to_dag(circ)
            dag_drawer(dag)
    """
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

    for instruction in circuit.data:
        dag_node = dagcircuit.apply_operation_back(
            instruction.operation, instruction.qubits, instruction.clbits  # 删除了原先的copy，所以就可以通过这个operation来找gate了
        )
        instruction.operation.circuit_instruction = instruction
        instruction.operation.dag_instruction = dag_node

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    return dagcircuit


dag_circ = _circuit_to_dag(circ)
# 遍历输出是按照插入的顺序来的，所以大概是对的吧
for gate in circ:
    print(gate)
    same_num = 0
    for node in dag_circ.nodes():
        if not isinstance(node, DAGOpNode):
            continue
        if id(node.op) == id(gate.operation): 
            print(node, gate)
            same_num += 1
    assert same_num == 1
