import pyzx as zx
from pyzx.circuit.qasmparser import QASMParser

from qiskit import QuantumCircuit, assemble
from numpy import pi

import matplotlib.pyplot as plt

# qubit_amount = 5
# gate_count = 80
# #Generate random circuit of Clifford gates
# circuit = zx.generate.cliffordT(qubit_amount, gate_count)
# #If running in Jupyter, draw the circuit
# fig = zx.draw(zx.extract_circuit(circuit))
# fig.savefig('temp_data/before_opt.png')
# #Use one of the built-in rewriting strategies to simplify the circuit
# zx.simplify.full_reduce(circuit)
# circuit = zx.extract_circuit(circuit)
# #See the result
# fig = zx.draw(circuit)
# fig.savefig('temp_data/after_opt.png')


# qiskit_circuit.cx(1, 2) 


# print(qiskit_circuit)

# print(qasm_string)

# p = QASMParser()
# zx_circuit = p.parse(qasm_string)
# zx_circuit.name = 'none'


def zx_optimize(qiskit_circuit: QuantumCircuit):
    qasm_string = qiskit_circuit.qasm()
    zx_circuit = zx.Circuit.from_qasm(qasm_string)

    # print(zx_circuit)

    # fig = zx.draw(zx_circuit)
    # fig.savefig('temp_data/before_opt.png')

    # circuit = zx.generate.cliffordT(5, 80)
    zx_graph = zx_circuit.to_graph()
    zx.simplify.full_reduce(zx_graph)
    new_zx_circuit = zx.extract_circuit(zx_graph)

    # fig = zx.draw(new_zx_circuit)
    # fig.savefig('temp_data/after_opt.png')

    new_qasm_string = new_zx_circuit.to_qasm()

    new_qiskit_circuit = QuantumCircuit.from_qasm_str(new_qasm_string)
    # print(new_qiskit_circuit)
    return new_qiskit_circuit

qiskit_circuit = QuantumCircuit(3)
qiskit_circuit.rx(pi/7, 1)
qiskit_circuit.rx(pi/7, 1)

print(qiskit_circuit)
print(zx_optimize(qiskit_circuit))