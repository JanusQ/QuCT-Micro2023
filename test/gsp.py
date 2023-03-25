import pygsti

from pygsti.modelpacks.legacy import std1Q_XYI

from pygsti.extras.rb.simulate import rb_with_pauli_errors,create_iid_pauli_error_model,circuit_simulator_for_tensored_independent_pauli_errors

from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.algorithms.randomcircuit import create_clifford_rb_circuit
import numpy as np


pspec = QPS(4, ['Gx','Gy','Gz'],geometry='line',qubit_labels='line')

print(pspec.qubit_labels)

p=0.8
err = np.array([[1.,0.,0.,0.],[1.,0.,0.,0.]])
error_model = {'0':err,'1':err}
# error_model = create_iid_pauli_error_model(pspec, 0.01, 0.01, 0.01)

lengths = [1, 10, 100, 1000]

# res = rb_with_pauli_errors(pspec, error_model,lengths,100,10)

# circuit = create_clifford_rb_circuit(pspec)

circuit = pygsti.circuits.Circuit( ('Gx','Gy','Gx') ,'line')
circuit.number_of_lines=1
print(circuit.line_labels)
print(pspec.qubit_labels)

print(circuit)

outcome = circuit_simulator_for_tensored_independent_pauli_errors(
                circuit, pspec, error_model,1000)

# mycircuit = pygsti.circuits.Circuit( ('Gx','Gy','Gx') )
# model = std1Q_XYI.target_model()
# # model.probabilities
# outcome_probabilities = model.probabilities(mycircuit)
# print(outcome_probabilities)