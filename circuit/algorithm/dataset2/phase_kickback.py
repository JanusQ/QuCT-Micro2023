## Programming Quantum Computers
##   by Eric Johnston, Nic Harrigan and Mercedes Gimeno-Segovia
##   O'Reilly Media
##
## More samples like this can be found at http://oreilly-qc.github.io

import math

from qiskit import QuantumCircuit, QuantumRegister


def get_cir():
    reg1 = QuantumRegister(2, name='reg1')
    reg2 = QuantumRegister(1, name='reg2')
    qc = QuantumCircuit(reg1, reg2)

    qc.h(reg1)         # put a into reg1 superposition of 0,1,2,3
    qc.cu1(math.pi/4, reg1[0], reg2)
    qc.cu1(math.pi/2, reg1[1], reg2)
    return qc
