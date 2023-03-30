## Programming Quantum Computers
##   by Eric Johnston, Nic Harrigan and Mercedes Gimeno-Segovia
##   O'Reilly Media
##
## More samples like this can be found at http://oreilly-qc.github.io
##
## A complete notebook of all Chapter 4 samples (including this one) can be found at
##  https://github.com/oreilly-qc/oreilly-qc.github.io/tree/master/samples/Qiskit

import math

from qiskit import QuantumCircuit, QuantumRegister


## Uncomment the next line to see diagrams when running in a notebook
#%matplotlib inline

## Example 4-1: Basic Teleportation

# Set up the program
def get_cir():
    alice = QuantumRegister(1, name='alice')
    ep    = QuantumRegister(1, name='ep')
    bob   = QuantumRegister(1, name='bob')
    qc = QuantumCircuit(alice, ep, bob)

    # entangle
    qc.h(ep)
    qc.cx(ep, bob)
    qc.barrier()

    # prep payload
    qc.reset(alice)
    qc.h(alice)
    qc.rz(math.radians(45), alice)
    qc.h(alice)
    qc.barrier()

    # send
    qc.cx(alice, ep)
    qc.h(alice)
    qc.barrier()

    # receive

    # verify
    qc.h(bob)
    qc.rz(math.radians(-45), bob)
    qc.h(bob)
    return qc

