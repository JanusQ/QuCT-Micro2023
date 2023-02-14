import random

from qiskit import QuantumCircuit


def initialize_qaoa(V, E):
    qc = QuantumCircuit(len(V))

    qc.h(range(len(V)))
    qc.barrier()
    return qc


def apply_cost_hamiltonian(qc, V, E, gamma):
    for k, l, weight in E:
        qc.cp(-2*gamma*weight, k, l)
        qc.p(gamma*weight, k)
        qc.p(gamma*weight, l)
    qc.barrier()
    return qc


def apply_mixing_hamiltonian(qc, V, E, beta):
    qc.rx(2*beta, range(len(V)))
    qc.barrier()
    return qc


def construct_full_qaoa(p, gammas, betas, V, E):
    qc = initialize_qaoa(V, E)
    for i in range(p):
        qc = apply_cost_hamiltonian(qc, V, E, gammas[i])
        qc = apply_mixing_hamiltonian(qc, V, E, betas[i])
    # qc = terminate_qaoa(qc, V, E)
    return qc


def get_cir(n):
    E = []
    for _ in range(random.randint(5, n * (n - 1) // 2)):
        sample = random.sample(range(0, n), 2)
        E.append((sample[0], sample[1], random.random()))
    return construct_full_qaoa(1, [.4], [.8], range(n), E)


if __name__ == '__main__':
    print(get_cir(100))

