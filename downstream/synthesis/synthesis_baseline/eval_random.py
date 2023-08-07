from baseline.qsearch.unitaries import toffoli, fredkin, peres, logical_or, qft, grover_unitary, full_adder
from experiment_tool.experiment_runner import ExperimentRunner
from experiment_tool.utils import random_unitary


def exp_toffoli(n_qubits, synthesiser_list):
    assert n_qubits == 3
    unitary = toffoli
    runner = ExperimentRunner('toffoli', unitary, synthesiser_list,'random')
    runner.run()


def exp_fredkin(n_qubits, synthesiser_list):
    assert n_qubits == 3
    unitary = fredkin
    runner = ExperimentRunner('fredkin', unitary, synthesiser_list,'random')
    runner.run()


def exp_peres(n_qubits, synthesiser_list):
    assert n_qubits == 3
    unitary = peres
    runner = ExperimentRunner('peres', unitary, synthesiser_list,'random')
    runner.run()


def exp_logical_or(n_qubits, synthesiser_list):
    assert n_qubits == 3
    unitary = logical_or
    runner = ExperimentRunner('logical_or', unitary, synthesiser_list,'random')
    runner.run()


def exp_full_adder(n_qubits, synthesiser_list):
    assert n_qubits == 4
    unitary = full_adder
    runner = ExperimentRunner('full_adder', unitary, synthesiser_list ,'random')
    runner.run()


def exp_random(n_qubit, synthesiser_list, num):
    for i in range(num):
        unitary = random_unitary(n_qubit)
        runner = ExperimentRunner(f'random-{n_qubit}-{i}', unitary, synthesiser_list,'random')
        runner.run()


def exp_qft(n_qubits, synthesiser_list):
    unitary = qft(2 ** n_qubits)
    runner = ExperimentRunner(f'qft-{n_qubits}', unitary, synthesiser_list,'random')
    runner.run()


def exp_grover(n_qubits, synthesiser_list):
    unitary = grover_unitary(n_qubits)
    runner = ExperimentRunner(f'grover-{n_qubits}', unitary, synthesiser_list,'random')
    runner.run()


if __name__ == '__main__':
    synthesiser = ['CSDSynthesis', 'QSDSynthesis', 'QiskitSynthesis', 'QFastSynthesis'] # 'QSearchSynthesis'
    exp_random(3, synthesiser, 10)
