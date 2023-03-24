import sys
# sys.path.extend(['/home/olcus/synthesis/quantum-circuit-synthesis'])
from experiment_tool.experiment_runner import ExperimentRunner
from experiment_tool.utils import random_unitary
from qsearch.unitaries import grover_unitary, qft, full_adder


def exp_qft4():
    synthesiser_list = ['QFastSynthesis', 'QiskitSynthesis', 'CSDSynthesis', 'QSDSynthesis']
    unitary = qft(2 ** 4)
    runner = ExperimentRunner('qft-4', unitary, synthesiser_list)
    runner.run()


def exp_full_adder():
    synthesiser_list = ['QFastSynthesis', 'QiskitSynthesis', 'CSDSynthesis', 'QSDSynthesis']
    unitary = full_adder
    runner = ExperimentRunner('full_adder', unitary, synthesiser_list)
    runner.run()


def exp_random4(num):
    synthesiser_list = ['QFastSynthesis', 'QiskitSynthesis', 'CSDSynthesis', 'QSDSynthesis']
    for i in range(num):
        unitary = random_unitary(4)
        runner = ExperimentRunner(f'random-4-{i}', unitary, synthesiser_list)
        runner.run()

def exp_grover4():
    synthesiser_list = ['QFastSynthesis', 'QiskitSynthesis', 'CSDSynthesis', 'QSDSynthesis']
    unitary = grover_unitary(4)
    runner = ExperimentRunner('grover-4', unitary, synthesiser_list)
    runner.run()


if __name__ == '__main__':
    exp_random4(1)
    # exp_qft4()
    # exp_grover4()
