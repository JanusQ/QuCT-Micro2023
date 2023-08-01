import sys
from typing import Type

import numpy as np
# QSearch
from baseline import qsearch
# CSD
from baseline.qclib2.unitary import unitary as decompose_unitary
# QFast
from baseline.qfast.synthesis import synthesize as qfast_synthesis
from baseline.qsearch.assemblers import ASSEMBLER_IBMOPENQASM
from common_function import log2
from experiment_tool.utils import ISynthesiser
from qiskit import QuantumCircuit, transpile
# Qiskit
from qiskit.extensions import UnitaryGate

_MOD = sys.modules[__name__]

optimization_level = 1


def get_synthesise_adapter(cls_name: str) -> Type[ISynthesiser]:
    cls = getattr(_MOD, cls_name)
    assert issubclass(cls, ISynthesiser)
    return cls


class CSDSynthesis(ISynthesiser):
    """
    Experiment wrapper for CSD decomposition
    """

    def __init__(self, unitary: np.ndarray):
        ISynthesiser.__init__(self, unitary)

    def get_synthesiser_name(self) -> str:
        return "CSD Synthesiser"

    def synthesis(self, unitary: np.ndarray):
        decomposed = decompose_unitary(unitary, decomposition='csd')
        transpiled = transpile(decomposed, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=optimization_level)
        return transpiled, 0


class QSDSynthesis(ISynthesiser):
    """
    Experiment wrapper for CSD decomposition
    """

    def __init__(self, unitary: np.ndarray):
        ISynthesiser.__init__(self, unitary)

    def get_synthesiser_name(self) -> str:
        return "QSD Synthesiser"

    def synthesis(self, unitary: np.ndarray):
        decomposed = decompose_unitary(unitary, decomposition='qsd')
        transpiled = transpile(decomposed, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=optimization_level)
        return transpiled, 0


class QSearchSynthesis(ISynthesiser):
    """
    Experiment wrapper for QSearch
    """

    def __init__(self, unitary: np.ndarray):
        ISynthesiser.__init__(self, unitary)

    def get_synthesiser_name(self) -> str:
        return "QSearch Synthesiser"

    def synthesis(self, unitary: np.ndarray):
        opt = qsearch.Options(target=unitary, verbosity=0)
        compiler = qsearch.SearchCompiler(options=opt)
        result = compiler.compile()
        qasm = ASSEMBLER_IBMOPENQASM.assemble(result, opt)
        return QuantumCircuit.from_qasm_str(qasm), result['cpu_time']


class QFastSynthesis(ISynthesiser):
    """
    Experiment wrapper for QFast
    """

    def __init__(self, unitary: np.ndarray):
        ISynthesiser.__init__(self, unitary)

    def get_synthesiser_name(self) -> str:
        return "QFast Synthesiser"

    def synthesis(self, unitary: np.ndarray):
        qubit_num = log2(len(unitary))
        qasm, cpu_time = qfast_synthesis(unitary)  # coupling_graph=[(i,i+1) for i  in range(qubit_num -1 )]
        return transpile(QuantumCircuit.from_qasm_str(qasm), basis_gates=['u1', 'u2', 'u3', 'cx'],
                         optimization_level=optimization_level), cpu_time


class QiskitSynthesis(ISynthesiser):
    """
    Experiment wrapper for Qiskit
    """

    def __init__(self, unitary: np.ndarray):
        ISynthesiser.__init__(self, unitary)

    def get_synthesiser_name(self) -> str:
        return "Qiskit Synthesiser"

    def synthesis(self, unitary: np.ndarray):
        dim, _ = unitary.shape
        num_qubits = int(np.log2(dim))
        assert 2 ** num_qubits == dim

        original_qc = QuantumCircuit(num_qubits)
        unitary_gate = UnitaryGate(unitary)
        original_qc.append(unitary_gate, list(range(unitary_gate.num_qubits)))
        return transpile(original_qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=optimization_level), 0
