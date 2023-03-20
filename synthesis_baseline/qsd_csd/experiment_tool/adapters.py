import numpy as np
import sys
from typing import Type
from qiskit import QuantumCircuit, transpile
from experiment_tool.utils import ISynthesiser

# CSD
from unitary import unitary as decompose_unitary
# QSearch
import qsearch
from qsearch.assemblers import ASSEMBLER_IBMOPENQASM
# QFast
from qfast.synthesis import synthesize as qfast_synthesis
# Qiskit
from qiskit.extensions import UnitaryGate


_MOD = sys.modules[__name__]


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
        transpiled = transpile(decomposed, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)
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
        transpiled = transpile(decomposed, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)
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
        qasm, cpu_time = qfast_synthesis(unitary)
        return QuantumCircuit.from_qasm_str(qasm), cpu_time


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
        assert 2**num_qubits == dim

        original_qc = QuantumCircuit(num_qubits)
        unitary_gate = UnitaryGate(unitary)
        original_qc.append(unitary_gate, list(range(unitary_gate.num_qubits)))
        return transpile(original_qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3), 0
