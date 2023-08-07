import math
import traceback
# import collections as co
from collections import deque

import csd.utilities_gen as utg
import numpy as np
from csd.BitVector import BitVector
from csd.CktEmbedder import CktEmbedder
from csd.Controls import Controls
from csd.HadamardTransform import HadamardTransform
from csd.Node import Node
from csd.OneQubitGate import OneQubitGate
from csd.UnitaryMat import UnitaryMat
from qiskit import QuantumCircuit
from treelib import Tree


class CSDTree:
    """
    This class creates a binary tree of nodes whose cargo is contained in
    the attributes of class Node. This class, being a child of class
    SEO_writer, is also capable of writing English & Picture files. After
    creating a binary tree, it proceeds to use that tree to produce a CS
    decomposition of the unitary matrix init_unitary_mat that is fed into
    its constructor. This CS (cosine-sine) decomp consists of a sequence of
    diagonal unitaries (DIAG lines in English file) and multiplexors (MP_Y
    lines in English file) whose product equals init_unitary_mat.

    If you wish to expand DIAG and MP_Y lines into cnots and single qubit
    rotations, use DiagUnitaryExpander and MultiplexorExpander classes.

    The CS decomposition was a famous decomp of Linear Algebra well before
    quantum computing. It was first applied to quantum computing in the 1999
    paper and accompanying C++ program cited below. Much of the code of the
    original C++ Qubiter has been rewritten in Python for the new pythonic
    Qubiter.

    Let init_unitary_mat be N dimensional, with N = 2^n, where n = number of
    qubits. A general N dimensional unitary matrix has N^2 dofs (real
    degrees of freedom). That's because it has N^2 complex entries, so 2*N^2
    real parameters, but those parameters are subject to N real constraints
    and N(N-1)/2 complex constraints, for a total of N^2 real constraints.
    So 2N^2 real parameters minus N^2 real constraints gives N^2 dofs.

    (a) Each DIAG (MP_Y, resp.) line of the CS decomp of init_unitary_mat
    depends on N (N/2, resp.) angles and there are about N DIAG and N MP_Y
    lines. So the DIAG lines alone have enough dofs, N^2 of them, to cover
    all N^2 dofs of init_unitary_mat. So clearly, there is a lot of
    redundancy in the CS decomp used by Qubiter. But, there is hope: the CS
    decomp is not unique, and it might be possible to choose a CS decomp
    that makes zero many of the angles in the DIAG and MP_Y lines. Some of
    those "compiler optimizations" are considered in references below.

    (b) The CS decomp as used here leads to order N^2 = 2^{2n} cnots and
    qubit rotations so it is impractical for large N. But for small N,
    it can be useful. For large N, it might be possible to discover
    approximations to individual MP_Y and DIAG lines. An approximation of
    this type is considered in MultiplexorExpander.

    Clearly, there is much room for future research to improve (a) and (b).

    References
    ----------
    1. R.R. Tucci, A Rudimentary Quantum Compiler(2cnd Ed.)
    https://arxiv.org/abs/quant-ph/9902062

    2. Qubiter 1.11, a C++ program whose first version was released together
    with Ref.1 above. Qubiter 1.11 is included in the
    quantum_CSD_compiler/LEGACY folder of this newer, pythonic version of
    Qubiter.

    3. R.R. Tucci, Quantum Fast Fourier Transform Viewed as a Special Case
    of Recursive Application of Cosine-Sine Decomposition,
    https://arxiv.org/abs/quant-ph/0411097

    Attributes
    ----------
    global_phase_rads : float
        If arr is the initial unitary matrix fed to the constructor,
        then this equals delta, where arr = exp(i*delta) arr1, where arr1 is
        a special unitary matrix (det(arr1) = 1)
    root_nd : Node
        The root or starting node of the tree. The only node without parents.
        Each node remembers its children, so you only need the root_nd to
        access all other nodes.

    """

    def __init__(self, emb: CktEmbedder, init_unitary_mat, verbose=False):
        """
        Constructor

        Parameters
        ----------
        emb : CktEmbedder
        init_unitary_mat : np.ndarray
            This is the matrix that is fed to cs_decomp() in root node
            constructor.
        verbose : bool

        Returns
        -------


        """
        self.emb = emb

        self.verbose = verbose
        assert UnitaryMat.is_unitary(init_unitary_mat)
        self.global_phase_rads = UnitaryMat.global_phase_rads(init_unitary_mat)

        self.all_nodes = []
        self.ph_fac = np.exp(1j * self.global_phase_rads)
        self.root_nd = self.build_tree(init_unitary_mat / self.ph_fac)

    # ===========================================
    #              Build CSD Tree
    # ===========================================

    def build_tree(self, init_unitary_mat):
        """
        This function is called by the constructor to build a tree of
        Node's. It returns the root node of the tree.

        Parameters
        ----------
        init_unitary_mat : np.ndarray

        Returns
        -------
        Node

        """

        nd_ctr = 0

        num_qbits = self.emb.num_qbits_bef
        num_rows = (1 << num_qbits)  # pow2以后可以学下这么写
        assert init_unitary_mat.shape == (num_rows, num_rows)
        root_nd = Node(nd_ctr, None, None,
                       init_unitary_mat=init_unitary_mat)  # nd_id,  pa_nd, side, init_unitary_mat=None

        if self.verbose:
            print('building tree------------')
            print(root_nd)
        node_q = deque([root_nd])

        # level = level of tree splitting = len(node_q)
        # level = 1 for root node
        # level = num_of_bits+1 for node whose
        # central_mat is list of 1 dim arrays
        level = 1

        self.all_nodes.append(root_nd)

        # 这人是不是树搜索写魔怔了
        while level != 0:
            # since level!=0, cur_nd is not None here
            cur_nd = node_q[0]  # siwei: 这是啥
            if level == num_qbits + 1 or cur_nd.is_barren():
                node_q.popleft()
                level -= 1
            else:
                if cur_nd.left_nd is None:
                    nd_ctr += 1
                    next_nd = Node(nd_ctr, cur_nd, 'left')  # node, parent, side
                    self.all_nodes.append(next_nd)

                    if self.verbose:
                        print(cur_nd, '-left->', next_nd)
                    node_q.appendleft(next_nd)
                    level += 1
                elif cur_nd.right_nd is None:
                    nd_ctr += 1
                    next_nd = Node(nd_ctr, cur_nd, 'right')
                    self.all_nodes.append(next_nd)

                    if self.verbose:
                        print(cur_nd, '-right->', next_nd)
                    node_q.appendleft(next_nd)
                    level += 1
                else:
                    node_q.popleft()
                    level -= 1

        # N个比特，会有N+1层，2^{N+1}-1个节点

        return root_nd

    def vis_tree(self):
        tree = Tree()

        root_node = self.all_nodes[0]
        # print(root_node)
        nd_id = str(root_node)
        tree.create_node(nd_id, nd_id, data=root_node)

        for node in self.all_nodes[1:]:
            # print(node)
            nd_id = str(node)
            tree.create_node(nd_id, nd_id, parent=str(node.pa_nd), data=node)

        tree.show()
        return tree

    def get_csd_results(self):
        """
        仿照write写的

        This function writes English & Picture files. It visits all the
        Node's of the tree from right to left (this way: <--). It calls
        self.write_node() for each node.

        """
        results = []
        node_q = deque()
        nd = self.root_nd
        if self.verbose:
            print("writing tree------------")
            print(nd)
        while True:
            if nd is not None:
                node_q.appendleft(nd)
                if self.verbose:
                    if nd.right_nd is not None:
                        print(nd, '-right->', nd.right_nd)
                    else:
                        print(nd, '-right->', 'None')
                nd = nd.right_nd
            else:
                # Extract first of the node_q and assign it to nd.
                # Exit while() loop if node_q is empty.
                try:
                    nd = node_q.popleft()

                    node_data = self.parse_node(nd)
                    if node_data is not None:
                        results.append(node_data)
                    if self.verbose:
                        if nd.left_nd is not None:
                            print(nd, '-left->', nd.left_nd)
                        else:
                            print(nd, '-left->', 'None')
                    nd = nd.left_nd
                except IndexError:
                    # Exit while() loop if node_q is empty.
                    break
                except:
                    traceback.print_exc()
                    break

        return results

    def parse_node(self, nd):
        """
        This function is called by self.write() for each node of the tree.
        For a node with level <= num_qbits, the function writes an MP_Y line,
        whereas if level = num_qbits + 1, it writes a DIAG line.
        """
        if nd.is_barren():
            return

        num_qbits = self.emb.num_qbits_bef

        assert 1 <= nd.level <= num_qbits + 1
        # tar_bit_pos = num_qbits - 1 for level=1
        # tar_bit_pos = 0 for level=num_qbits
        # tar_bit_pos = -1 for level=num_qbits+1
        tar_bit_pos = num_qbits - nd.level

        trols = Controls(num_qbits)
        if tar_bit_pos >= 0:
            trols.bit_pos_to_kind = {c: c for c in range(tar_bit_pos)}
            for c in range(tar_bit_pos, num_qbits - 1):
                trols.bit_pos_to_kind[c + 1] = c
        else:
            trols.bit_pos_to_kind = {c: c for c in range(num_qbits)}
        trols.refresh_lists()

        rad_angles = []
        # central_mats is list of numpy arrays
        for dmat in nd.central_mats:
            rad_angles += list(dmat.flatten())

        # permute arr bit indices
        if 0 <= tar_bit_pos <= num_qbits - 3:
            # turn rad_angles into equivalent bit indexed tensor
            arr = np.array(rad_angles).reshape([2] * (num_qbits - 1))
            perm = list(range(tar_bit_pos)) + \
                   list(range(tar_bit_pos + 1, num_qbits - 1)) + [tar_bit_pos]
            if self.verbose:
                print("permutation", perm)
            arr.transpose(perm)
            # flatten arr and turn it into a list
            rad_angles = list(arr.flatten())

        if self.verbose:
            print("target bit", tar_bit_pos)
            print("controls", trols.bit_pos_to_kind)
            print("rad_angles", rad_angles)
        if tar_bit_pos >= 0:
            return 'multiplexor', tar_bit_pos, trols, rad_angles, nd
        else:
            return 'diag', trols, rad_angles, nd

    # ===========================================
    #              Swap Control
    # ===========================================

    def _emb_tar_bit_pos_and_trols(self, emb: CktEmbedder, tar_bit_pos, trols):
        aft_tar_bit_pos = emb.aft(tar_bit_pos)

        aft_trols = trols.new_embedded_self(emb)
        # add extra controls if there are any
        extra_dict = emb.extra_controls.bit_pos_to_kind
        if extra_dict:
            aft_trols.bit_pos_to_kind.update(extra_dict)
            aft_trols.refresh_lists()

        return aft_tar_bit_pos, aft_trols

    # ===========================================
    #              Parse DIAG
    # ===========================================

    def _emb_for_du(self, controls):
        T_bpos = []
        F_bpos = []
        MP_bpos = []
        for bpos, kind in controls.bit_pos_to_kind.items():
            # bool is subclass of int
            # so isinstance(x, int) will be true if x is bool!
            if isinstance(kind, bool):
                if kind:
                    T_bpos.append(bpos)
                else:
                    F_bpos.append(bpos)
            else:
                MP_bpos.append(bpos)
        T_bpos.sort()
        F_bpos.sort()
        MP_bpos.sort()

        bit_map = T_bpos + F_bpos + MP_bpos

        num_qbits = self.emb.num_qbits_bef
        assert len(bit_map) == len(set(bit_map)), \
            "bits used to define d-unitary are not unique"
        assert len(bit_map) <= num_qbits

        nt = len(T_bpos)
        nf = len(F_bpos)
        emb = CktEmbedder(num_qbits, num_qbits, bit_map)
        return emb, nt, nf

    def parse_diag(self, controls, rad_angles):
        assert utg.all_floats(rad_angles)

        diag_seo = []

        emb, nt, nf = self._emb_for_du(controls)
        num_qbits = emb.num_qbits_bef
        ntf = nt + nf
        num_MP_trols = num_qbits - ntf
        rads_arr = np.array(rad_angles)

        if np.linalg.norm(rads_arr) < 1e-6:
            print("unit d-unitary")
            return diag_seo

        conj_rads = HadamardTransform.ht(num_MP_trols, rads_arr)
        num_factors = (1 << num_MP_trols)
        f, lazy = BitVector.lazy_advance(0, 0)  # start at f=1
        cur_rot_bpos = 0
        prev_rot_bpos = 0
        cur_bvec = BitVector(num_MP_trols + 1, 1)  # start at 1
        prev_bvec = BitVector(num_MP_trols + 1, 0)
        diff_bvec = BitVector(num_MP_trols + 1, 0)

        TF_dict = dict(enumerate([True] * nt + [False] * nf))
        trols1 = Controls(num_qbits)
        trols1.bit_pos_to_kind = TF_dict.copy()
        trols1.refresh_lists()
        trols2 = Controls(num_qbits)

        def write_cnots(diff_bvec1, init_prev_T_bit):
            prev_T_bit = init_prev_T_bit
            while True:
                cur_T_bit = diff_bvec1.find_T_bit_to_left_of(prev_T_bit)
                if cur_T_bit == -1:
                    break
                trols2.bit_pos_to_kind = TF_dict.copy()
                trols2.bit_pos_to_kind[cur_T_bit + ntf] = True
                trols2.refresh_lists()

                aft_tar_bit_pos, aft_trols = self._emb_tar_bit_pos_and_trols(emb, ntf + init_prev_T_bit, trols2)

                # CX, control, target (Only one elm in bit_pos)
                elm = 'SIGX', aft_trols.bit_pos[0], aft_tar_bit_pos
                diag_seo.append(elm)
                # self.write_controlled_one_qbit_gate(
                #     ntf + init_prev_T_bit, trols2, OneQubitGate.sigx)

                prev_T_bit = cur_T_bit

        norma = np.power(np.sqrt(2), num_MP_trols)
        # for first A factor, f = 0, just global phase
        # write conditioned global phase
        global_ph = conj_rads[0] * norma / len(conj_rads)
        if abs(global_ph) > 1e-6:
            print('write global phase', global_ph, ntf)
            # GP, theta, qubit
            aft_tar_bit_pos, _ = self._emb_tar_bit_pos_and_trols(emb, ntf, trols1)
            elm = 'GLOBAL_PHAS', global_ph, aft_tar_bit_pos
            diag_seo.append(elm)
            # self.write_controlled_one_qbit_gate(ntf, trols1,
            #                                     OneQubitGate.phase_fac, [global_ph])

        while f < num_factors:
            cur_bvec.dec_rep = lazy
            # Since we have excluded f=0, f always has at least one T bit.
            cur_rot_bpos = cur_bvec.find_rightmost_T_bit()
            rads = conj_rads[cur_bvec.dec_rep] / norma
            if abs(rads) < 1e-6:
                pass
            else:
                # If cur_rot_bpos equals (doesn't equal) prev_rot_bpos,
                # then there is (isn't) cancellation between:
                # (1)the c-nots sigma_x(cur_rot_bpos)^n()
                # contributed by the right part of the current A factor
                # and
                # (2)the c-nots sigma_x(prev_rot_bpos)^n()
                # contributed by the left part of the previous A factor.

                if cur_rot_bpos == prev_rot_bpos:
                    diff_bvec = BitVector.new_with_T_on_diff(
                        cur_bvec, prev_bvec)
                    write_cnots(diff_bvec, cur_rot_bpos)
                else:
                    write_cnots(prev_bvec, prev_rot_bpos)
                    write_cnots(cur_bvec, cur_rot_bpos)
                    diff_bvec = BitVector.copy(cur_bvec)

                # RZ, theta, qubit
                aft_tar_bit_pos, _ = self._emb_tar_bit_pos_and_trols(emb, ntf + cur_rot_bpos, trols1)
                elm = 'ROTZ', rads, aft_tar_bit_pos
                diag_seo.append(elm)
                # self.write_controlled_one_qbit_gate(
                #     ntf + cur_rot_bpos, trols1, OneQubitGate.rot_ax, [rads, 3])

                prev_bvec = BitVector.copy(cur_bvec)
                prev_rot_bpos = cur_rot_bpos

            f, lazy = BitVector.lazy_advance(f, lazy)

        # Don't forget the leftmost c-nots
        write_cnots(prev_bvec, prev_rot_bpos)

        return diag_seo

    # ===========================================
    #              Parse MP_Y
    # ===========================================

    def _emb_for_plexor(self, tar_bit_pos, controls):
        """
        This is an internal function used inside the function use_MP_Y().
        The function returns emb, nt, nf to be used as arguments of a
        MultiplexorSEO_writer that will be used to expand the MP_y line
        currently being considered. emb is a circuit embedder, nt is the
        number of T bits and nf is the number of F bits detected in the
        input argument 'controls'.

        Parameters
        ----------
        tar_bit_pos : int
            target bit position of multiplexor currently being considered.
        controls : Controls
            controls of the MP_Y currently being considered.

        Returns
        -------
        CktEmbedder, int, int

        """
        T_bpos = []
        F_bpos = []
        MP_bpos = []
        for bpos, kind in controls.bit_pos_to_kind.items():
            # bool is subclass of int
            # so isinstance(x, int) will be true if x is bool!
            if isinstance(kind, bool):
                if kind:
                    T_bpos.append(bpos)
                else:
                    F_bpos.append(bpos)
            else:
                MP_bpos.append(bpos)
        T_bpos.sort()
        F_bpos.sort()
        MP_bpos.sort()

        bit_map = T_bpos + F_bpos + MP_bpos + [tar_bit_pos]

        num_qbits = self.emb.num_qbits_bef

        assert len(bit_map) == len(set(bit_map)), \
            "bits used to define multiplexor are not unique"
        assert len(bit_map) <= num_qbits

        nt = len(T_bpos)
        nf = len(F_bpos)
        emb = CktEmbedder(num_qbits, num_qbits, bit_map)
        return emb, nt, nf

    def parse_mp_y(self, tar_bit_pos, controls, rad_angles):
        assert utg.all_floats(rad_angles)

        mp_y_seo = []

        emb, nt, nf = self._emb_for_plexor(tar_bit_pos, controls)
        num_qbits = emb.num_qbits_bef
        ntf = nt + nf
        num_MP_trols = num_qbits - ntf - 1
        rads_arr = np.array(rad_angles)

        if np.linalg.norm(rads_arr) < 1e-6:
            print("unit multiplexor")
            return mp_y_seo

        conj_rads = HadamardTransform.ht(num_MP_trols, rads_arr)
        num_factors = (1 << num_MP_trols)

        cur_bvec = BitVector(num_MP_trols + 1, 0)  # start at zero
        prev_bvec = BitVector(num_MP_trols + 1, 0)

        TF_dict = dict(enumerate([True] * nt + [False] * nf))
        trols1 = Controls(num_qbits)
        trols1.bit_pos_to_kind = TF_dict.copy()
        trols1.refresh_lists()
        trols2 = Controls(num_qbits)

        def write_cnots(diff_bvec1):
            prev_T_bit = num_MP_trols
            while True:
                cur_T_bit = diff_bvec1.find_T_bit_to_right_of(prev_T_bit)
                if cur_T_bit == -1:
                    break
                trols2.bit_pos_to_kind = TF_dict.copy()
                trols2.bit_pos_to_kind[cur_T_bit + ntf] = True
                trols2.refresh_lists()

                aft_tar_bit_pos, aft_trols = self._emb_tar_bit_pos_and_trols(emb, ntf + num_MP_trols, trols2)

                # CX, control, target (Only one elm in bit_pos)
                elm = 'SIGX', aft_trols.bit_pos[0], aft_tar_bit_pos
                mp_y_seo.append(elm)
                # self.write_controlled_one_qbit_gate(
                #     ntf + num_MP_trols, trols2, OneQubitGate.sigx)

                prev_T_bit = cur_T_bit

        norma = np.power(np.sqrt(2), num_MP_trols)
        f = 0
        lazy = 0
        while f < num_factors:
            rads = conj_rads[cur_bvec.dec_rep] / norma
            if abs(rads) < 1e-6:
                pass
            else:
                diff_bvec = BitVector.new_with_T_on_diff(cur_bvec, prev_bvec)
                write_cnots(diff_bvec)

                # RY, theta, qubit
                aft_tar_bit_pos, aft_trols = self._emb_tar_bit_pos_and_trols(emb, ntf + num_MP_trols, trols1)
                elm = 'ROTY', rads, aft_tar_bit_pos
                mp_y_seo.append(elm)
                # self.write_controlled_one_qbit_gate(
                #     ntf + num_MP_trols, trols1, OneQubitGate.rot_ax, [rads, 2])

                prev_bvec = BitVector.copy(cur_bvec)
            f, lazy = BitVector.lazy_advance(f, lazy)
            cur_bvec.dec_rep = lazy

        # Don't forget the leftmost c-nots:
        diff_bvec = prev_bvec
        write_cnots(diff_bvec)

        return mp_y_seo

    # ===========================================
    #            Merge DIAG and MP_Y
    # ===========================================

    def swap_csd_results(self):
        csd_results = self.csd_results
        new_results = []
        for item in csd_results:
            item_type = item[0]
            # print(*item)
            if item_type == 'diag':  # 对应根节点
                _, trols, rad_angles, nd = item
                diag_seo = self.parse_diag(trols, rad_angles)
                new_results += diag_seo
                # new_results.append(item)
            else:  # 'multiplexor'  # 对应其他节点
                _, tar_bit_pos, trols, rad_angles, nd = item
                mp_y_seo = self.parse_mp_y(tar_bit_pos, trols, rad_angles)
                new_results += mp_y_seo
        return new_results

    # ===========================================
    #            Generate qiskit qc
    # ===========================================

    def generate_qc(self) -> QuantumCircuit:
        num_qbits = self.emb.num_qbits_bef
        qc = QuantumCircuit(num_qbits)

        sum_global_phase = 0

        merged_results = self.merged_results
        for item in merged_results:
            item_type = item[0]
            if item_type == 'SIGX':  # two-qubit CX
                _, control, target = item
                qc.cx(control, target)
            else:  # one-qubit gateF
                _, theta, qubit = item
                if item_type == 'GLOBAL_PHAS':
                    # print('PHAS', theta)
                    sum_global_phase += theta
                    # gp(qc, theta, qubit)
                    continue
                else:  # ROTY, ROTZ
                    rota(qc, item_type, theta, qubit)

        print('SUM global phase', sum_global_phase)
        return qc


# 不知道为啥会有问题，放进去误差会变大
# global_phase
def gp(qc: QuantumCircuit, theta, qubit):
    qc.p(theta, qubit)
    qc.x(qubit)
    qc.p(theta, qubit)
    qc.x(qubit)


ROT_MAP = {
    'ROTX': 1,
    'ROTY': 2,
    'ROTZ': 3,
}


def rota(qc: QuantumCircuit, rot_type, angle_rads, tar_bit_pos):
    axis = ROT_MAP[rot_type]
    arr = OneQubitGate.rot_ax(angle_rads, axis)

    delta, left_rads, center_rads, right_rads = UnitaryMat.u2_zyz_decomp(arr)
    phi = -2 * left_rads
    theta = -2 * center_rads
    lam = -2 * right_rads

    if abs(phi) < 1e-6 and abs(theta) < 1e-6:
        qc.p(lam, tar_bit_pos)
    elif abs(theta - np.pi / 2) < 1e-6:
        qc.u(math.pi / 2, phi, lam, tar_bit_pos)
    else:
        qc.u(theta, phi, lam, tar_bit_pos)


def print_qubiter_result(t: CSDTree):
    merged_results = t.merged_results
    for i in range(len(merged_results)):
        item_type = merged_results[i][0]
        if item_type == 'diag':
            item = merged_results[i]
            rad_angles = "\t".join([format((i / math.pi) * 180, ".6f") for i in item[2]])
            print(f'DIAG\tIF\t{item[1].bit_pos_to_kind}\tBY\t{rad_angles}')
        elif item_type == 'multiplexor':
            item = merged_results[i]
            rad_angles = "\t".join([format((i / math.pi) * 180, ".6f") for i in item[3]])
            print(f'MP_Y\tAT\t{item[1]}\tIF\t{item[2].bit_pos_to_kind}\tBY\t{rad_angles}')
        elif item_type == 'SIGX':
            print(f'SIGX\tAT\t{merged_results[i][2]}\tIF\t{merged_results[i][1]}T')
        else:
            print(
                f'{merged_results[i][0]}\t{format((merged_results[i][1] / math.pi) * 180, ".6f")}\tAT\t{merged_results[i][2]}')
