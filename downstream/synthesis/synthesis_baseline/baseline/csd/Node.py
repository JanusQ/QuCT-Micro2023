import csd.utilities_gen as ut

from csd.UnitaryMat import *
from scipy.linalg import cossin


class Node:
    """
    This class carries the "cargo" of each node of a tree. Included in that
    cargo are

    * self's parent and 2 children nodes,
    * its side, level and id,
    * its left, central and right matrix lists.

    This class also performs the very important task of calling within its
    constructor the function CS_Decomp.get_csd() which fills the
    node's left, central and right matrix lists.

    Attributes
    ----------
    central_mats : list(np.ndarray)
        Central matrix list returned by call to CS_Decomp.get_csd( ). A
        central_mats is a list of dmats. A dmat= D matrix  is numpy array
        containing floats (radian angles).
    left_mats : list(np.ndarray)
        Left matrix list returned by call to CS_Decomp.get_csd()
    left_nd : Node
        Node to left of self.
    level : int
        1<= level <= num_qbits+1. level = 1 for root node, level =
        num_of_bits+1 for node whose central_mat is list of 1 dim arrays
    nd_id : int
        node id, int assigned by Tree, nd_id=0 for first (root) node created
        by Tree, nd_id=1 for second node created, etc.
    pa_nd : Node
        parent node
    right_mats : list(np.ndarray)
        Right matrix list returned by call to CS_Decomp.get_csd()
    right_nd : Node
        Node to right of self.
    side : str
        to which side of its parent does self find itself, either 'right' or
        'left'


    """

    def __init__(self, nd_id, pa_nd, side, init_unitary_mat=None):
        """
        Constructor

        Parameters
        ----------
        nd_id : int
        pa_nd : Node|None
        side : str|None
        init_unitary_mat : np.ndarray
            This is the matrix that is fed to CS_Decomp.get_csd() in root
            node constructor. pa_nd and side are ignored if this is not None.
        Returns
        -------


        """

        self.nd_id = nd_id
        self.pa_nd = pa_nd  # pa=parent, nd=node
        self.side = side  # either 'left' or 'right'
        self.level = None
        # "is None" does not work for numpy array,
        if ut.is_arr(init_unitary_mat):
            pa_nd = None
            side = None

        self.left_nd = None
        self.right_nd = None

        # mats = matrices
        # left_mats, central_mats and right_mats are all list(nd.array)
        self.left_mats = None
        self.central_mats = None
        self.right_mats = None

        if ut.is_arr(init_unitary_mat):  # 头结点的分解
            self.level = 1
            [self.left_mats, self.central_mats, self.right_mats] = \
                self.get_csd([init_unitary_mat])
            # release memory
            init_unitary_mat = None
        else:
            self.level = pa_nd.level + 1
            in_mats = None
            if side == 'left':
                in_mats = pa_nd.left_mats
                pa_nd.left_nd = self
            elif side == 'right':
                in_mats = pa_nd.right_mats
                pa_nd.right_nd = self
            else:
                assert False
            # central就是D吧 
            [self.left_mats, self.central_mats, self.right_mats] = \
                self.get_csd(in_mats)
            if ut.is_arr(in_mats):
                # release memory
                in_mats = None

    @staticmethod
    def get_csd(unitary_mats):
        """
        This function does a CS (cosine-sine) decomposition (by calling the
        LAPACK function cuncsd.f. The old C++ Qubiter called zggsvd.f
        instead) of each unitary matrix in the list of arrays unitary_mats.
        This function is called by the constructor of the class Node and is
        fundamental for decomposing a unitary matrix into multiplexors and
        diagonal unitaries.

        Parameters
        ----------
        unitary_mats : list(np.ndarray)

        Returns
        -------
        list(np.ndarray), list(np.ndarray), list(np.ndarray)

        """
        block_size = unitary_mats[0].shape[0]
        num_mats = len(unitary_mats)
        for mat in unitary_mats:
            assert mat.shape == (block_size, block_size)

        if block_size == 1:  # 根节点
            left_mats = None
            right_mats = None
            vec = np.array([unitary_mats[k][0, 0]
                            for k in range(0, num_mats)])
            vec1 = vec[0] * np.ones((num_mats,))
            if np.linalg.norm(vec - vec1) < 1e-6:
                central_mats = None
            else:
                c_vec = np.real(vec)
                s_vec = np.imag(vec)
                central_mats = np.arctan2(s_vec, c_vec)
        else:
            left_mats = []
            central_mats = []
            right_mats = []

            for mat in unitary_mats:
                dim = mat.shape[0]
                assert dim % 2 == 0
                hdim = dim >> 1  # half dimension

                (u1, u2), theta, (v1t, v2t) = cossin(mat, p=hdim, q=hdim, separate=True, swap_sign=True)

                left_mats.append(u1)
                left_mats.append(u2)
                central_mats.append(theta)
                right_mats.append(v1t)
                right_mats.append(v2t)

        return left_mats, central_mats, right_mats

    def get_circuit(self):
        return
        # qiskit.circuit

    def is_barren(self):
        """
        Returns True iff node's left, central and right matrix lists are all
        None.

        Returns
        -------
        bool

        """
        return self.left_mats is None and \
            self.central_mats is None and \
            self.right_mats is None

    def make_barren(self):
        """
        Sets node's left, central and right matrix lists to None.

        Returns
        -------
        None

        """
        self.left_mats = None
        self.central_mats = None
        self.right_mats = None

    def __str__(self):
        """
        Gives a readable description of self when self is ordered to print.
        For example, if nd_id = 3, level = 4, node prints as '3(L4)'.

        Returns
        -------
        str

        """
        return str(self.nd_id) + '(L' + str(
            self.level) + ')'  # + f'{self.left_mats.shape},{self.central_mats.shape},{self.right_mats.shape}'


if __name__ == "__main__":
    def main():
        print(5)


    main()
