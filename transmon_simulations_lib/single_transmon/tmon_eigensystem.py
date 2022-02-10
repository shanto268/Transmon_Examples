import numpy as np
import scipy.sparse as sp
import qutip as qp


class TmonEigensystem:
    def __init__(self, Ec=None, Ej=None, alpha=None, d=None,
                 phi=None, evecs=None, H_op=None,
                 n_op=None, Nc=None):
        """
        Eigensystem solution as a class. `TmonEigensystem` consists of
            all relevant operators and parameters describing
            numerical problem in eigenbasis.

        Parameters
        ----------
        Ec: float
            Charge energy of a qubit [GHz]
            Explicit formula (SI): e^2/(2*C*h)*1e-9, C - capacitance,
            h - plank constant
        Ej: float
            energy of the largest JJ in transmon [GHz]
            .math
            I_k \Pni_0 / (2 * pi * h),
            I_k - junction critical current in SI, \Phi_0 - flux quantum.
        alpha : float
            asymmetry parameter. I_k1/I_k2 < 1, ration of lower critical
            current to the larger one.
        d: float
            asymmetry parameter, alternative to `alpha`.
            `d = (1 - alpha)/(1 + alpha)`. Used in Koch.
        phi: float
            flux phase of the transmon in radians (from `0` to `2 pi`).
        Nc : int
            Maximum cooper pair number in charge basis (>=0).
            Charge basis size is `2*Nc + 1`.
        evecs : list[qp.Qobj]
            list of ket objects representing eigenvectors of a problem.
        H_op: qp.Qobj
            Hamiltonian in its eigenbasis
        n_op: qp.Qobj
            cooper pair number operator in eigenbasis of a problem.
        """
        if (alpha is not None) and (d is not None):
            raise ValueError("only `alpha` or `d` can be supplied since "
                             "they are in one-to-one correspondence")
        self.Ec = Ec
        self.Ej = Ej
        if (alpha is not None) and (d is None):
            self.alpha = alpha
            self.d = (1 - alpha) / (1 + alpha)
        elif (d is not None) and (alpha is None):
            self.alpha = (1 - d) / (1 + d)
            self.d = d
        elif (d is None) and (alpha is None):
            raise ValueError("none of the asymmetry parameters "
                             "supplied: `d` or `alpha`")
        self.phi = phi
        self.evecs = evecs

        self.Nc = Nc
        self.Ns = 2*Nc+1
        self.H_op = H_op
        self.n_op = n_op

    ''' GETTER FUNCTIONS SECTION START '''

    def w01(self):
        return self.H_op[1, 1] - self.H_op[0, 0]

    def w12(self):
        return self.H_op[2, 2] - self.H_op[1, 1] \
            if len(self.H_op.shape[0]) > 2 else None

    def anharm(self):
        return self.w12() - self.w01()

    ''' GETTER FUNCTIONS SECTION END '''


def my_transform(operator, evecs, sparse=False):
    """
    Partial copy of Qobj.transform. The difference
    is this one works for members of a single Hermite space only
    (dims problem below) yet allows to
    shrink new operator to basis with less vectors.
    TODO: needs a little addition to work with several Hermite spaces (
        fix out.dims during creation)

    Parameters
    ----------
    operator : qp.qObj
        operator you wish to transform
    evecs : list[qp.ket]
    sparse : bool
        whether or to construct S-matrix as sparse

    Returns
    -------

    """
    if isinstance(evecs, list) or (isinstance(evecs, np.ndarray) and
                                   len(evecs.shape) == 1):
        if len(evecs) > max(operator.shape):
            raise TypeError(
                'Invalid size of ket list for basis transformation')
        if sparse:
            S = sp.hstack([psi.data for psi in evecs],
                          format='csr', dtype=complex).conj().T
        else:
            S = np.hstack([psi.full() for psi in evecs]).conj().T
    elif isinstance(evecs, qp.Qobj) and evecs.isoper:
        S = evecs.data
    elif isinstance(evecs, np.ndarray):
        S = evecs.conj()
        sparse = False
    else:
        raise TypeError('Invalid operand for basis transformation')

    if operator.isket:
        data = S * operator.data
    elif operator.isbra:
        data = operator.data.dot(S.conj().T)
    else:
        if sparse:
            data = S * operator.data * (S.conj().T)
        else:
            data = S.dot(operator.data.dot(S.conj().T))

    out = qp.Qobj(data)
    out._isherm = operator._isherm
    out.superrep = operator.superrep

    return out
