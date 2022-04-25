from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import qutip as qp


TmonPars = namedtuple(
    "TmonPars",
    ["Ec", "Ej", "alpha", "phiExt1", "phiExt2", "Nc"]
)


class TmonEigensystem:
    def __init__(self, tmon_pars: TmonPars, evecs=None, evals=None, n_op=None):
        """
        `TmonEigensystem` is the result of solving
        eigenvalue problem for a Tmon at a particular parameter space
        point. `TmonPars` used as a key in cache structure.
        `TmonPars` also contains its key for convinience.

        Parameters
        ----------
        tmon_pars: TmonPars
            Transmon's Hamiltonian parameters
        evecs : list[qp.Qobj]
            List of ket objects representing eigenvectors of a problem.
            Order is corresponding to `evals`.
        evals: np.ndarray
            list for eigenvalues ordered corresponding to `evecs`
        n_op: qp.Qobj
            cooper pair number operator in eigenbasis of a problem.
        """
        self.pars: TmonPars = tmon_pars
        self.evecs: list[qp.Qobj] = evecs
        self.evals: np.ndarray = evals
        self.n_op = n_op

    ''' GETTER FUNCTIONS SECTION START '''

    def E01(self):
        raise NotImplemented

    def E12(self):
        raise NotImplemented

    def anharmonicity(self):
        raise NotImplemented

    ''' GETTER FUNCTIONS SECTION END '''


def my_transform(operator, evecs, sparse=False):
    """
    Partial copy of Qobj.transform. The difference
    is this one works for members of a single Hermite space only
    (dims problem below, no support for tensor products) yet allows to
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
