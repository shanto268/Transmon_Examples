from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import qutip as qp


TmonPars = namedtuple(
    "TmonPars",
    [
        "Ec",
        "Ej",
        "alpha",
        "phiExt1",
        "phiExt2",
        "gamma_rel",
        "gamma_phi"
    ]
)
'''
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
        gamma_rel : float
            longitudal relaxation speed in GHz.
            For Lindblad operator expressed as
            `sqrt(gamma_rel/2) \sigma_m`.
            And lindblad entry is defiened as `2 L p L^+ - {L^+ L,p}`.
        gamma_phi : float
            phase relaxation frequency.
            For lindblad operator expressed as
            `sqrt(gamma_phi) \sigma_m`.
            And lindblad entry is defiened as `2 L p L^+ - {L^+ L,p}`.
'''


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
            cooper pair number operator representation in eigenbasis
        """
        self.pars: TmonPars = tmon_pars
        self.evecs: list[qp.Qobj] = evecs
        self.evals: np.ndarray = evals
        self.n_op: qp.Qobj = None

    ''' GETTER FUNCTIONS SECTION START '''

    def n_op(self, truncate_N=3):
        if self.n_op is None:
            n_op = qp.charge(len(self.evals))
            n_op_eigenbasis = my_transform(
                n_op, self.evecs[:truncate_N]
            )
        else:
            return self.n_op

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
