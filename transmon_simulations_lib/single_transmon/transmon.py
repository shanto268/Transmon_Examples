from typing import List

import numpy as np
import qutip as qp
from transmon_simulations_lib.custom_ops import raising_op, lowering_op

# TODO: make everything to be single package "QHSim"
from .tmon_eigensystem import TmonEigensystem
from .tmon_eigensystem import my_transform, TmonPars

class Transmon():
    def __init__(
            self, pars_list: List[TmonPars],
            Nc=2, res_trunc=3, index=0):
        """
        Class represents single Tmon.
        It can calculate its spectrum in charge basis and represent
        charge operators in its spectral basis.

        Class can cache out eigenproblems solution for arbitrary mesh of
        parameters. Cache for eigenproblems solutions can be utilized in
        further intensive calculations.

        pars_list : list[TmonPars]
            list of tmon parameters space point to solve eigenvalue
            problem.
        Nc : int
            Maximum cooper pair number in charge basis (>=0).
            Charge basis size is `2*Nc + 1`.
        res_trunc : int
            number of eigenvectors used to represent result.
        index : int
            index of a transmon. For navigation in a Tmons structures
            that utilizes `Transmon(...)` instances.

        nonlinear_osc : bool
            Not Implemented
            TODO: ask Gleb, assumed fastened
                analytical solution for eigensystem, I presume.
        """
        self.pars_list: List[TmonPars] = pars_list

        # numerical parameters
        self.Nc = Nc  # maximum cooper pair number
        self.m_dim = 2 * self.Nc + 1  # matrix dimension

        # truncate output to this number of
        # dimensions
        self.res_trunc = res_trunc

        # index used if transmon is embedded into connected structure
        self.index = index

        # solutions for already solved parameter space points are stored
        # in this cache
        self._eigsys_sol_cache: dict[TmonPars, TmonEigensystem] = {}

    def clear_caches(self):
        self._eigsys_sol_cache = {}

    ''' GETTERS SECTION START '''

    def get_index(self):
        return self.index

    ''' GETTERS SECTION END '''

    ''' HELP FUNCTIONS SECTION START '''

    ''' HELP FUNCTIONS SECTION END '''

    def calc_Hc_cb(self, pars: TmonPars):
        """
        Calculate Hc in charge bassis.
        Ec = 2e^2/(C h) [GHz]

        Returns
        -------
        qp.Qobj
        """
        Hc = pars.Ec * qp.charge(self.Nc) ** 2
        return Hc

    def calc_Hj_cb(self, pars: TmonPars):
        """
        Generate Josephson energy matrix in charge bassis.
        phi = 2*pi Flux/Flux_quantum

        phi: float
            phase canonical variable
            .math: \phi_1 = \mathbf{\Phi_e}_1 + \phi
            \phi_2 = \mathbf{\Phi_e}_1 - \phi
        phiExt1: float
            External flux through simple loop that contains capacitance.
        phiExt2: float
            External flux through SQUID formed by JJ.
        Returns
        -------
        qp.Qobj
        """
        phiExt1 = pars.phiExt1
        phiExt2 = pars.phiExt2
        Ej = pars.Ej
        alpha = pars.alpha

        small_jj_op = alpha/2*(
                np.exp(-1j*phiExt1)*raising_op(self.m_dim) +
                np.exp(1j*phiExt1)*lowering_op(self.m_dim)
        )
        big_jj_op = 1/2*(
                np.exp(-1j*phiExt2)*raising_op(self.m_dim) +
                np.exp(1j*phiExt2)*lowering_op(self.m_dim)
        )

        return Ej*qp.Qobj(small_jj_op + big_jj_op)

    def calc_Hfull_cb(self, pars: TmonPars):
        """
        Calculate Hamiltonian operator matrix based on class parameters.
        Charge basis

        Returns
        -------
        qp.Qobj
        """
        return self.calc_Hc_cb(pars) + self.calc_Hj_cb(pars)

    def solve(self, sparse=False):
        """
        Solve eigensystem problem for every point supplied during
        construction and stored into `self.pars`.

        Returns
        -------
        list[TmonEigensystem]
            `TmonEigensystem` - database entry
        """
        result = []

        for pars_pt in self.pars_list:
            try:
                solution = self._eigsys_sol_cache[pars_pt]
                result.append(solution)
            except KeyError:
                H_full = self.calc_Hfull_cb(pars_pt)
                n_full = qp.charge(self.Nc)
                evals, evecs = H_full.eigenstates(
                    sparse=sparse, sort="low", eigvals=self.res_trunc
                )

                n_op = my_transform(n_full, evecs)
                solution = TmonEigensystem(
                    self.pars_list,
                    evecs=evecs,
                    evals=evals - evals[0],
                    n_op=n_op.tidyup()
                )
                self._eigsys_sol_cache[pars_pt] = solution
                result.append(solution)

        return result

    ''' for using in `qp.mcsolve` and `qp.mesolve` '''





