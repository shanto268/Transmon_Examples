from typing import List, Callable

import numpy as np
import qutip as qp
import tqdm
from p_tqdm import p_tqdm
from functools import partial
import os

from transmon_simulations_lib.custom_ops import raising_op, lowering_op
# TODO: make everything to be single package "QHSim"
from .tmon_eigensystem import TmonEigensystem
from .tmon_eigensystem import my_transform, TmonPars
# https://docs.python.org/3/library/multiprocessing.html#multiprocessing.cpu_count
NUM_CPUS = os.cpu_count()
print("Transmon.py module: cpu number detected", NUM_CPUS)


class Transmon():
    def __init__(
            self, pars_list: List[TmonPars],
            Nc=2, res_trunc=5, index=0):
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

        # whether or not solutions sequence is in time domain
        # activates if supplied `TmonPars.Amp_d != 0`
        # also requires `TmonPars.t` to be supplied.
        self._time_domain = False

    def clear_caches(self):
        self._eigsys_sol_cache = {}

    ''' GETTERS SECTION START '''

    def get_index(self):
        return self.index

    ''' GETTERS SECTION END '''

    ''' HELP FUNCTIONS SECTION START '''

    ''' HELP FUNCTIONS SECTION END '''

    '''  QUBIT DIAGONALIZATION AS A STANDALONE DEVICE SECTION START '''
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
        \cos{(\phi - \varphi_{ext})}

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

        small_jj_op = alpha*(
                np.exp(-1j*phiExt1)*raising_op(self.m_dim) +
                np.exp(1j*phiExt1)*lowering_op(self.m_dim)
        )
        big_jj_op = (
                np.exp(-1j*phiExt2)*raising_op(self.m_dim) +
                np.exp(1j*phiExt2)*lowering_op(self.m_dim)
        )
        # 1/2 is transerred here from `small_jj_op` and `big_jj_op` for a
        # lil optimization
        return Ej/2*qp.Qobj(small_jj_op + big_jj_op)

    def calc_Hinternal_cb(self, pars_pt: TmonPars):
        """
        Calculate Hamiltonian operator matrix based on class
        parameters. Does not incude any kind of external objects coupling.
        Calculations and result returned are in charge basis.

        Returns
        -------
        qp.Qobj
        """
        Hinternal = self.calc_Hc_cb(pars_pt) + self.calc_Hj_cb(pars_pt)
        return Hinternal

    def solve_internal(self, pars_pt: TmonEigensystem, sparse=False):
        """
        Solve eigensystem problem for a given point in parameter space.
        Does not include any external couplings.
        Calculations and result returned both are in charge basis.


        Parameters
        ----------
        pars_pt : TmonEigensystem
        sparse : bool
            Solver regime

        Returns
        -------
        TmonEigensystem
            `TmonEigensystem` class containing solution in charge basis.
        """
        solution_internal = self._solve_eigsys_problem(
            pars_pt, self.calc_Hinternal_cb,
            sparse=sparse,
        )

        return solution_internal

    '''  QUBIT DIAGONALIZATION AS A STANDALONE DEVICE SECTION END '''

    def calc_Hdrive_cb(self, pars_pt: TmonPars):
        """
        Calculate capacitive external drive in qubit space after Mollow
        transform.
        Calculates only interaction part. Does not include drive field
        energy and Ec modification terms.

        Parameters
        ----------
        pars_pt : TmonPars
            point in parameter space

        Returns
        -------
        qp.Qobj
            drive operator in charge basis
        """
        Amp_d = pars_pt.Amp_d
        omega_d = pars_pt.omega_d
        phase_d = pars_pt.phase_d
        t = pars_pt.time
        if Amp_d == 0:
            return qp.qzero(self.m_dim)
        Hdrive = -Amp_d*np.cos(omega_d*t + phase_d) * qp.charge(self.Nc)
        return Hdrive

    def calc_Hfull_cb(self, pars_pt: TmonPars):
        """
        Calculate Hamiltonian operator matrix based on class parameters.
        Calculated and returned an the charge basis.

        Only supports 1 external drive at the moment.

        Returns
        -------
        qp.Qobj
        """
        Hfull = self.calc_Hinternal_cb(pars_pt)
        if pars_pt.Amp_d != 0:
            Hfull += self.calc_Hdrive_cb(pars_pt)

        return Hfull

    def calc_HdriveRWA_eb(self, pars_pt: TmonPars):
        """
        Calculate Hamiltonian operator matrix based on class
        parameters.
        Uses `self.solve_internal()` and returned an the charge basis.

        Only supports 1 external drive at the moment.

        Returns
        -------
        qp.Qobj
        """
        # solution in cooper basis
        solution_intenal_cb = self.solve_internal(pars_pt)
        return solution_intenal_cb

    def solve(self, rwa=False, sparse=False):
        """
        Solve eigensystem problem for every point in supplied during
        construction and stored into `self.pars`.

        rwa : bool
            if True, returns solution in Rotating Wave Approximation.
            if False, return solution without approximations.

        Parameters
        ----------
        rwa : bool
            Whether or not to utilize Rotating Wave Approximation for
            solution. RWA applied as transforming to interaction
            picture with `H0 = diag(0, -\omega_q, -2*\omega_q, ...,
            -n\omega_q)` in eigenbasis.
            So, before moving towards interaction picture,
            diagonalization of the qubit as a standalone device is
            calculated using `self.solve_internal(...)`.
        sparse : bool

        Returns
        -------
        list[TmonEigensystem]
            List of solutions. Basis is included in returned structe.
            If `rwa=True` supplied, than solution is in the qubit's
            internal eigenbasis obtained by calling
            `self.solve_internal()`.
        """
        result = []

        if rwa is True:
            # internal eigenbasis
            H_generator = self.calc_Hfull_RWA_eb
        else:
            H_generator = self.calc_Hfull_cb
            # Default behaviour, solve for every point without any
            # assumptions

        # parallelized solver
        for solution in p_tqdm.p_imap(
                partial(
                    self._solve_eigsys_problem,
                    H_generator=H_generator,
                    sparse=sparse
                ),
                self.pars_list,
                num_cpus=NUM_CPUS
        ):
            result.append(solution)

        return result

    def _solve_eigsys_problem(self,
                              pars_pt: TmonPars,
                              H_generator: Callable,
                              sparse=True, res_trunc=None):
        """
        Diagonalizes H_generator(pars_pt, sparse=sparse) and returns result
        as `TmonEigensystem`.
        Parameters
        ----------
        H_generator
        pars_pt
        sparse

        Returns
        -------

        """
        if res_trunc is None:
            # argument passed can override
            # internal settings for this `one-time` calculation
            res_trunc = self.res_trunc

        try:  # looking in cache for solution
            solution = self._eigsys_sol_cache[pars_pt]
        except KeyError:  # solve eigenproblem and store into cache
            Hfull = H_generator(pars_pt)
            n_full = qp.charge(self.Nc)
            evals, evecs = Hfull.eigenstates(
                sparse=sparse, sort="low",
                eigvals=res_trunc
            )

            n_op = my_transform(n_full, evecs)
            solution = TmonEigensystem(
                self.pars_list,
                evecs=evecs,
                evals=evals - evals[0],
                n_op=n_op.tidyup()
            )

            self._eigsys_sol_cache[pars_pt] = solution

        return solution
    ''' for using in `qp.mcsolve` and `qp.mesolve` '''





