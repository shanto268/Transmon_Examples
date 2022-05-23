from typing import List, Callable

import numpy as np
import scipy.sparse as spsp
import qutip as qp
import tqdm
from p_tqdm import p_tqdm
from functools import partial
import os
from copy import deepcopy

from transmon_simulations_lib.custom_ops import raising_op, lowering_op
# TODO: make everything to be single package "QHSim"
from .tmon_eigensystem import TmonEigensystem
from .tmon_eigensystem import my_transform, TmonPars, TMON_BASIS
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
        Eigenbasis is sorted in order ascending by eigenvalues. The
        lowest eigenvalue is forced to be zero by shifting all
        eigenvalues by certain number.

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
        return self.inde
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

    def solve_internal_cb(self, pars_pt: TmonPars, sparse=False,
                          res_trunc: int =None):
        """
        Solve eigensystem problem for a given point in parameter space.
        Does not include any external couplings.
        Calculations and result returned both are in charge basis.


        Parameters
        ----------
        pars_pt : TmonPars
        sparse : bool
            Solver regime
        res_trunc : int
            positive integers corresponding to amount
            of eigenvectors of lowest energy subspace requested

        Returns
        -------
        TmonEigensystem
            `TmonEigensystem` class containing solution in charge basis.
        """
        solution_internal_cb = self._solve_eigsys_problem(
            pars_pt, self.calc_Hinternal_cb,
            sparse=sparse, res_trunc=res_trunc,
            basis=TMON_BASIS.COOPER_PAIRS_BASIS
        )

        return solution_internal_cb
    '''  QUBIT DIAGONALIZATION AS A STANDALONE DEVICE SECTION END '''

    ''' QUBIT WITH EXTERNAL CAPACITIVE DRIVE SECTION START '''
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

    # TODO: `calc_HdriveRWA_eb` pending deletion due to no use
    def calc_HdriveRWA_eb(self, pars_pt: TmonPars, sparse=True,
                          res_trunc: int = None):
        """
        Calculates Drive Hamiltonian of the Transmon with single
        capacitive external coupling utilizing RWA approximation.
        Operator is returned in qubit's internal eigenbasis.

        Hamiltonain parameters supplied with `pars_pt`.

        Notes
        -------
        Utilizes
        `self.solve_internal(
            pars_pt=pars_pt, sparse=sparse, res_trunc=res_trunc
        )`

        RWA:
            Cooper pair number operator representation
        in qubit's internal eigenbasis is modified.
            Then all
        but nearest-neighbour couplings between energy levels are
        neglected to average to zero.
            This is achieved saving only 1st upper diagonal `n_l` and 1st lower
        diagonal `n_p`  of cooper pair number operator `n_op`
        represented in qubit's internal eigenbasis.

        Analytic result for RWA drive term in qubit's internal
        eigenbasis is:
            .math:
        H_{d}^{(RWA+dip)} = - 1/2 \A_d cos(\omega_d t + \phi_d)(n_l+n_p)

        Does not include drive field energy and Ec modification terms.

        Only supports 1 external drive at the moment.

        Parameters
        -------
        pars_pt : TmonPars
            Hamiltonian parameters. See `TmonPars` definition.
        sparse : bool
            Whether or not to invoke sparse solver for solving internal
            eigensystem.
        res_trunc : int
            positive integer defines number of lowest energy subspace
            eigenvectors requested

        Returns
        -------
        qp.Qobj
            Drive Hamiltonian in RWA approximation truncated to lowest
            energy subspace containinig `res_trunc` eigenvectors.
            Qubit's internal eigenbasis representation.
        """
        # solution in Cooper Pairs Number basis
        solution_intenal_cb = self.solve_internal_cb(
            pars_pt=pars_pt, sparse=sparse, res_trunc=res_trunc
        )
        n_op = solution_intenal_cb.n_op
        # lowering part of cooper pair number operator (1st upper diagonal)
        n_l = qp.Qobj(
            spsp.spdiags(
                n_op.data.diagonal(k=1), 1, self.m_dim, self.m_dim
            )
        )
        # uppering part of cooper pair number operator (1st lower diagonal)
        n_p = qp.Qobj(
            spsp.spdiags(
                n_op.data.diagonal(k=-1), -1, self.m_dim, self.m_dim
            )
        )
        ''' `n_l` and `n_p` parts of `n_op` both remain in RWA 
        approximation. Other terms from `n_op` has to be neglected '''
        Amp_d = pars_pt.Amp_d
        omega_d = pars_pt.omega_d
        phase_d = pars_pt.phase_d
        t = pars_pt.time
        n_op_rwa = qp.Qobj(n_l + n_p)
        return -Amp_d*np.cos(omega_d*t + phase_d)*n_op_rwa

    # TODO: `calc_HfullRWA_eb` pending deletion due to no use
    def calc_HfullRWA_eb(self, pars_pt: TmonPars,
                          sparse=True,
                          res_trunc: int = None):
        """
        Calculates full Hamiltonian of Transmon with single capacitive
        external coupling utilizing RWA approximation.
        Operator is returned in qubit external eigenbasis.

        Hamiltonain parameters supplied with `pars_pt`.

        Notes
        -------
        Utilizes
        `self.solve_internal(
            pars_pt=pars_pt, sparse=sparse, res_trunc=res_trunc
        )`

        RWA:
            Cooper pair number operator representation
        in qubit's internal eigenbasis is modified.
            Then all
        but nearest-neighbour couplings between energy levels are
        neglected to average to zero.
            This is achieved saving only 1st upper diagonal `n_l` and 1st lower
        diagonal `n_p`  of cooper pair number operator `n_op`
        represented in qubit's internal eigenbasis.

        Analytic result for RWA drive term in qubit's internal basis is:
            .math:
        H^{(RWA+dip)} = H_{internal} - H_0 - 1/2 \A_d cos(\omega_d t + \phi_d)(n_l+n_p)

        Does not include drive field energy and Ec modification terms.

        Only supports 1 external drive at the moment.

        Parameters
        -------
        pars_pt : TmonPars
            Hamiltonian parameters. See `TmonPars` definition.
        sparse : bool
            Whether or not to invoke sparse solver for solving internal
            eigensystem.
        res_trunc : int
            positive integer defines number of lowest energy subspace
            eigenvectors requested

        Returns
        -------
        qp.Qobj
            Full Hamiltonian in RWA approximation truncated to lowest
            energy subspace containinig `res_trunc` eigenvectors.
            Qubits internal eigenbasis representation.
        """
        HdriveRWA_eb = self.calc_HdriveRWA_eb(
            pars_pt=pars_pt, sparse=sparse, res_trunc=res_trunc
        )
        # eigenvalues should be already calculated and cached in
        # previous line of code
        evals = self.solve_internal_cb(
            pars_pt=pars_pt, sparse=sparse, res_trunc=res_trunc
        ).evals_arr

        Hinternal_eb = qp.Qobj(
            np.diag(evals, k=0)
        )

        Hfull_RWA_eb = Hinternal_eb + HdriveRWA_eb
        return Hfull_RWA_eb

    def calc_HdriveRWA_dip(self,  pars_pt: TmonPars,
                          sparse=True,
                          res_trunc: int = None):
        """
        Calculates Drive Hamiltonian of the Transmon with single
        capacitive external coupling utilizing RWA approximation.
        Operator is returned in "dip" basis.

        Hamiltonain parameters supplied with `pars_pt`.

        Notes
        -------
        Utilizes
        `self.solve_internal(
            pars_pt=pars_pt, sparse=sparse, res_trunc=res_trunc
        )`

        RWA:
            Cooper pair number operator representation
        in qubit's internal eigenbasis is modified.
            Then all
        but nearest-neighbour couplings between energy levels are
        neglected to average to zero.
            This is achieved saving only 1st upper diagonal `n_l` and 1st lower
        diagonal `n_p`  of cooper pair number operator `n_op`
        represented in qubit's internal eigenbasis.

        "dip":
            Stays for Drive Interaction Picture defined by
        `H0 = diag(0, \omega_d, 2*\omega_d, ..., res_trunc*\omega_d)`
        Analytic result for RWA drive term in "dip" is:
            .math:
        H_{d}^{(RWA+dip)} = - 1/2 \Omega_d *
        (n_l e^{-i phase_d} + n_p e^{i phase_d}

        Does not include drive field energy and Ec modification terms.

        Only supports 1 external drive at the moment.

        Parameters
        -------
        pars_pt : TmonPars
            Hamiltonian parameters. See `TmonPars` definition.
        sparse : bool
            Whether or not to invoke sparse solver for solving internal
            eigensystem.
        res_trunc : int
            positive integer defines number of lowest energy subspace
            eigenvectors requested

        Returns
        -------
        qp.Qobj
            Drive Hamiltonian in RWA approximation truncated to lowest
            energy subspace containinig `res_trunc` eigenvectors.
            "dip" basis representation.
        """

        # solution in Cooper Pairs Number basis
        solution_intenal_cb = self.solve_internal_cb(
            pars_pt=pars_pt, sparse=sparse, res_trunc=res_trunc
        )
        n_op = solution_intenal_cb.n_op
        # lowering part of cooper pair number operator (1st upper diagonal)
        n_l = qp.Qobj(
            spsp.spdiags(
                n_op.data.diagonal(k=1), 1, self.m_dim, self.m_dim
            )
        )
        # uppering part of cooper pair number operator (1st lower diagonal)
        n_p = qp.Qobj(
            spsp.spdiags(
                n_op.data.diagonal(k=-1), -1, self.m_dim, self.m_dim
            )
        )
        ''' `n_l` and `n_p` parts of `n_op` both remain in RWA 
        approximation. Other terms from `n_op` has to be neglected '''
        Amp_d = pars_pt.Amp_d
        phase_d = pars_pt.phase_d
        Hdrive_RWA_dip = -1/2 * Amp_d * \
                         (n_l*np.exp(-1j*phase_d) + n_p*np.exp(1j*phase_d))
        return Hdrive_RWA_dip

    def calc_HfullRWA_dip(self, pars_pt: TmonPars,
                          sparse=True,
                          res_trunc: int = None):
        """
        Calculates full Hamiltonian of Transmon with single capacitive
        external coupling utilizing RWA approximation.
        Operator is returned in "dip" basis.

        Hamiltonain parameters supplied with `pars_pt`.

        Notes
        -------
        Utilizes
        `self.solve_internal(
            pars_pt=pars_pt, sparse=sparse, res_trunc=res_trunc
        )`

        RWA:
            Cooper pair number operator representation
        in qubit's internal eigenbasis is modified.
            Then all
        but nearest-neighbour couplings between energy levels are
        neglected to average to zero.
            This is achieved saving only 1st upper diagonal `n_l` and 1st lower
        diagonal `n_p`  of cooper pair number operator `n_op`
        represented in qubit's internal eigenbasis.

        "dip":
            Stays for Drive Interaction Picture defined by
        `H0 = diag(0, \omega_d, 2*\omega_d, ..., res_trunc*\omega_d)`
        Analytic result for RWA drive term in "dip" is:
            .math:
        H^{(RWA+dip)} = H_{internal} - H_0 - 1/2 \Omega_d *
        (n_l e^{-i phase_d} + n_p e^{i phase_d}

        Does not include drive field energy and Ec modification terms.

        Only supports 1 external drive at the moment.

        Parameters
        -------
        pars_pt : TmonPars
            Hamiltonian parameters. See `TmonPars` definition.
        sparse : bool
            Whether or not to invoke sparse solver for solving internal
            eigensystem.
        res_trunc : int
            positive integer defines number of lowest energy subspace
            eigenvectors requested

        Returns
        -------
        qp.Qobj
            Full Hamiltonian in RWA approximation truncated to lowest
            energy subspace containinig `res_trunc` eigenvectors.
            "dip" basis representation.
        """
        HdriveRWA_dip = self.calc_HdriveRWA_dip(
            pars_pt=pars_pt, sparse=sparse, res_trunc=res_trunc
        )
        # eigenvalues should be already calculated and cached in
        # previous line of code
        evals = self.solve_internal_cb(
            pars_pt=pars_pt, sparse=sparse, res_trunc=res_trunc
        ).evals_arr

        # Calculating H_{internal} - H_0
        omega_d = pars_pt.omega_d
        Hinternal_dip = qp.Qobj(
            np.diag(evals, k=0)
        )
        H0 = qp.Qobj(np.diag([i*omega_d for i in range(res_trunc)], k=0))
        Hfull_RWA_dip = Hinternal_dip - H0 + HdriveRWA_dip

        return Hfull_RWA_dip

    ''' SOLVER SECTION START '''
    def solve(self, rwa=False, dip=False, sparse=False, res_trunc=None):
        """
        Solve eigensystem problem for every point in array`self.pars`.
        `self.pars` is passed to class constructor.

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
            calculated using `self.solve_internal_cb(...)`.
        dip : bool
            Use drive interaction picture representation.
            Only works if `rwa==True`
        sparse : bool
            Whether or not to invoke sparse solver

        Returns
        -------
        list[TmonEigensystem]
            List of solutions. Basis is included in returned structure.
            If `rwa=True` supplied, than solution is in the qubit's
            internal eigenbasis obtained by calling
            `self.solve_internal()`.
        """
        result = []

        if rwa is True:
            # internal eigenbasis
            H_generator = self.calc_HfullRWA_eb
            basis = TMON_BASIS.INTERNAL_EIGENBASIS
            if dip is True:
                H_generator = self.calc_HfullRWA_dip
                basis = TMON_BASIS.DRIVE_INTERACTION_PICTURE
        else:
            H_generator = self.calc_Hfull_cb
            basis = TMON_BASIS.COOPER_PAIRS_BASIS
            # Default behaviour, solve for every point without any
            # assumptions

        # parallelized solver
        for solution in p_tqdm.p_imap(
                partial(
                    self._solve_eigsys_problem,
                    H_generator=H_generator,
                    sparse=sparse,
                    basis=basis
                ),
                self.pars_list,
                num_cpus=NUM_CPUS
        ):
            result.append(solution)

        return result

    def _solve_eigsys_problem(self,
                              pars_pt: TmonPars,
                              H_generator: Callable,
                              sparse=True,
                              res_trunc=None, cache_it=True,
                              basis=TMON_BASIS.COOPER_PAIRS_BASIS):
        """
        Diagonalizes H_generator(pars_pt=pars_pt, sparse=sparse) and
        returns result as `TmonEigensystem` containing `res_trunc`
        amount of
        eigenvectors corresponding to lowest energy.

        Parameters
        ----------
        H_generator : Callable
            H_generator(pars_pt=`pars_pt`, sparse=`sparse`) is called and
            assumed to return Hamiltonian which eigensystem solution is
            requested.
        pars_pt : TmonPars
            Hamiltonian parameters space point, where eigenproblem is
            to be solved.
        sparse : bool
            Whether or not to use sparse solver
        res_trunc : int
            positive integer of how many eigenvectors to find.
            Eigenvectors count starts with lowest eigenvalue.
        cache_it : bool
            Whether or not to store result in a cache.
        basis : TMON_BASIS
            Enum instance indicates basis that objects are represented at.
            Default - cooper pair basis.

        Returns
        -------
        TmonEigensystem
            class containing info about solution and its calculation
            details. Automatically stored into `self._eigsys_sol_cache`,
            unless `cache_it==False`.
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

            pars_pt.basis = basis
            solution = TmonEigensystem(
                pars_pt,
                evecs=evecs,
                evals=evals - evals[0],
                n_op=n_op.tidyup()
            )

            if cache_it:
                self._eigsys_sol_cache[pars_pt] = solution

        return solution
    ''' SOLVER SECTION END '''





