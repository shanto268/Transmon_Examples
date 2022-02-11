import numpy as np
import qutip as qp
from transmon_simulations_lib.helper_ops import raising_op, lowering_op
import bisect

from collections import namedtuple

SolKey = namedtuple("Solkey", ["Ec", "Ej", "alpha", "phi", "Nc"])

# TODO: make everything to be single package "QHSim"
from .tmon_eigensystem import TmonEigensystem
from .tmon_eigensystem import my_transform


class Transmon:
    def __init__(self, Ec=0.6, Ej=28, alpha=0.2, d=None, phi=None,
                 gamma_rel=0.0,
                 gamma_phi=0.0,
                 Nc=2, eigspace_N=2, index=0, nonlinear_osc=False):
        """
        Class represents single Tmon.
        It can calculate its spectrum in charge basis and represent
        charge operators in its spectral basis.

        In order to fasten calculations that often require sweep over
        flux variable, eigenproblem solution is can be obtained on an
        arbitrary mesh before the extensive calcs take place. These
        solutions then can be used to sample from (or interpolate from)
        to get eigenproblem solution faster.

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
            longitudal relaxation frequency.
            For Lindblad operator expressed as
            `sqrt(gamma_rel/2) \sigma_m`.
            And lindblad entry is defiened as `2 L p L^+ - {L^+ L,p}`.
        gamma_phi : float
            phase relaxation frequency.
            For lindblad operator expressed as
            `sqrt(gamma_phi) \sigma_m`.
            And lindblad entry is defiened as `2 L p L^+ - {L^+ L,p}`.
        Nc : int
            Maximum cooper pair number in charge basis (>=0).
            Charge basis size is `2*Nc + 1`.
        eigspace_N : int
            number of eigensystem components with lowest energy that all
            operators should be restricted onto.
        index : int
            index of a transmon. For navigation in a Tmons structures
            that utilizes `Transmon(...)` instances.

        nonlinear_osc : bool
            Not Implemented
            TODO: ask Gleb, assumed fastened
                analytical solution for eigensystem, I presume.
        """
        self.Ec = Ec
        self.Ej = Ej
        if (d is None) and (alpha is not None):
            self.d = (1 - alpha) / (alpha + 1)
            self.alpha = alpha
        elif (alpha is None) and (d is not None):
            self.d = d
            self.alpha = (1 - d) / (1 + d)

        self.phi = phi
        self.Nc = Nc
        self.eigspace_N = eigspace_N
        # index used if transmons is build into connected strucre
        self.index = index

        # dimension of a charge basis
        self.space_dim = Nc * 2 + 1
        self._gamma_rel = gamma_rel
        self._gamma_phi = gamma_phi

        self._nonlinear_osc = False
        if self._nonlinear_osc is True:
            raise NotImplementedError("`nonlinear_osc=True` not "
                                      "implemented yet")

        self._eigsys_sol_cache: dict[SolKey, TmonEigensystem] = {}

    def clear_caches(self):
        self._eigsys_sol_cache = {}

    ''' GETTERS SECTION START '''

    def get_Ns(self):
        return self.eigspace_N

    def get_index(self):
        return self.index

    ''' GETTERS SECTION END '''

    ''' HELP FUNCTIONS SECTION START '''

    def _phi_coeff(self, phi):
        return np.sign(np.cos(phi))*np.sqrt(
            1 + self.alpha ** 2 + 2 * self.alpha * np.cos(phi)
        )

    ''' HELP FUNCTIONS SECTION END '''

    def calc_Hc_cb(self):
        """
        Calculate Hc in charge bassis.

        Returns
        -------
        qp.Qobj
        """
        Hc = self.Ec * qp.charge(self.Nc) ** 2
        return Hc

    def calc_Hj_cb(self, phi):
        """
        Calculate Josephson energy in charge bassis.
        phi = pi Flux/Flux_quantum
        Returns
        -------
        qp.Qobj
        """
        import scipy.stats
        scipy.stats.norm()
        scalar = - self.Ej * self._phi_coeff(phi)\
                 / 2
        phi0 = np.arctan(self.d * np.tan(phi))
        op = np.exp(-1j * phi0) * raising_op(
            self.space_dim) + \
             np.exp(1j * phi0) * lowering_op(self.space_dim)
        return scalar * qp.Qobj(op)

    def calc_Hj_cb2(self, phi):
        """
        Calculate Josephson energy in charge bassis.
        phi = pi Flux/Flux_quantum
        Returns
        -------
        qp.Qobj
        """
        scalar = - np.sign(np.cos(phi))*self.Ej * \
            self._phi_coeff(phi) / 2
        op = qp.tunneling(self.space_dim, 1)
        return scalar * op

    def calc_Hfull_cb(self, phi):
        """
        Calculate total Hamiltonian from class parameters

        Returns
        -------
        qp.Qobj
        """
        return self.calc_Hc_cb() + self.calc_Hj_cb(phi)

    def calc_Hfull_cb2(self, phi):
        return self.calc_Hc_cb() + self.calc_Hj_cb2(phi)

    def solve(self, use=1):
        """
        Solve eigensystem problem and return operators in

        Returns
        -------
        list[TmonEigensystem]
            Eigensystem solution as a class. `TmonEigensystem` consists of
            all relevant operators and parameters describing
            numerical problem in eigenbasis.
        """
        result = []
        ctr = False
        if isinstance(self.phi, np.ndarray):
            if isinstance(self.phi, list):
                self.phi = np.array(self.phi, dtype=float)
            sol_keys = (SolKey(self.Ec, self.Ej, self.alpha, phi,
                               self.Nc) for phi in self.phi)
        else:
            sol_keys = (SolKey(self.Ec, self.Ej, self.alpha, self.phi,
                               self.Nc),)  # 1 entry tuple

        for sol_key in sol_keys:
            try:
                solution = self._eigsys_sol_cache[sol_key]
                result.append(solution)
            except KeyError:
                if use == 1:
                    if not ctr:
                        print("use 1")
                        ctr = True
                    H_full = self.calc_Hfull_cb(sol_key.phi)
                else:
                    if not ctr:
                        print("use 2")
                        ctr = True
                    H_full = self.calc_Hfull_cb2(sol_key.phi)
                n_full = qp.charge(self.Nc)
                evals, evecs = H_full.eigenstates(sort="low")

                # TODO: parallelize
                H_op = my_transform(H_full, evecs)
                n_op = my_transform(n_full, evecs)

                H_op = H_op - H_op[0, 0] * qp.identity(self.space_dim)
                solution = TmonEigensystem(
                    self.Ec, self.Ej, self.alpha,
                    phi=sol_key.phi, evecs=evecs,
                    H_op=H_op.tidyup(), n_op=n_op.tidyup(), Nc=self.Nc
                )
                self._eigsys_sol_cache[sol_key] = solution
                result.append(solution)
        if isinstance(self.phi, np.ndarray) and len(self.phi) == 1:
            return solution
        else:
            return result

    def Hdr_RF_RWA(self, amplitude, start, duration, phase):
        if self._nonlinear_osc:
            raise NotImplementedError

        else:
            op1 = (self.n(0) / self.n(0).matrix_element(self.g_state(),
                                                        self.e_state())).data.copy()
            for i in range(op1.shape[0]):
                for j in range(op1.shape[1]):
                    if i < j:
                        op1[i, j] = 0
            half = qp.Qobj(op1, dims=[[3], [3]]) * np.exp(-1j * phase)
            op = half + half.dag()
            return [
                op,
                "0.5*%f*(1+np.sign(t-%f))*(1+np.sign(-t+%f))/4" %
                (amplitude, start, start + duration)
            ]

    def g_state(self):
        return qp.basis(self.eigspace_N, 0)

    def e_state(self):
        return qp.basis(self.eigspace_N, 1)

    def f_state(self):
        return qp.basis(self.eigspace_N, 2)

    '''
        def Hj_td(self, phi_waveform):
            return [- self.Ej / 2 * tunneling(self.cb_dim, 1), self._phi_coeff(phi_waveform)]
        '''

    """
    def H_diag_trunc(self, phi):
        if self._nonlinear_osc:
            return self.H_nonlinear_osc(phi)

        try:
            return self._H_diag_trunc_cache[phi]
        except KeyError:
            H_charge_basis = self.Hc() + self.Hj(phi)
            evals, evecs = H_charge_basis.eigenstates()
            H = self._truncate(H_charge_basis.transform(evecs))
            self._H_diag_trunc_cache[phi] = H - H[0, 0]
            return self._H_diag_trunc_cache[phi]
    """

    '''
    def H_nonlinear_osc(self, phi):
        return self._linear_osc_H_stub * self._phi_coeff(phi) + self._nonlinear_osc_H_stub
    '''
    '''
        def Hdr(self, amplitude, duration, start, phase=0, freq=None, flux=0):
            raise NotImplementedError
            if freq is None:
                freq = self.ge_freq_approx(flux)

            if self._nonlinear_osc:
                return [self._nonlinear_osc_n,
                        "%f*cos(2*pi*%.16f*t+%f)*(1+np.sign(t-%f))*(1+np.sign(-t+%f))/4" % \
                        (amplitude, freq, phase, start, start + duration)]

            return [self.n(0) / self.n(0).matrix_element(self.g_state(),
                                                         self.e_state()),
                    "%f*cos(2*pi*%.16f*t+%f)*(1+np.sign(t-%f))*(1+np.sign(-t+%f))/4" % \
                    (amplitude, freq, phase, start, start + duration)]'''
    '''    
    def H_td_diag_trunc_approx_str_old(self, params, number):
        if (number == 1):
            base = 0
            offset = params['phi_offset']
            t_start = params['start']
            t_finish = params['start'] + params['duration']
            delta = 0.1
            tau = delta *(t_finish - t_start)
            wf_rise = "(t-%f)*(1+np.sign(t-%f))*(1+np.sign(%f + %f -t))/%f/4*%f"%(t_start, t_start, t_start, tau, tau, offset) 
            wf_down = "(%f - t)*(1 + np.sign(t-%f+%f))*(1 + np.sign(%f - t))/4/%f*%f"%(t_finish, t_finish, tau, t_finish, tau, offset)
            wf_flat = "(1+np.sign(t - %f - %f))*(1+np.sign(%f -%f -t))/4*%f"%(t_start, tau, t_finish, tau, offset) 
            waveform = wf_rise + ' + ' + wf_flat + ' + ' + wf_down + ' + %f'%base
            return [self.H_diag_trunc(0), self._phi_coeff_str(waveform) ]
        if (number == 2):
            base = 1/2
            return [self.H_diag_trunc(0), '%f  + t - t'%base ]   
    '''

    '''
    def eigenlevels_approx(self, phi):
        evals = self.H_diag_trunc_approx(phi).eigenenergies()
        return evals

    def ge_freq_approx(self, phi):
        evals = self.H_diag_trunc_approx(phi).eigenenergies()
        return (evals[1] - evals[0]) / 2 / pi
    '''

    '''    
    def raising(self, phi):
        if self._nonlinear_osc:
            return self._nonlinear_osc_raising

        evecs = [basis(self.eigspace_N, i) for i in range(self.eigspace_N)]
        return sum([abs(self.n(phi).matrix_element(evecs[j + 1], evecs[j])) /
                    abs(self.n(phi).matrix_element(evecs[0], evecs[1])) *
                    evecs[j + 1] * evecs[j].dag() for j in range(0, self.eigspace_N - 1)])

    def lowering(self, phi):
        if self._nonlinear_osc:
            return self._nonlinear_osc_lowering

        evecs = [basis(self.eigspace_N, i) for i in range(self.eigspace_N)]
        return sum([abs(self.n(phi).matrix_element(evecs[j], evecs[j + 1])) /
                    abs(self.n(phi).matrix_element(evecs[0], evecs[1])) *
                    evecs[j] * evecs[j + 1].dag() for j in range(0, self.eigspace_N - 1)])

    def rotating_dephasing(self, phi):
        return self.raising(phi) * self.lowering(phi)

    def c_ops(self, phi):
        try:
            return self._c_ops_cache[phi]
        except KeyError:
            self._c_ops_cache[phi] = [sqrt(self._gamma_rel) * self.lowering(phi),
                                      sqrt(self._gamma_phi) * self.rotating_dephasing(phi)]
            return self._c_ops_cache[phi]

    '''

    # driving!! utilized in double-tone spectroscopy
    '''

    def Hdr_cont(self, amplitude):

        if self._nonlinear_osc:
            op = self._nonlinear_osc_n
        else:
            op = self.n(0) / self.n(0).matrix_element(self.g_state(),
                                                      self.e_state())

        try:
            return self._Hdr_cont_cache[amplitude]
        except KeyError:
            self._Hdr_cont_cache[amplitude] = [
                amplitude * op,
                "cos(wd%d*t)" % self.index]
            return self._Hdr_cont_cache[amplitude]

    def Hdr_cont_RF_RWA(self, amplitude):

        if self._nonlinear_osc:
            op = self._nonlinear_osc_n
        else:
            op = self.n(0) / self.n(0).matrix_element(self.g_state(),
                                                      self.e_state())

        try:
            return self._Hdr_RF_RWA_cache[amplitude]
        except KeyError:
            self._Hdr_RF_RWA_cache[amplitude] = amplitude / 2 * op
            return self._Hdr_RF_RWA_cache[amplitude]
    '''
    '''
    def waveform_td (self, offset, t_start, t_finish):
        delta = 0.1
        tau = delta *(t_finish - t_start)
        th_sigma = 0.2
        wf_rise_sign = "(t-%f)*(1+np.sign(t-%f))*(1+np.sign(%f + %f -t))/%f/4*%f"%(t_start, t_start, t_start, tau, tau, offset) 
        wf_down_sign = "(%f - t)*(1 + np.sign(t-%f+%f))*(1 + np.sign(%f - t))/4/%f*%f"%(t_finish, t_finish, tau, t_finish, tau, offset)
        wf_flat_sign = "(1+np.sign(t - %f - %f))*(1+np.sign(%f -%f -t))/4*%f"%(t_start, tau, t_finish, tau, offset) 
        
        wf_rise_tanh = "((np.tanh((t - 2 *%f - %f) / %f)+1)/2)"%(th_sigma, t_start, th_sigma)
        wf_down_tanh = "((np.tanh((-t - 2 *%f + %f) / %f)+1)/2)"%(th_sigma, t_finish, th_sigma)
        
        waveform_sign = '( ' + wf_rise_sign + ' + ' + wf_flat_sign + ' + ' + wf_down_sign + ' )'
        waveform_tanh = '( ' + wf_rise_tanh + ' * '  + wf_down_tanh + ' *%f )'%(offset)
        return waveform_tanh
    
    def H_td_diag_trunc_approx_str(self, waveform):
        return [self.H_diag_trunc_approx(0), self._phi_coeff_str(waveform) ]
    
    def _phi_coeff_str (self, waveform): #takes waveform string and gives phicoeff string
        phi_coeff = "np.power( ( np.cos( " + waveform  + " * pi) )**2 + " + "(%f * np.sin( "%self._d + waveform + " * pi)) ** 2 ,0.25) "
        return phi_co
'''
