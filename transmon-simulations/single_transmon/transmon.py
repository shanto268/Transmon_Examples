import numpy as np
import qutip as qp
import bisect

# TODO: make everything to be single package "QHSim"
from .tmon_eigensystem import TmonEigensystem

class Transmon:
    def __init__(self, Ec=0.6, Ej=28, alpha=0.2, d=None, gamma_rel=0.0,
                 gamma_phi=0.0,
                 Nc=2, N_trunc=2, index=0, nonlinear_osc=False):
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
        gamma_rel : float
            longitudal relaxation frequency.
            For lindblad operator expressed as
            `sqrt(gamma_rel/2) \sigma_m`.
            And lindblad entry is defiened as `2 L p L^+ - {L^+ L,p}`.
        gamma_phi : float
            phase relaxation frequency.
            For lindblad operator expressed as
            `sqrt(gamma_phi) \sigma_m`.
            And lindblad entry is defiened as `2 L p L^+ - {L^+ L,p}`.
        Nc : float
            Charge basis size is `2*Nc + 1`.
        N_trunc : int
            number of eigensystem components with lowest energy that all
            operators should be trunctated to.
        index : int
            index of a transmon. For navigation in a Tmons structures
            that utilizes `Transmon(...)` instances.
        nonlinear_osc : bool
            Not Implemented (TODO: ask Gleb)
        """
        self._Ec = Ec
        self._Ej = Ej
        if (d is None) and (alpha is not None):
            self._d = (1 - alpha) / (alpha + 1)
            self._alpha = alpha
        elif (alpha is None) and (d is not None):
            self._d = d
            self._alpha = (1 - d) / (1 + d)

        self._Nc = Nc
        self._Ns = Nc * 2 + 1
        self._gamma_rel = gamma_rel
        self._gamma_phi = gamma_phi
        self.index = index
        self._nonlinear_osc = nonlinear_osc
        if self._nonlinear_osc is True:
            raise NotImplementedError("`nonlinear_osc=True` not "
                                      "implemented yet")

        self._linear_osc_H_stub = (np.sqrt(2 * Ej * Ec) - Ec / 4) * \
                                  qp.create(
            N_trunc) * qp.destroy(N_trunc)
        # self._nonlinear_osc_H_stub = \
        #     - Ec / 2 * create(N_trunc) * destroy(
        #     N_trunc) * (create(N_trunc) * destroy(N_trunc) - 1)
        # self._nonlinear_osc_raising = create(N_trunc)
        # self._nonlinear_osc_lowering = destroy(N_trunc)
        # self._nonlinear_osc_n = destroy(N_trunc) + create(N_trunc)

        self._N_trunc = N_trunc

        self._ops_phi_cache: list[TmonEigensystem] = []

    def clear_caches(self):
        self._ops_phi_cache = []

    def _truncate(self, operator, n_trunc=None):
        if n_trunc is not None:
            pass
        else:
            n_trunc = self._N_trunc
        return qp.Qobj(operator[:n_trunc, :n_trunc])

    def h_c(self):
        return (4 * self._Ec) * qp.charge(self._Nc) ** 2

    def h_j(self, phi):
        return - self._Ej / 2 * qp.tunneling(self._Ns,
                                             1) * self._phi_coeff(
            phi)

    def h_diag_trunc(self, phi, from_cache=False, interpolation_order=0):
        if self._nonlinear_osc:
            raise NotImplementedError
        if from_cache:
            try:
                return self._H_diag_trunc_cache[phi]
            except KeyError:
                raise NotImplementedError(
                    "interpolation from cache is "
                    "not implemented yet"
                )

        else:
            h_charge_basis = self.h_c() + self.h_j(phi)
            # H_fixed_flux = self.Hc() + self.Hj(phi_base)
            # evals, evecs = H_fixed_flux.eigenstates()
            evals, evecs = h_charge_basis.eigenstates()
            h = self._truncate(h_charge_basis.transform(evecs))
            te = TmonEigensystem(Ec=self._Ec, Ej=self._Ej,
                                 alpha=self._alpha)
            self._ops_phi_cache.insort(h - h[0, 0])
            return self._H_diag_trunc_cache[phi]

    def H_uu(self, freq):
        Huu = freq * qp.ket2dm(
            self.e_state()) + 2 * freq * qp.ket2dm(self.f_state())
        return Huu

    def H_diag_trunc_approx(self, phi):
        Ej = self._Ej
        Ec = self._Ec
        omega_01 = np.sqrt(2 * Ec * Ej * self._phi_coeff(phi))
        H = self.e_state() * self.e_state().dag() * (
                    omega_01 - Ec/4) + self.f_state() * self.f_state(

        ).dag() * (
                        2 * omega_01 - 3 * Ec/4)
        return H

    def H_td_diag_trunc_approx(self, waveform):
        Ej = self._Ej
        Ec = self._Ec
        omega_01 = np.sqrt(8 * Ec * Ej * self._phi_coeff(waveform))
        H11 = [self.e_state() * self.e_state().dag(), omega_01 - Ec]
        H22 = [self.f_state() * self.f_state().dag(),
               2 * omega_01 - 3 * Ec]
        # approximating f_q = f_q^max * sqrt(cos sqrt(1+ d^2tan^2))
        return [H11] + [H22]

    def _phi_coeff(self, phi):
        return (np.cos(phi) ** 2 + (self._d) ** 2 * np.sin(
            phi) ** 2) ** 0.5

    def get_Ns(self):
        return self._N_trunc

    def get_index(self):
        return self.index

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

        # op = self.n(0) / self.n(0).matrix_element(self.g_state(),
        # self.e_state())
        # return [op, "0.5*%f*(1+np.sign(t-%f))*(1+np.sign(%f-t))/4"%\
        # (amplitude, start, start + duration)]

    def n_approx(self, phi):
        try:
            return self._n_cache_approx[phi]
        except:
            self._n_cache_approx[phi] = self.n(0) * (
                        self._phi_coeff(phi) ** 0.25)
            return self._n_cache_approx[phi]

    def g_state(self):
        return qp.basis(self._N_trunc, 0)

    def e_state(self):
        return qp.basis(self._N_trunc, 1)

    def f_state(self):
        return qp.basis(self._N_trunc, 2)

    def n(self, phi):

        if self._nonlinear_osc:
            raise NotImplementedError

        try:
            return self._ops_phi_cache[phi]
        except:
            H_charge_basis = self.h_c() + self.h_j(phi)
            evals, evecs = H_charge_basis.eigenstates()
            self._n_cache[phi] = self._truncate(
                qp.Qobj(abs(qp.charge(self._Nc).transform(evecs))))
            return self._n_cache[phi]

def foo(args):
    return args["phi_offset"]

    '''
        def Hj_td(self, phi_waveform):
            return [- self._Ej / 2 * tunneling(self._Ns, 1), self._phi_coeff(phi_waveform)]
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

        evecs = [basis(self._N_trunc, i) for i in range(self._N_trunc)]
        return sum([abs(self.n(phi).matrix_element(evecs[j + 1], evecs[j])) /
                    abs(self.n(phi).matrix_element(evecs[0], evecs[1])) *
                    evecs[j + 1] * evecs[j].dag() for j in range(0, self._N_trunc - 1)])

    def lowering(self, phi):
        if self._nonlinear_osc:
            return self._nonlinear_osc_lowering

        evecs = [basis(self._N_trunc, i) for i in range(self._N_trunc)]
        return sum([abs(self.n(phi).matrix_element(evecs[j], evecs[j + 1])) /
                    abs(self.n(phi).matrix_element(evecs[0], evecs[1])) *
                    evecs[j] * evecs[j + 1].dag() for j in range(0, self._N_trunc - 1)])

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

