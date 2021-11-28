import numpy as np
from qutip import *
from numpy import pi, sqrt


class Transmon:

    def __init__(self, Ec, Ej, d, gamma_rel, gamma_phi,
                 Nc, N_trunc, index, nonlinear_osc=False):
        self._Ec = Ec
        self._Ej = Ej
        self._d = d
        self._Nc = Nc
        self._Ns = Nc * 2 + 1
        self._gamma_rel = gamma_rel
        self._gamma_phi = gamma_phi
        self.index = index
        self._nonlinear_osc = nonlinear_osc

        self._linear_osc_H_stub = (sqrt(8 * Ej * Ec) - Ec) * create(N_trunc) * destroy(N_trunc)
        self._nonlinear_osc_H_stub = - Ec / 2 * create(N_trunc) * destroy(N_trunc) * \
                                     (create(N_trunc) * destroy(N_trunc) - 1)
        self._nonlinear_osc_raising = create(N_trunc)
        self._nonlinear_osc_lowering = destroy(N_trunc)
        self._nonlinear_osc_n = destroy(N_trunc) + create(N_trunc)

        self._N_trunc = N_trunc

        self._n_cache = {}
        self._c_ops_cache = {}
        self._Hdr_RF_RWA_cache = {}
        self._Hdr_cont_cache = {}
        self._H_diag_trunc_cache = {}

    def clear_caches(self):
        self._n_cache = {}
        self._c_ops_cache = {}
        self._Hdr_RF_RWA_cache = {}
        self._Hdr_cont_cache = {}
        self._H_diag_trunc_cache = {}

    def _truncate(self, operator):
        return Qobj(operator[:self._N_trunc, :self._N_trunc])

    def Hc(self):
        return 4 * (self._Ec) * charge(self._Nc) ** 2

    def Hj(self, phi):
        return - self._Ej / 2 * tunneling(self._Ns, 1) * self._phi_coeff(phi)

    def H_diag_trunc(self, phi):
        if self._nonlinear_osc:
            return self.H_nonlinear_osc(phi)

        try:
            return self._H_diag_trunc_cache[phi]
        except KeyError:
            H_charge_basis = self.Hc() + self.Hj(phi)
            #H_fixed_flux = self.Hc() + self.Hj(phi_base)
            #evals, evecs = H_fixed_flux.eigenstates()
            evals, evecs = H_charge_basis.eigenstates()
            H = self._truncate(H_charge_basis.transform(evecs))
            self._H_diag_trunc_cache[phi] = H - H[0, 0]
            return self._H_diag_trunc_cache[phi]
        
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
    def H_uu(self, freq):
        Huu = freq*2*pi*ket2dm(self.e_state())+2*freq*2*pi*ket2dm(self.f_state())
        return Huu
    
    def H_diag_trunc_approx(self, phi):
        Ej = self._Ej
        Ec = self._Ec
        omega_01=sqrt(8*Ec*Ej*self._phi_coeff(phi))
        H = self.e_state()*self.e_state().dag()*(omega_01-Ec) + self.f_state()*self.f_state().dag()*(2*omega_01-3*Ec)
        return H

    
    def H_td_diag_trunc_approx(self, waveform):
        Ej = self._Ej
        Ec = self._Ec
        omega_01=sqrt(8*Ec*Ej*self._phi_coeff(waveform))
        H11 = [self.e_state()*self.e_state().dag(),omega_01 - Ec]
        H22 = [self.f_state()*self.f_state().dag(),2*omega_01 - 3*Ec]
        # approximating f_q = f_q^max * sqrt(cos sqrt(1+ d^2tan^2))
        return [H11] + [H22]
    
    def _phi_coeff(self, phi):
        return (np.cos(phi * pi)**2 + (self._d)**2 * np.sin(phi * pi) ** 2) ** 0.5

    def get_Ns(self):
        return self._N_trunc

    def get_index(self):
        return self.index

    def Hdr(self, amplitude, duration, start, phase=0, freq=None, flux = 0):

        if freq is None:
            freq = self.ge_freq_approx(flux)

        if self._nonlinear_osc:
            return [self._nonlinear_osc_n,
                    "%f*cos(2*pi*%.16f*t+%f)*(1+np.sign(t-%f))*(1+np.sign(-t+%f))/4" % \
                    (amplitude, freq, phase, start, start + duration)]

        return [self.n(0) / self.n(0).matrix_element(self.g_state(), self.e_state()),
                "%f*cos(2*pi*%.16f*t+%f)*(1+np.sign(t-%f))*(1+np.sign(-t+%f))/4" % \
                (amplitude, freq, phase, start, start + duration)]
    
    
    def Hdr_RF_RWA(self, amplitude, start, duration, phase):

        if self._nonlinear_osc:
            op = self._nonlinear_osc_n
           
        else:
            op1 = (self.n(0) / self.n(0).matrix_element(self.g_state(),
                                                      self.e_state())).data.copy()
            for i in range(op1.shape[0]):
                for j in range(op1.shape[1]):
                    if i < j:
                        op1[i, j] = 0
            half = Qobj(op1, dims = [[3],[3]])*np.exp(-1j*phase)
            op = half + half.dag()       
            return [op, "0.5*%f*(1+np.sign(t-%f))*(1+np.sign(-t+%f))/4"%\
                                                 (amplitude, start, start + duration)]
       
        
        #op = self.n(0) / self.n(0).matrix_element(self.g_state(), self.e_state())
        #return [op, "0.5*%f*(1+np.sign(t-%f))*(1+np.sign(%f-t))/4"%\
                                                 #(amplitude, start, start + duration)]
        
        
    def n_approx (self, phi):
        
        try:
            return self._n_cache_approx[phi]
        except:
            self._n_cache_approx[phi] = self.n(0)*(self._phi_coeff(phi)**0.25)
            return self._n_cache_approx[phi]
        
        
    def n(self, phi):

        if self._nonlinear_osc:
            return self._nonlinear_osc_n

        try:
            return self._n_cache[phi]
        except:
            H_charge_basis = self.Hc() + self.Hj(phi)
            evals, evecs = H_charge_basis.eigenstates()
            self._n_cache[phi] = self._truncate(Qobj(abs(charge(self._Nc).transform(evecs))))
            return self._n_cache[phi]    
        
    def g_state(self):
        #         evals, evecs = self.H(phi).eigenstates()
        #         return evecs[0]
        return basis(self._N_trunc, 0)

    def e_state(self):
        return basis(self._N_trunc, 1)

    def f_state(self):
        return basis(self._N_trunc, 2)   
    
    
    def sz(self):
        return -ket2dm(basis(3, 0)) + ket2dm(basis(3, 1))

    def sx(self):
        return basis(3, 0) * basis(3, 1).dag() + basis(3, 1) * basis(3, 0).dag()

    def sy(self):
        return 1j * basis(3, 0) * basis(3, 1).dag() - 1j * basis(3, 1) * basis(3, 0).dag()
    

        
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
        return phi_coeff
    '''
    
    
    
