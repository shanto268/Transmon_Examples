import scipy
import qutip as qp
from single_transmon import
from matplotlib.pyplot import *
from tqdm import *

class DoubleTransmonSystem:
    def __init__(self, tr1, tr2, g):
        self._tr1 = tr1
        self._tr2 = tr2
        self._tr_list = [tr1, tr2]
        self._g = g

        self._Huu_cache = {}
        
    def clear_caches(self):
        self._Huu_cache = {}
        self._tr1.clear_caches()
        self._tr2.clear_caches()

    def two_qubit_operator(self, qubit1_operator=None, qubit2_operator=None):

        mask = [qp.identity(self._tr1.get_Ns()), qpidentity(self._tr2.get_Ns())]

        if qubit1_operator is not None:
            mask[0] = qubit1_operator
        if qubit2_operator is not None:
            mask[1] = qubit2_operator

        return qp.tensor(*mask)

    def H(self, phi1, phi2):
        H = self.two_qubit_operator(qubit1_operator=self._tr1.h_diag_trunc(phi1)) + \
            self.two_qubit_operator(qubit2_operator=self._tr2.h_diag_trunc(phi2)) + \
            self.Hint(phi1, phi2)
        return H - H[0, 0]

    def H_RF_RWA(self, phi1, phi2, rotating_frame_freq):
        H = self.two_qubit_operator(qubit1_operator=self._tr1.h_diag_trunc(phi1)) + \
            self.two_qubit_operator(qubit2_operator=self._tr2.h_diag_trunc(phi2)) + \
            self.Hint_RF_RWA(phi1, phi2)

        return H - H[0, 0] - self._Huu(rotating_frame_freq)
    
    
    def H_iswap_RF_RWA_td(self, waveform1, waveform2, params, rotations):
        ## H1
        dur1, dur2 = rotations[0][0], rotations[1][0]
        phi1, phi2 = rotations[0][1], rotations[1][1]
        flux1 = params['phi_base_level']
        Htd1 = self._tr1.H_td_diag_trunc_approx(waveform1)
        evals1, evecs1 = self._tr1.H_diag_trunc_approx(flux1).eigenstates()
        freq1 = (evals1[1]- evals1[0])/2/pi
        Hdr1 = self._tr1.Hdr_RF_RWA(params['drive_amplitude'], params['start'] - dur1 - 10, dur1, phi1)
        Huu1 = self._tr1.H_uu(freq1)*(-1)
        H1 = Htd1 + [Hdr1] + [Huu1]
        H1_2q = [[self.two_qubit_operator(qubit1_operator = H1[0][0]),H1[0][1]],
        [self.two_qubit_operator(qubit1_operator = H1[1][0]),H1[1][1]],
        [self.two_qubit_operator(qubit1_operator = H1[2][0]),H1[2][1]],
        self.two_qubit_operator(qubit1_operator = H1[3])]
        ## H2
        flux2 = params['phi2z_base_level']
        Htd2 = self._tr2.H_td_diag_trunc_approx(waveform2)
        evals2, evecs2 = self._tr2.H_diag_trunc_approx(flux2).eigenstates()
        freq2 = (evals2[1]- evals2[0])/2/pi
        Hdr2 = self._tr2.Hdr_RF_RWA(params['drive_amplitude'], params['start'] - dur2 - 10, dur2, phi2)
        Huu2 = self._tr2.H_uu(freq2)*(-1)
        H2 = Htd2 + [Hdr2] + [Huu2]
        H2_2q = [[self.two_qubit_operator(qubit2_operator = H2[0][0]),H2[0][1]],
        [self.two_qubit_operator(qubit2_operator = H2[1][0]),H2[1][1]],
        [self.two_qubit_operator(qubit2_operator = H2[2][0]),H2[2][1]],
        self.two_qubit_operator(qubit2_operator = H2[3])]
        Hint = self.Hint_RF_RWA(flux1, flux2, (freq2-freq1)*2*pi, td = True)
        H = H1_2q + H2_2q + Hint
        #print (dur1, dur2, phi1 ,phi2)
        return H
    

    def _Huu(self, freq):
        try:
            return self._Huu_cache[freq]
        except:
            self._Huu_cache[freq] = freq * 2 * pi * self.two_qubit_operator(
                qubit1_operator=ket2dm(self._tr1.e_state())) + \
                                    2 * freq * 2 * pi * self.two_qubit_operator(
                qubit1_operator=ket2dm(self._tr1.f_state())) + \
                                    freq * 2 * pi * self.two_qubit_operator(
                qubit2_operator=ket2dm(self._tr2.e_state())) + \
                                    2 * freq * 2 * pi * self.two_qubit_operator(
                qubit2_operator=ket2dm(self._tr2.f_state()))
            return self._Huu_cache[freq]

    def H_td(self, waveform1, waveform2):
        return [self.Hc()] + self.Hj_td(waveform1, waveform2) + [self.Hint()]
    
    def H_td_diag_approx_str(self, params, iswap2 = True):
        waveform1_iswap1 = 't-t'
        waveform1_iswap2 = 't-t '
        waveform1_zgate1 = 't-t '
        waveform1_zgate2 = 't-t '
        waveform2_zgate2 = 't-t '
        if (iswap2):
            gap = 120.00
            waveform1_iswap2 = self._tr1.waveform_td(params['phi_offset'],params['start'] + gap,params['start'] + params['duration'] +gap)
            waveform1_zgate2 = self._tr1.waveform_td(params['phiz_offset'], params['start'] + 90 +120, 
                                                     params['start'] + 90 + gap + params['t_zgate'])
            waveform2_zgate2 = self._tr2.waveform_td(params['phi2z_offset'], params['start'] + 90 +gap, 
                                                 params['start'] + 90 + gap + params['t_zgate2'])
            
        waveform1_iswap1 = self._tr1.waveform_td(params['phi_offset'],params['start'],params['start'] + params['duration'])
        
        waveform1_iswap =  '( ' + waveform1_iswap1 + ' + ' +  waveform1_iswap2 + ' )'
        #waveform1_zgate = 't-t '
        waveform1_zgate1 = self._tr1.waveform_td(params['phiz_offset'], params['start'] + 90, params['start'] + 90 + params['t_zgate'])
        
        waveform1_zgate = '( ' + waveform1_zgate1 + ' + ' +  waveform1_zgate2 + ' )'
        waveform1_sum = '( ' + waveform1_iswap + ' + ' + waveform1_zgate + ' + ' + ' %f'%params['phi_base_level']  + ' )'      
        #waveform2_zgate = 't-t '
        waveform2_zgate1 = self._tr2.waveform_td(params['phi2z_offset'], params['start'] + 90, params['start'] + 90 + params['t_zgate2'])
        
        waveform2_zgate = '( ' + waveform2_zgate1 + ' + ' +  waveform2_zgate2 + ' )'
        
        waveform2_sum = '( ' + waveform2_zgate + ' + ' + ' %f'%params['phi2z_base_level'] + ' )'
        Hj_td1 = self._tr1.H_td_diag_trunc_approx_str(waveform1_sum)        
        Hj_td1[0] = self.two_qubit_operator(qubit1_operator=Hj_td1[0])
       
        Hj_td2 = self._tr2.H_td_diag_trunc_approx_str(waveform2_sum)
        Hj_td2[0] = self.two_qubit_operator(qubit2_operator=Hj_td2[0])
       
        Hint = [self.Hint_RF_RWA( 0.4770,0.5),"1"]
        return [Hj_td1, Hj_td2, Hint]
    
    def H_td_diag_approx(self, waveform1, waveform2):
        Hj_td1 = self._tr1.H_td_diag_trunc_approx(waveform1)
        Hj_td1[0] = self.two_qubit_operator(qubit1_operator=Hj_td1[0])
        Hj_td2 = self._tr2.H_td_diag_trunc_approx(waveform2)
        Hj_td2[0] = self.two_qubit_operator(qubit2_operator=Hj_td2[0])
        #return [Hj_td1, Hj_td2, self.Hint(waveform1[0], waveform2[0])]
        return [Hj_td1, Hj_td2, self.Hint_RF_RWA_waveform(waveform1, waveform2)]
        #return [self.Hint_RF_RWA_waveform(waveform1, waveform2)]

    def H_diag_approx(self, phi1, phi2):
        return self.two_qubit_operator(qubit1_operator=self._tr1.H_diag_trunc_approx(phi1)) + \
               self.two_qubit_operator(qubit2_operator=self._tr2.H_diag_trunc_approx(phi2)) + \
               self.Hint_RF_RWA(0.4770, 0.5)

    def Hc(self):
        return self.two_qubit_operator(qubit1_operator=self._tr1.h_c()) + \
               self.two_qubit_operator(qubit2_operator=self._tr2.h_c())

    def Hj(self, phi1, phi2):
        return self.two_qubit_operator(qubit1_operator=self._tr1.h_j(phi1)) + \
               self.two_qubit_operator(qubit2_operator=self._tr2.h_j(phi2))

    def Hj_td(self, waveform1, waveform2):
        Hj_td1 = self._tr1.Hj_td(waveform1)
        Hj_td1[0] = self.two_qubit_operator(qubit1_operator=Hj_td1[0])
        Hj_td2 = self._tr2.Hj_td(waveform2)
        Hj_td2[0] = self.two_qubit_operator(qubit2_operator=Hj_td2[0])
        return [Hj_td1, Hj_td2]

    def Hint(self, phi1, phi2):
        return self.two_qubit_operator(self._tr1.n(phi1), self._tr2.n(phi2)) * self._g

    def Hint_RF_RWA(self, phi1, phi2, offset = 0.02, td = False):
        op1 = self._tr1.n(phi1).data.copy()
        op2 = self._tr2.n(phi2).data.copy()
        # zeroing triangles
        for i in range(op1.shape[0]):
            for j in range(op1.shape[1]):
                if i < j:
                    op1[i, j] = 0
                if i > j:
                    op2[i, j] = 0
        half = self.two_qubit_operator(Qobj(op1), Qobj(op2))
        if td:
            return [[self._g * half, "exp(-1j*(%f)*t)"%offset],[self._g * half.dag(), "exp(1j*(%f)*t)"%offset]]
        else:
            return self._g * (half + half.dag())
    def Hint_RF_RWA_waveform(self, waveform1, waveform2, td = False):
        return [self.Hint_RF_RWA(0, 0.5), array([self._tr1._phi_coeff(phi)**0.25 for phi in waveform1])]
                #array([(self._tr1._phi_coeff(waveform1[i])*self._tr2._phi_coeff(waveform2[i])) for i in range(len (waveform1))])]

    def Hdr(self, amplitudes, durations, starts, phases, freqs,fluxes):
        #Hdr1 = self._tr1.Hdr_cont(amplitudes[0], durations[0], starts[0], phases[0])
        Hdr1 = self._tr1.Hdr(amplitudes[0], durations[0], starts[0], phases[0], freqs[0], fluxes[0])
        Hdr1 = [self.two_qubit_operator(qubit1_operator=Hdr1[0]), Hdr1[1]]

        #Hdr2 = self._tr2.Hdr(amplitudes[1], durations[1], starts[1], phases[1])
        Hdr2 = self._tr2.Hdr(amplitudes[1], durations[1], starts[1], phases[1], freqs[1], fluxes[1])
        Hdr2 = [self.two_qubit_operator(qubit2_operator=Hdr2[0]), Hdr2[1]]

        return [Hdr1] + [Hdr2]

    def Hdr_cont(self, amplitudes):
        Hdr1 = self._tr1.Hdr_cont(amplitudes[0])
        Hdr1 = [self.two_qubit_operator(qubit1_operator=Hdr1[0]), Hdr1[1]]

        Hdr2 = self._tr2.Hdr_cont(amplitudes[1])
        Hdr2 = [self.two_qubit_operator(qubit2_operator=Hdr2[0]), Hdr2[1]]

        return [Hdr1] + [Hdr2]

    def Hdr_cont_RF_RWA(self, amplitudes):
        return [self.two_qubit_operator(qubit1_operator=self._tr1.Hdr_cont_RF_RWA(amplitudes[0]))]+\
        [self.two_qubit_operator(qubit2_operator=self._tr2.Hdr_cont_RF_RWA(amplitudes[1]))]

    def _remove_global_phase(self, state):
        state_full = state.full()
        return state * sign(state_full[argmax(abs(state_full))])[0]

    def gg_state(self, phi1, phi2, energy=False):
        evals, evecs = self.H_diag_approx(phi1, phi2).eigenstates()
        evec = self._remove_global_phase(evecs[0])

        return evec if not energy else (evec, evals[0])

    def e_state(self, phi1, phi2, qubit_idx, energy=False):
        evals, evecs = self.H_diag_approx(phi1, phi2).eigenstates()

        if qubit_idx == 1:
            model_state = self.two_qubit_operator(self._tr1.e_state(), self._tr2.g_state())
        elif qubit_idx == 2:
            model_state = self.two_qubit_operator(self._tr1.g_state(), self._tr2.e_state())

        for idx, evec in enumerate(evecs):
            if abs((evec.dag() * model_state).full()) > 0.6:
                evec = self._remove_global_phase(evec)
                return evec if not energy else (evec, evals[idx])
    
    def f_state(self, phi1, phi2, qubit_idx, energy=False):
        evals, evecs = self.H_diag_approx(phi1, phi2).eigenstates()

        if qubit_idx == 1:
            model_state = self.two_qubit_operator(self._tr1.f_state(), self._tr2.g_state())
        elif qubit_idx == 2:
            model_state = self.two_qubit_operator(self._tr1.g_state(), self._tr2.f_state())

        for idx, evec in enumerate(evecs):
            if abs((evec.dag() * model_state).full()) > 0.6:
                evec = self._remove_global_phase(evec)
                return evec if not energy else (evec, evals[idx])

    def ee_state(self, phi1, phi2, energy=False):
        evals, evecs = self.H_diag_approx(phi1, phi2).eigenstates()
        model_state = self.two_qubit_operator(self._tr1.e_state(), self._tr2.e_state())
        for idx, evec in enumerate(evecs):
            if abs((evec.dag() * model_state).full()) > 0.6:
                evec = self._remove_global_phase(evec)
                return evec if not energy else (evec, evals[idx])

    def c_ops(self, phi1, phi2):
        c_ops = []
        c_ops1 = self._tr1.c_ops(phi1)
        for c_op in c_ops1:
            c_ops.append(self.two_qubit_operator(qubit1_operator=c_op))
        c_ops2 = self._tr2.c_ops(phi2)
        for c_op in c_ops2:
            c_ops.append(self.two_qubit_operator(qubit2_operator=c_op))
        return c_ops

    def plot_spectrum(self, phi1s, phi2s, exact_H = True, currents=None):
        assert len(phi1s) == len(phi2s)

        fluxes = list(zip(phi1s, phi2s))

        fixed_flux_spectra = []
        if (exact_H):
            for phi1, phi2 in tqdm_notebook(fluxes, desc='Energy levels calculation'):
                H = self.H(phi1, phi2)
                evals = H.eigenenergies()
                fixed_flux_spectra.append(evals)
        else:
            for phi1, phi2 in tqdm_notebook(fluxes, desc='Energy levels calculation'):
                H_approx = self.H_diag_approx(phi1, phi2)
                evals = H_approx.eigenenergies()
                fixed_flux_spectra.append(evals)

        fixed_flux_spectra = array(fixed_flux_spectra)
        eigenlevels = fixed_flux_spectra.T
        transitions_from_g = eigenlevels - eigenlevels[0]
        
        self._single_photon_transitions = transitions_from_g[1:3].T / 2 / pi
        self._two_photon_transitions = transitions_from_g[3:6].T / 2 / pi / 2

        if currents is not None:
            plot(currents, transitions_from_g[1:3].T / 2 / pi)
            plot(currents, transitions_from_g[3:6].T / 2 / pi / 2)
        else:
            plot(phi1s, transitions_from_g[1:3].T / 2 / pi)
            plot(phi1s, transitions_from_g[3:6].T / 2 / pi / 2)

        grid()

    
    def plot_per_qubit_xyz_dynamics(self, Ts, result):
        X1, Y1, Z1 = [], [], []
        X2, Y2, Z2 = [], [], []
        X1 = result.expect[0]
        Y1 = result.expect[1]
        Z1 = result.expect[2]
        X2 = result.expect[3]
        Y2 = result.expect[4]
        Z2 = result.expect[5]
        fig, axes = subplots(2, 1, figsize=(15, 6))
        axes[0].plot(Ts, X1, label=r"$\langle x\rangle$")
        axes[0].plot(Ts, Y1, label=r"$\langle y\rangle$")
        axes[0].plot(Ts, Z1, label=r"$\langle z\rangle$")
        axes[1].plot(Ts, X2, label=r"$\langle x\rangle$")
        axes[1].plot(Ts, Y2, label=r"$\langle y\rangle$")
        axes[1].plot(Ts, Z2, label=r"$\langle z\rangle$")
        for ax in axes:
            ax.grid()
            ax.legend()
        return fig, axes
        '''
        states_rf = []
        for t, state in zip(Ts, states):
            U = (1j * t * self.H_diag_approx(phi1, phi2)).expm()
            states_rf.append(U * state * U.dag())

        X1, Y1, Z1 = [], [], []
        X2, Y2, Z2 = [], [], []
        for state in states_rf:
            X1.append(expect(self.two_qubit_operator(qubit1_operator=self._tr1.sx()), state))
            Y1.append(expect(self.two_qubit_operator(qubit1_operator=self._tr1.sy()), state))
            Z1.append(expect(self.two_qubit_operator(qubit1_operator=self._tr1.sz()), state))

            X2.append(expect(self.two_qubit_operator(qubit2_operator=self._tr2.sx()), state))
            Y2.append(expect(self.two_qubit_operator(qubit2_operator=self._tr2.sy()), state))
            Z2.append(expect(self.two_qubit_operator(qubit2_operator=self._tr2.sz()), state))

        fig, axes = subplots(1, 2, figsize=(15, 3))
        axes[0].plot(Ts, X1, label=r"$\langle x\rangle$")
        axes[0].plot(Ts, Y1, label=r"$\langle y\rangle$")
        axes[0].plot(Ts, (array(Z1) + 1) / 2, label=r"$\langle z\rangle$")
        axes[1].plot(Ts, X2)
        axes[1].plot(Ts, Y2)
        axes[1].plot(Ts, (array(Z2) + 1) / 2)
        for ax in axes:
            ax.grid()
        axes[0].legend()
        '''
