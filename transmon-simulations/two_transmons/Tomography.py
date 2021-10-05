from scipy import *
from qutip import *
from numpy import sqrt
from matplotlib.pyplot import *
from itertools import product
from scipy.linalg import cholesky
from tqdm import tqdm_notebook
from tqdm import tqdm
import p_tqdm
from p_tqdm import p_map
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib import cm
from scipy.optimize import *
import multiprocessing
from multiprocessing import Pool
from functools import partial
from numpy.linalg import norm
import numpy

from two_transmons.ZPulse import ZPulse

class Tomography:
    
    def __init__(self, dts, Ts, params, readout_resonator):
        self._dts = dts
        self._Ts = Ts
        self._params = params
        self._r = readout_resonator
        self._pi_duration = 33.45 # don't touch
        self._drive_amplitude = 0.01*2*pi/2*3 # don't touch
        
        
        E0 = dts.gg_state(0, 1/2, True)[1]
        E10 = dts.e_state(0, 1/2, 1, True)[1]
        E01 = dts.e_state(0, 1/2, 2, True)[1]
        self._freqs = ((E10-E0)/2/pi,(E01-E0)/2/pi) #  resonance frequencies of qubits 
        self._q1_rotations = [(0, 0),
                              (self._pi_duration, 0),
                              (self._pi_duration/2, 0),
                              (self._pi_duration/2, pi),
                              (self._pi_duration/2, pi/2),                              
                              (self._pi_duration/2, -pi/2)]
            
        self._q2_rotations = [(0, 0),
                              (self._pi_duration, 0),
                              (self._pi_duration/2, 0),
                              (self._pi_duration/2, pi),
                              (self._pi_duration/2, pi/2),                              
                              (self._pi_duration/2, -pi/2)]
        self._2q_rotations = list(product(self._q1_rotations, self._q2_rotations)) #time and angle representation
        self._H0 = dts.H_td_diag_approx(*self.build_waveforms()) # part of Hamiltonian without drive
        
        self._measurement_operators = []
        
        self._meas_op = self._dts.two_qubit_operator(qubit1_operator = 60*Qobj([[-1,0],[0,1]])) +\
        self._dts.two_qubit_operator(qubit2_operator = 50*Qobj([[-1,0],[0,1]])) +\
        40*self._dts.two_qubit_operator(qubit1_operator = Qobj([[-1,0],[0,1]]),qubit2_operator = Qobj([[-1,0],[0,1]]))
        rotations_1 = [Qobj([[1,0],[0,1]]),
            Qobj(array([[0,1j],[1j,0]])),
            Qobj(1/sqrt(2)*array([[1,1j],[1j,1]])),
            Qobj(1/sqrt(2)*array([[1,-1j],[-1j,1]])),           
            Qobj(1/sqrt(2)*array([[1,1],[-1,1]])),
            Qobj(1/sqrt(2)*array([[1,-1],[1,1]]))]
        rotations_2 = rotations_1
        self._rotations_matrix = [] #matrix representation of rotations
        for state1 in rotations_1:
            for state2 in rotations_2:
                self._rotations_matrix.append(self._dts.two_qubit_operator(state1,state2))
        for rotation in self._rotations_matrix:
            self._measurement_operators.append(rotation.dag()*self._meas_op*rotation)
        
        pauli_mat_1q = [basis(2,0)*basis(2,0).dag()+basis(2,1)*basis(2,1).dag(),
                        basis(2, 0) * basis(2, 1).dag() + basis(2, 1) * basis(2, 0).dag(),
                        -(-1j * basis(2, 0) * basis(2, 1).dag() + 1j * basis(2, 1) * basis(2, 0).dag()),
                        -(ket2dm(basis(2, 0)) - ket2dm(basis(2, 1)))]
        self._pauli_mat_2qubits = []               #basis of 2-qubit operators for process tomography
        for p1 in pauli_mat_1q:
            for p2 in pauli_mat_1q:
                self._pauli_mat_2qubits.append(tensor(p1,p2))
#        self._rho0 = dts.ee_state(0,1/2)
#        self._rho0 = 1/2*(dts.gg_state(0,1/2)+dts.e_state(0, 1/2,1)+
#                           dts.e_state(0, 1/2,2)+
#                          dts.ee_state(0, 1/2))
#        self._rho0 = dts.e_state(0, 1/2, 1)
#        self._rho0 = 1/sqrt(2)*(dts.e_state(0, 1/2, 1)*1j+dts.e_state(0, 1/2, 2))
        self._rho0 = dts.gg_state(0,1/2)
        self._rho0 = self._rho0*self._rho0.dag()
        self._rho_prepared_states = []
        for rotation in self._rotations_matrix:
            self._rho_prepared_states.append(rotation*self._rho0*rotation.dag()) #base states for maximum likelihood estimation
    def set_pi_duration (self, pi_duration):
        self._pi_duration = pi_duration
        self._q1_rotations = [(0, 0),
                              (self._pi_duration, 0),
                              (self._pi_duration/2, 0),
                              (self._pi_duration/2, pi),
                              (self._pi_duration/2, pi/2),                              
                              (self._pi_duration/2, -pi/2)]
            
        self._q2_rotations = [(0, 0),
                              (self._pi_duration, 0),
                              (self._pi_duration/2, 0),
                              (self._pi_duration/2, pi),
                              (self._pi_duration/2, pi/2),                              
                              (self._pi_duration/2, -pi/2)]
        self._2q_rotations = list(product(self._q1_rotations, self._q2_rotations))
    def set_freqs (self, freq1, freq2):
        self._freqs = (freq1, freq2,)
        
    def build_waveforms(self):
        waveform1 = ZPulse(self._Ts, self._params).waveform()
        #waveform1 = ones_like(self._Ts)*0
        waveform2 = ones_like(self._Ts)*1/2
        return waveform1, waveform2
        
    def run(self):
        self._options = Options(nsteps=20000, store_states=True)
        
        self._c_ops = self._dts.c_ops(0, 1/2)
        
        try: 
            self._results
        except AttributeError:
            self._results = []
        
       
        
        self._results = parallel_map(self._tomo_step_parallel, self._2q_rotations)
            
        return self._results

    def run_rabi (self, freqs, n_qubit):
        self._options = Options(nsteps=20000, store_states=True)
        
        self._c_ops = self._dts.c_ops(0, 1/2)
        
       
        self._results_rabi = []
        
        E0 = self._dts.gg_state(0, 1/2, True)[1]
        E10 = self._dts.e_state(0, 1/2, 1, True)[1]
        E01 = self._dts.e_state(0, 1/2, 2, True)[1]
        #self._freqs = ((E10 - E0)/2/pi, (E01 - E0)/2/pi)
        if (n_qubit == 1):
            self._results_rabi = self._tomo_step ([(self._pi_duration, 0),(0,0)], freqs)
        elif ( n_qubit == 2):
            self._results_rabi = self._tomo_step ([(0,0),(self._pi_duration, 0)], freqs)
        
        return self._results_rabi

    def run_iswap_check(self):
        self._options = Options(nsteps=20000, store_states=True)
        self._c_ops = self._dts.c_ops(0, 1/2)     
       
        self._results_iswap_check = []
        self._results_iswap_check = self._tomo_step([(0,0),(0,0)], self._freqs)
        return self._results_iswap_check

    def run_iswap (self, store_states = False):
        self._options = Options(nsteps=20000, store_states=False)
        self._c_ops = self._dts.c_ops(0, 1 / 2)
        self._results_iswap = []
      
     

        self._results_iswap = p_map(self.run_iswap_step_parallel, self._2q_rotations, [False]*len(self._2q_rotations))

    def run_iswap_step_parallel (self, rotations, full_output = False):
        dur1, dur2 = rotations[0][0], rotations[1][0]
        phi1, phi2 = rotations[0][1], rotations[1][1]
        H0 = self._dts.H_td_diag_approx(*self.build_waveforms()) #H0 - is the Hamiltonian of state preparing and our gate, after that we                                                                      #need tomography
        H0 += self._dts.Hdr([self._drive_amplitude] * 2,
                           (dur1, dur2),
                           (10,10),  # time of starting state preparation
                           (phi1, phi2),
                           self._freqs)
        density_matrix_simulated = []
        density_matrix_final_states = [] 
        for rotation in tqdm_notebook(self._2q_rotations):
            dur1_tomo, dur2_tomo = rotation[0][0], rotation[1][0]
            phi1_tomo, phi2_tomo = rotation[0][1], rotation[1][1]
            H = H0 + self._dts.Hdr([self._drive_amplitude] * 2,
                           (dur1_tomo, dur2_tomo),
                           # (self._params["duration"]+20, self._params["duration"]+20+self._pi_duration+10),
                           (10 + self._params["duration"]+self._params['start'], 10 +self._params["duration"] + self._params['start']), 
                           (phi1_tomo, phi2_tomo),
                           self._freqs)
            density_matrix = mesolve(H, self._rho0, self._Ts, c_ops = self._c_ops,progress_bar=None,options=self._options).states
            if (full_output):
                density_matrix_simulated.append(density_matrix)
            density_matrix_final_states.append(density_matrix[-1])
            del H
        
        if full_output:
            return (density_matrix_simulated, density_matrix_final_states)
        return density_matrix_final_states
            
    def _find_rho_iswap(self, averages, rho_final_simulated): #rho_final_simulated is an array of 
                                                               #density matrixes at the end of simulation, for tomography
        x0 = random.rand(16)
        for n in tqdm_notebook(range(averages), desc='Tomography iswap: Likelihood minimization', ncols=700):
            new = minimize (self._likelihood_rho_iswap, x0, args = (rho_final_simulated), method='BFGS', tol= 1e-09,\
                            options = {'gtol': 1e-05,'maxiter': 1000})
            if (n==0):
                best = new
            elif (new.fun < best.fun):
                best = new           
            x0 =best.x + 0.1*(random.rand(16)*2 - 1)
            print (best.fun, new.fun)            
        return self.x_to_rho(best.x) 
    
    
    def _likelihood_rho_iswap (self, x, rho_final_simulated): # calculation likelihood giving results of the system evolution
        rho = self.x_to_rho(x)     #with process parametrization x  
        lh = 0
        for ind, rho_true in enumerate(rho_final_simulated):
            meas_op = self._measurement_operators[ind]
            lh+=(expect(self._meas_op, rho_true) - expect(meas_op,rho))**2
        return abs(lh)
    
    def _operator_representation (self, chi, number): #gives part of sum which includes chi
        d = 4
        q_dim = 2
        operator = Qobj(numpy.zeros((4,4)),dims=[[q_dim, q_dim],[q_dim, q_dim]])
        pauli = self._pauli_mat_2qubits
        rho_initial = self._rho_prepared_states[number]
        for j in range (d**2):
            for k in range (d**2):
                operator+= chi[j,k]*pauli[j]*rho_initial*pauli[k]
        return operator
                
   
         
    def likelihood_iswap (self, x, density_matrix_after_gate): #density matrix array - is an array of resulte_dm (lambda) after tomography
        chi = Tomography.x_to_chi(x)
        lh = 0
        for ind, rho_initial in enumerate (self._prepared_states):
            rho_final = density_matrix_after_gate[ind]
            delta_rho = rho_final - self._operator_representation(chi,ind)
            lh += norm(delta_rho, ord = 'fro')
        return lh
    
    
    def process_tomo(self, averages): #start only after run_iswap
        density_matrix_after_gate = []
        #for ind, rho_initial in enumerate(self._rho_prepared_states):
            #density_matrix_after_gate.append(self._find_rho_iswap(3,self._results_iswap[ind])) # можно распараллелить
        find_rho_partial = partialmethod(self._find_rho_iswap,3)
        density_matix_after_gate = p_map(self.find_rho_partial, self._results_iswap)
        x0 = random.rand(256)
        for n in tqdm_notebook(range(averages), desc='Tomography iswap_operator: Likelihood minimization', ncols=700):
            new = minimize (self._likelihood_iswap, x0, args = (density_matrix_after_gate), method='BFGS', tol= 1e-09,\
                            options = {'gtol': 1e-05,'maxiter': 1000})
            if (n==0):
                best = new
            elif (new.fun < best.fun):
                best = new           
            x0 =best.x + 0.1*(random.rand(16)*2 - 1)
            print (best.fun, new.fun)            
        return self.x_to_chi(best.x) 
            

    def _tomo_step(self, rotations, freqs):
        dur1, dur2 = rotations[0][0], rotations[1][0]
        phi1, phi2 = rotations[0][1], rotations[1][1]

        self._H = self._H0
        self._H += self._dts.Hdr([self._drive_amplitude]*2, 
                                (dur1, dur2), 
                                #(self._params["duration"]+20, self._params["duration"]+20+self._pi_duration+10),
                                (10 + self._params["duration"]+self._params['start'], 10+ self._params["duration"] + self._params['start']),                                 (phi1, phi2),
                                freqs)         #part of drive for Tomography
        #self._H += self._dts.Hdr_cont([self._drive_amplitude]*2)

        return mesolve(self._H, self._rho0, 
                       self._Ts, c_ops = self._c_ops, 
                       progress_bar=None, 
                       options=self._options)
    def _tomo_step_parallel(self, rotations):
        dur1, dur2 = rotations[0][0], rotations[1][0]
        phi1, phi2 = rotations[0][1], rotations[1][1]

        H = self._dts.H_td_diag_approx(*self.build_waveforms())
        H += self._dts.Hdr([self._drive_amplitude]*2,
                                (dur1, dur2), 
                                #(self._params["duration"]+20, self._params["duration"]+20+self._pi_duration+10),
                                (10 + self._params["duration"]+self._params['start'], 10 +self._params["duration"] + self._params['start']),                                 (phi1, phi2),
                                self._freqs)
        #self._H += self._dts.Hdr_cont([self._drive_amplitude]*2)

        return mesolve(H, self._rho0,
                       self._Ts, c_ops = self._c_ops, 
                       progress_bar=None, 
                       options=self._options)
    
                       
    def _joint_expect(self, state, f, chi1, chi2):
        return expect(state, self._r.measurement_operator(f, chi1, chi2))
    
    def visualize_joint_readout(self, f, chi1, chi2):
        
        fig, axes = subplots(1, 2, figsize=(15,5))
        for result in self._results:
            data = array(serial_map(self._joint_expect, result.states, task_args = (f, chi1, chi2)))
            axes[0].plot(self._Ts, data.real)
            axes[1].plot(self._Ts, data.imag)
            
    def build_2qubit_operator (self, q1_ax, q2_ax):
        q_ax =[q1_ax,q2_ax]
        q_ans = [Qobj(identity(2)),Qobj(identity(2))]
        
        for i in range (2):
            if q_ax[i]=='X':
                q_ans[i] = basis(2, 0) * basis(2, 1).dag() + basis(2, 1) * basis(2, 0).dag()
            if q_ax[i]=='Y':
                q_ans[i] = -(-1j * basis(2, 0) * basis(2, 1).dag() + 1j * basis(2, 1) * basis(2, 0).dag())
            if q_ax[i]=='Z': 
                q_ans[i] = -(ket2dm(basis(2, 0)) - ket2dm(basis(2, 1)))
        
        
        return tensor(*q_ans)
    
    
    def plot_qubits_dynamic (T, n, dm = None):   #start only after run function
        Ts = T._Ts
        if dm is not None:
            density_matrix = dm
        else:
            density_matrix = T._results[n].states
        H0 = T._H0[0][0]*T._H0[0][1][0]+T._H0[1][0]*T._H0[1][1][0] + T._H0[2] # constant part of Hamiltonian
        H0_list_expm = empty_like(Ts,dtype = Qobj)
        H0_list_expm_conj = empty_like(Ts, dtype = Qobj)
        dm_interaction = empty_like(Ts, dtype = Qobj)
        for ind, t in enumerate (Ts):
            H_current = H0*t*1j
            matrix_exp = H_current.expm()
            H0_list_expm [ind] = matrix_exp
            H0_list_expm_conj[ind] = matrix_exp.dag()
            dm_interaction[ind] = matrix_exp*density_matrix[ind]*matrix_exp.dag()

        qubit1_z_interaction = expect(T.build_2qubit_operator('Z','id'), dm_interaction)
        qubit1_y_interaction = expect(T.build_2qubit_operator('Y','id'), dm_interaction)
        qubit1_x_interaction = expect(T.build_2qubit_operator('X','id'), dm_interaction)
        qubit2_z_interaction = expect(T.build_2qubit_operator('id','Z'), dm_interaction)
        qubit2_y_interaction = expect(T.build_2qubit_operator('id','Y'), dm_interaction)
        qubit2_x_interaction = expect(T.build_2qubit_operator('id','X'), dm_interaction)
        
        fig = figure(figsize=(14,7))
        axes = fig.add_axes([0.05, 0.2, 0.4, 0.8], projection="3d")
        axes1 = fig.add_axes([0.55, 0.2, 0.4, 0.8], projection="3d")


        sph = Bloch(fig=fig, axes=axes)
        sph.clear()
        sph.sphere_alpha = 0
        sph.zlabel = [r'$\left|e\rightangle\right.$', r"$\left|g\rightangle\right.$"]
        sph.xlpos = [1.3, -1.3]
        sph.xlabel = [r'$\left.|+\right\rangle$', r"$\left.|-\right\rangle$"]
        sph.ylpos = [1.2, -1.3]
        sph.ylabel = [r'$\left.|+_i\right\rangle$', r"$\left.|-_i\right\rangle$"]
        sph.xlpos = [1.3, -1.3]
        nrm=matplotlib.colors.Normalize(0,Ts[-1])
        colors=cm.RdBu_r(nrm(Ts))
        sph.point_size=[40]
        sph.point_color = list(colors)
        sph.point_marker=['.']
        sph.add_points([qubit1_x_interaction, qubit1_y_interaction, qubit1_z_interaction], meth='m')
        #sph.add_points ([[1/sqrt(2),1,0],[1/sqrt(2),0,0],[0,0,1]])  
        sph.render(fig,axes)

        sph1 = Bloch(fig=fig, axes=axes1)
        sph1.clear()
        sph1.sphere_alpha = 0
        sph1.zlabel = [r'$\left|e\rightangle\right.$', r"$\left|g\rightangle\right.$"]
        sph1.xlpos = [1.3, -1.3]
        sph1.xlabel = [r'$\left.|+\right\rangle$', r"$\left.|-\right\rangle$"]
        sph1.ylpos = [1.2, -1.3]
        sph1.ylabel = [r'$\left.|+_i\right\rangle$', r"$\left.|-_i\right\rangle$"]
        sph1.xlpos = [1.3, -1.3]
        sph1.point_size=[40]
        sph1.point_color = list(colors)
        sph1.point_marker=['.']
        sph1.add_points([qubit2_x_interaction, qubit2_y_interaction, qubit2_z_interaction], meth='m')
        #sph.add_points ([[1/sqrt(2),1,0],[1/sqrt(2),0,0],[0,0,1]])  
        sph1.render(fig,axes)


        m = cm.ScalarMappable(cmap=cm.RdBu_r, norm=nrm)
        m.set_array(Ts)
        m.set_clim(0, Ts[-1])
        position=fig.add_axes([0.05,0.15,0.4,0.03])
        cb = fig.colorbar(m, orientation='horizontal', cax=position)
        cb.set_label("Time, ns")
        cb.set_ticks(np.linspace(0,round(Ts[-1]),6))
        sph.make_sphere()

        m1 = cm.ScalarMappable(cmap=cm.RdBu_r, norm=nrm)
        m1.set_array(Ts)
        m1.set_clim(0, Ts[-1])
        position1=fig.add_axes([0.55,0.15,0.4,0.03])
        cb1 = fig.colorbar(m1, orientation='horizontal', cax=position1)
        cb1.set_label("Time, ns")
        cb1.set_ticks(np.linspace(0,round(Ts[-1]),6))
        sph1.make_sphere()
        axes.text2D(0.3,1, "Qubit 1 state evolution",transform=axes.transAxes, fontsize = 20)
        axes1.text2D(0.9,1, "Qubit 2 state evolution",transform=axes.transAxes, fontsize = 20)  
        return fig
    
    
    def show_density_matrix (density_matrix):
        density_matrix_re = real (density_matrix)
        density_matrix_im = imag (density_matrix)
        fig = figure (figsize = (12,9) )
        ax1=fig.add_subplot(121, projection = '3d')
        ax2=fig.add_subplot(122, projection = '3d')
        xlabels = ['00','01','10','11']
        ylabels = ['00','01','10','11']
        matrix_histogram (density_matrix_re, xlabels, ylabels, (r'$Re\ \rho$'), limits = [-1,1], fig = fig, ax = ax1)
        matrix_histogram (density_matrix_im, xlabels, ylabels, (r'$Im\ \rho$'), limits = [-1,1], fig = fig, ax = ax2)
        ax1.view_init(azim = -30, elev = 40)
        ax2.view_init(azim = -30, elev = 40)
        return fig
        
    
 
    def likelihood (self, x, n):                    # Вычисление Likelihood по загруженным измерениям \
        rho = self.x_to_rho(x)                      # для матрицы плотности заданной через x
        lh = 0
        rho_true = self._results[n].states[-1]      #replace to number of state
        for meas_op_result in self._measurement_operators:
            lh+=(expect(meas_op_result, rho_true) - expect(meas_op_result,rho))**2
        return abs(lh)
    
    def find_rho (self, averages, number): # number - number of rotation
        
        x0 = random.rand(16)
        for n in tqdm_notebook(range(averages), desc='Tomography: Likelihood minimization', ncols=700):
            new = minimize (self.likelihood, x0, args = (number), method='BFGS', tol= 1e-09, options = {'gtol': 1e-05,\
                                                                                                    'maxiter': 1000})
            if (n==0):
                best = new
            elif (new.fun < best.fun):
                best = new           
            x0 =best.x + 0.1*(random.rand(16)*2 - 1)
            print (best.fun, new.fun)
            
        return self.x_to_rho(best.x)
    
    def find_rho_all_parallel (self, averages, x_start = random.rand(16)):
        with Pool(4) as p:
            self.rho_tomo = []
            find_rho_avg = partial(Tomography.find_rho, self, averages)
            for item in tqdm(p.imap(find_rho_avg, range(len(self._2q_rotations)))):
                self.rho_tomo.append(item)
        
    def find_rho_all (self, averages, x_start = random.rand(16)):
        self.rho_tomo = []
        for number in tqdm(range(len(self._2q_rotations))):
            self.rho_tomo.append(self.find_rho(averages, number))
   
    @staticmethod
    def x_to_rho(x):                                # Density matrix parametrization via Choletsky decomposition
        dim = int(sqrt(len(x)))
        t = np.identity(dim, complex)
        for i in range(dim):
            t[i, i] = abs(x[i])
        k = dim
        for i in range(dim):
            for l in range(i):
                t[i, l] = x[k] + 1j * x[k + 1]
                k += 2
        q_dim = 2
        L = Qobj(t, dims=[[q_dim, q_dim],[q_dim, q_dim]])
        L1 = L.full()
        L2 = L.dag().full()
        L12=dot(L1,L2)
        rho = Qobj(L12,dims=[[q_dim, q_dim],[q_dim, q_dim]])
        rho = rho / rho.tr()
        return rho

    @staticmethod
    def rho_to_x(rho):
        if rho is None:
            return None
        rho = rho*rho.dag()
        ps_rho = rho + 1e-12                                            # wtf KOSTYL'
        L_ar = (Qobj(cholesky(ps_rho.full() ))).dag().full()             # dont touch
        L_dim = L_ar.shape[0]
        x = np.zeros(L_dim ** 2)
        k = L_dim
        for i in range(L_dim):
            x[i] = L_ar[i, i]
            for j in range(i):
                x[k] = real(L_ar[i, j])
                x[k + 1] = imag(L_ar[i, j])
                k += 2
        return x         
    @staticmethod
    def x_to_chi(x):
        dim = int(sqrt(len(x)))
        t = np.identity(dim, complex)
        for i in range(dim):
            t[i, i] = abs(x[i])
        k = dim
        for i in range(dim):
            for l in range(i):
                t[i, l] = x[k] + 1j * x[k + 1]
                k += 2
        q_dim = 4
        L = Qobj(t, dims=[[q_dim, q_dim],[q_dim, q_dim]])
        L1 = L.full()
        L2 = L.dag().full()
        L12=dot(L1,L2)
        chi = Qobj(L12)
        return chi 
    def chi_to_R (self, chi):
        R = numpy.zeros_like(chi)
        d = 4
        pauli = self._pauli_mat_2qubits
        for i in range(d**2):
            for j in range (d**2):
                for k in range (d**2):
                    for l in range (d**2):
                        R[i,j]+=chi[k,l]* numpy.trace((pauli[i]*pauli[k]*pauli[j]*pauli[l]).full)
                R[i][j]/=d        
        
    def Fidelity (self, R):
        fidelity = 1.0/20*numpy.trace(dot(R.H,R) + 4*identity(len(R[0])))
        return fidelity
        
        
        
