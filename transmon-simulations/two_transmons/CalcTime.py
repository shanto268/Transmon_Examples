import two_transmons.DoubleTransmonSystem
from two_transmons.DoubleTransmonSystem import *
import two_transmons.VacuumRabiSimulation
from two_transmons.VacuumRabiSimulation import *
import single_transmon.Transmon
from single_transmon.Transmon import *
from ReadoutResonator import *
from qutip import *
from multiprocessing import Pool

"""
params = {
    'duration': 20,
    'tanh_sigma': .1,
    "start": 10,
    "finish": 100,
    "phi_base_level": 0,
    'phi_offset': 0.4675,
    "t_points": 300,
    "frequency": 0.2
}
"""
durations = linspace(20,100,160)

def calc_time (farg):
    params=farg['params']
    phi=farg['phi']
    
    
    
    Nc = 7
    Ec1 = 0.25 * 2 * np.pi
    Ec2 = 1.03 * Ec1
    Ecc = 0.01 / 1.2 ** 2 * 2 * np.pi  # 1.2 is my estimate for n_{ge} matrix element
    Ej1 = Ec1 * 100
    Ej2 = Ec2 * 100
    d = 0.2

    T1_1, T2_1 = 10e3, 5e3
    T1_2, T2_2 = 5e2, 1e2

    tr1 = Transmon(Ec1, Ej1, d, 1 / T1_1, 1 / T2_1, Nc, 3, 1)
    tr2 = Transmon(Ec2, Ej2, d, 1 / T1_2, 1 / T2_2, Nc, 3, 2)
    dts = DoubleTransmonSystem(tr1, tr2, Ecc)
    r = ReadoutResonator(6, 1118, 1964, phi=0.29)
    r.set_qubit_parameters(.05, 0.05, 5, 5.3, .2, .2)
    ar1 = []
    ar2 = []
    for dur in durations:
        Ts = linspace(params['start'],params["finish"],params['t_points'])      
        params1 = params
        params1['phi_offset'] = phi
        params1['duration'] = dur
        VRS = VacuumRabiSimulation(dts, Ts, params1, r)
        result = VRS.run()
        state1 = VRS._rho0
        state2 = VRS._dts.e_state(1 / 2, 1 / 2, 2)
        state2 = state2 * state2.dag()
        
        ar1.append(expect(result.states[-1],state1))
        ar2.append(expect(result.states[-1],state2))
        
    ret = {'projections1': ar1, 'projections2': ar2}
    return ret

def calc_projections(farg):
#     phis=[]
#     params=farg[0]['params']
#     for i in range(len(farg)):
#         phis.append(farg[i]['phi'])
#     Nphi=len(phis)
    projections1 = []
    projections2 = []
    result=[]

    with Pool(4) as p:
        result = []
        for item in tqdm(p.imap(calc_time, farg)):
            result.append(item)
    #print (result)        
    for ind in result:     
        projections1.append(ind['projections1'])
        projections2.append(ind['projections2'])
    return (projections1, projections2)
def cook_farg(Nphi,freq):
    phis = linspace(0.41, 0.53, Nphi)
    params = {
        'duration': 26.65, #vaccuum rabi time, derived from simulation
        'tanh_sigma': .1,
        "start": 60,
        "finish": 200,
        "phi_base_level": 0,
        'phi_offset': 0.4664,
        "t_points": 10000,
        "frequency": freq
    }
    farg=[]

    for i in range (len(phis)):
        farg.append({'phi' : phis[i],'params' : params})
    return farg
       

