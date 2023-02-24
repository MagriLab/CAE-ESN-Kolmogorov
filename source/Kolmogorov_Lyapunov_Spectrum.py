import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"
import h5py
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from kolsol.torch.solver import KolSol


def QR(M, N_exp):
    ''' Compute an orthogonal basis, Q, and the change in the norm along element of the basis, S. '''

    for i in range(N_exp):
        M[i]    = M[i].flatten()

    Q    = [None]*N_exp
    S    = np.empty(N_exp)

    S[0] = np.linalg.norm(M[0])
    
    Q[0] = M[0] / S[0]

    for i in range(1,N_exp):

        temp = 0
        for j in range(i):
            temp += np.dot(Q[j],M[i])*Q[j]   

        Q[i]  = M[i] - temp
        S[i]  = np.linalg.norm(Q[i])
        Q[i] /= S[i] 


    for i in range(N_exp):
        Q[i]  = Q[i].reshape(Phys_size1,Phys_size1,2)

    return Q, np.log(S)

def RK4(q0,dt,N,func):
    ''' 4th order RK for autonomous systems described by func '''

    for i in range(N):

        k1   = dt * func(q0)
        k2   = dt * func(q0 + k1/2)
        k3   = dt * func(q0 + k2/2)
        k4   = dt * func(q0 + k3)

        q0   = q0 + (k1 + 2*k2 + 2*k3 + k4)/6

    return  q0

def FE(q0,dt,N,func):
    ''' Forward Euler method'''

    for i in range(N):
        q0   = q0 + dt * func(q0)

    return  q0


Re = 34.
nk = 32
Phys_size1 = 2*nk+1  #size of the physical grid (has to be bigger than 2*nk)


ks = KolSol(nk=nk, nf=4, re=Re, ndim=2, device=DEVICE)


# Integration parameters
N_exp     = 15            #Number of exponents to compute
dt        = .01           #timestep
N_norm    = 10            #number of steps before orthonormalization of the basis
N         = int(20000/dt) #total length of time series 
N_times   = N//N_norm

in_skip   = 300

integr    = RK4

eps       = 1e-2

u_hat = ks.random_field(magnitude=10.0, sigma=2.0, k_offset=[0, 3]) #random initial condition

t_skip = 100
N_skip = int(t_skip/dt)

print('Computing transient')

x1P = RK4(u_hat,dt,N_skip,ks.dynamics)            #initial condition on the attractor (after transient to reach it)
x_tP1 = ks.fourier_to_phys(x1P,Phys_size1)        #unperturbed initial condition in physical space

print(np.linalg.norm(x_tP1.cpu().numpy()), dt, Re, nk)


# initializing quantities
S         = 0
SS        = np.zeros((N_times,N_exp))
x0P       = []

for ii in range(N_exp):
    xinn     = np.random.rand(Phys_size1,Phys_size1,2)
    xinn    /= np.linalg.norm(xinn)
    x0P     += [ks.phys_to_fourier(x_tP1 + torch.tensor(xinn*eps, device=DEVICE))]        #perturbed initial condition in physical space

#N_times is the number of orthonormalizations
for jj in range(N_times):
    
    if jj > 0:
        x0P = [ks.phys_to_fourier(x_tP1+torch.tensor(xx*eps, device=DEVICE)) for xx in aa]  #perturbed initial condition with orthonormal basis

    x0P   = [integr(xx,dt,N_norm,ks.dynamics) for xx in x0P]            #computing perturbed trajectories in Fourier domain
    X_P   = [ks.fourier_to_phys(xx,Phys_size1) for xx in x0P]

    x1P   = integr(x1P,dt,N_norm,ks.dynamics)                           #computing unperturbed trajectory in Fourier domain    
    x_tP1 = ks.fourier_to_phys(x1P,Phys_size1)

            
    b  = x_tP1.cpu().numpy() 
    a  = [(xx.cpu().numpy() - b)/eps for xx in X_P]                     #perturbation as difference between trajectories

    aa, S1 = QR(a, N_exp)
    
    if jj > in_skip:
        S      += S1
        SS[jj]  = S/((jj-in_skip)*dt*N_norm)
        if jj%(N_times//200) == 0: print('Lyapunov exponents:', SS[jj], jj/N_times)   


fln = 'Lyap_exp_Re' + str(Re) + '_dt=' +str(dt)+'_nk' + str(nk)+'.h5'
hf = h5py.File(fln,'w')
hf.create_dataset('Lyap_conv',data=SS)
hf.close()

np.savetxt('Lyap_exp_Re' + str(Re) + '_dt=' +str(dt)+'_nk' + str(nk)+'.csv', 
           SS[-1],
           delimiter =", ", 
           fmt ='% s')  

