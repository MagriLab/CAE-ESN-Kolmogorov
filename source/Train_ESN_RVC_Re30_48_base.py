#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = '4' # imposes only one core
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sparse_dot_mkl
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
import matplotlib as mpl
from scipy.io import loadmat
import time
from scipy.io import savemat
from scipy.stats import wasserstein_distance, entropy
from scipy import signal
from scipy.linalg import solve
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
plt.style.use('dark_background')
#Latex
mpl.rc('text', usetex = True)
mpl.rc('font', family = 'serif')
# In val_functions there are the validation strategies
exec(open("Val_Functions.py").read())
exec(open("Functions.py").read())



################## For a TUTORIAL on ESNs -- with similar implementation -- check https://github.com/alberacca/Echo-State-Networks


upsample_0 = 1
upsample   = 10
data_len   = 300000
transient  = 10000


Nx = 48
Nu = 2
dt       = 0.1*upsample


ker_size   = [(5,5), (3,3), (7,7)]             #kernel size
N_latent   = 9
N_layer    = 3
N_dim      = N_latent
act        = 'tanh'
#obtained by last cells of Autoencoder.ipynb
fln  = '/rds/user/ar994/hpc-work/48_Encoded_data_Re30_base_double_' + str(ker_size)         + '_'+str(N_latent) + '_' + str(N_layer) + '_' +  act +  '.h5'
hf  = h5py.File(fln,'r')
U     = np.array(hf.get('U_enc'))[::upsample]
hf.close()
N_latent = N_dim

shape = U.shape
print(shape)
U = U.reshape(shape[0], shape[1]*shape[2]*shape[3])[:, :N_dim]

noises   = np.array([0])
data_std = np.std(U,axis=0)

sigma_ph     = np.sqrt(np.mean(np.linalg.norm(U, axis=1)**2))
threshold_ph = 0.2


NN_units  = [2000]     #number of units
NN_train  = [500,1000] #number of training points

for N_units in NN_units:

    for N_trains in NN_train:

        # number of time steps for washout, train, validation, test
        N_washout = 100
        N_train   = int(N_trains/dt)
        N_val     = int(500/dt)

        print(N_train, N_val)
        print(N_val/N_lyap)
        print('data %:',(N_train)/(U.shape[0]))

        #compute norm
        U_data = U[:N_washout+N_train+N_val]
        m = U_data.min(axis=0)
        M = U_data.max(axis=0)
        norm = M-m

        # washout
        U_washout = U[:N_washout]
        # training
        U_t   = U[N_washout:N_washout+N_train-1]
        Y_t   = U[N_washout+1:N_washout+N_train]
        # training + validation
        U_tv  = U[N_washout:N_washout+N_train-1].copy()


        Y_tv  = np.zeros((len(noises), N_train-1, N_latent))
        for jj in range(noises.size):
            for i in range(N_latent):
                    Y_tv[jj,:,i] = U[N_washout+1:N_washout+N_train,i].copy()
        Y_tv = Y_tv.astype('float32')
        U_tv = U_tv.astype('float32')
        # validation
        Y_v  = U[N_washout+N_train:N_washout+N_train+N_val]



    # ### ESN Initiliazation Hyperparameters
    # 
    # To generate the Echo State Networks realizations we set the hyperparameters (even the ones that are optimized during validation)


        bias_in   = np.array([0.1], dtype='float32') #input bias
        bias_out  = np.array([1.], dtype='float32') #output bias 

        # N_units      = 1000
        dim          = U.shape[1] # dimension of inputs (and outputs) 
        connectivity = 3 #N_latent
        n_inp        = 1
        sparseness   = 1 - connectivity/(N_units-1) 
        print(N_units, connectivity)

        load    = False
        tikh = np.array([1e-3,1e-6,1e-9])  # Tikhonov factor


        # ###  Grid Search and Bayesian Optimization
        # 
        # Here we set the parameters for Grid Search and Bayesian Optimization.

        n_in  = 0    #Number of Initial random points

        spec_in     = .8  #np.log10(.1)   #range for hyperparameters (spectral radius and input scaling)
        spec_end    = 1.2 #np.log10(1.)  
        in_scal_in  = round(np.log10(0.01),3)
        in_scal_end = round(np.log10(5.0),3)

        # In case we want to start from a grid_search, the first n_grid^2 points are from grid search
        # if n_grid^2 = n_tot then it is pure grid search
        n_grid_x = 5  # (with n_grid**2 < n_tot you get Bayesian Optimization)
        n_grid_y = 5
        n_tot    = n_grid_x*n_grid_y + 5   #Total Number of Function Evaluatuions


        # computing the points in the grid
        if n_grid_x > 0:
            x1    = [[None] * 2 for i in range(n_grid_x*n_grid_y)]
            k     = 0
            for i in range(n_grid_x):
                for j in range(n_grid_y):
                    x1[k] = [spec_in + (spec_end - spec_in)/(n_grid_x-1)*i,
                             in_scal_in + (in_scal_end - in_scal_in)/(n_grid_y-1)*j]
                    k   += 1

        # range for hyperparameters
        search_space = [Real(spec_in, spec_end, name='spectral_radius'),
                        Real(in_scal_in, in_scal_end, name='input_scaling')]

        # ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
        kernell = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0))*Matern(length_scale=[0.2,0.2], nu=2.5, length_scale_bounds=(1e-2, 1e1)) 


        #Hyperparameter Optimization using either Grid Search or Bayesian Optimization
        def g(val):
            
            #Gaussian Process reconstruction
            b_e = GPR(kernel = kernell,
                    normalize_y = True, #if true mean assumed to be equal to the average of the obj function data, otherwise =0
                    n_restarts_optimizer = 3,  #number of random starts to find the gaussian process hyperparameters
                    noise = 1e-10, # only for numerical stability
                    random_state = 10) # seed
            
            
            #Bayesian Optimization
            res = skopt.gp_minimize(val,                         # the function to minimize
                              search_space,                      # the bounds on each dimension of x
                              base_estimator       = b_e,        # GP kernel
                              acq_func             = "EI",       # the acquisition function
                              n_calls              = n_tot,      # total number of evaluations of f
                              x0                   = x1,         # Initial grid search points to be evaluated at
                              n_random_starts      = n_in,       # the number of additional random initialization points
                              n_restarts_optimizer = 3,          # number of tries for each acquisition
                              random_state         = 10,         # seed
                                   )   
            return res


        # ### Validate Echo State
        # Select validation function to select the hyperparameters for each realization in the ensemble of networks


        #Number of Networks in the ensemble
        ensemble = 10
        # Which validation strategy (implemented in Val_Functions.ipynb)
        val      = RVC_Noise
        N_fo     = 30  # number of folds
        N_in     = N_washout  # interval before the first fold
        N_fw     = (N_train-1-N_val)//(N_fo-1) # how many steps forward the validation interval is shifted (in this way they are evenly spaced)
        N_splits = 4

        #How many intervals in the test set
        N_test   = 30
        N_tstart = N_washout + N_train
        k_t      = 1.
        N_intt   = int(k_t*N_val)
        N_fw     = N_intt//2

        if N_units > 10000:
            step = step_multi
        else:
            step = step_single

        minimum       = np.zeros((ensemble, 4))
        Woutt         = np.zeros(((ensemble, N_units+1,dim)))
        Winn          = []
        Ws            = []

        for j in range(ensemble):

            #generate input and state matrices
            seed= j+1
            rnd = np.random.RandomState(seed)

            Win  = lil_matrix((N_units,dim+1)) 
            for jj in range(N_units):
                Win[jj,rnd.choice(dim+1, size=n_inp, replace=False)] = rnd.uniform(-1, 1)
            Win = Win.tocsr()

            W = csr_matrix(
                rnd.uniform(-1, 1, (N_units, N_units)) * (rnd.rand(N_units, N_units) < (1-sparseness)))
            spectral_radius2 = np.abs(sparse_eigs(W, k=1, which='LR', return_eigenvectors=False,maxiter=N_units*50,tol=1e-6))[0]
            W = (1/spectral_radius2)*W #scaled to have unitary spec radius

            #initialize values
            minn     = []
            sigma_ns = []
            hypers   = []
            non_max  = True
            kk       = 0
            ti       = time.time()
            tikh_opt = np.zeros(n_tot)
            noise_opt= np.zeros(n_tot)
            sign     = 1
            sigma_n  = 0.001 #first value of noise

            max_iterations = 4
            U_tvv     = np.empty((max_iterations, N_train-1, N_latent))

            while non_max:
                
                # training set with sigma_n noise
                k    = 0
                seed = 0                      
                rnd1  = np.random.RandomState(seed)
                for i in range(N_latent):
                        U_tvv[kk,:,i] = U[N_washout:N_washout+N_train-1,i].copy() \
                                        + rnd1.normal(0, sigma_n*data_std[i], N_train-1)

                U_tv = U_tvv[kk].copy()
                
                # run validation
                res         = g(val)
                
                # save results
                sigma_ns   += [sigma_n]
                sigma_n     = sigma_n*(np.sqrt(10)**(sign))
                        
                #compute performance in the test set (to select best performing sigma_n)
                rho, sigma_in = res.x
                tikk          = tikh_opt[np.argmin(np.array(res.func_vals))]
                hypers     += [ [np.append(res.x,[tikk])] ]
                sigma_in      = round(10**sigma_in,3)
                Wout          = train_save_n(U_washout, U_tv, U[N_washout+1:N_washout+N_train],
                                      tikk,sigma_in,rho,0)
                
                Errors   = np.zeros(N_test)
                
                #Different Folds in the test set
                for i in range(N_test):
                    
                    # data for washout and target in each interval
                    U_wash    = U[N_tstart - N_washout +i*N_fw : N_tstart + i*N_fw].copy()
                    Y_t       = U[N_tstart + i*N_fw            : N_tstart + i*N_fw + N_intt].copy() 
                            
                    #washout for each interval
                    Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
                    xa1     = Xa1[-1].copy()
                    
                    # Mean Square Error
                    Yh_t        = closed_loop(N_intt-1, xa1, Wout, sigma_in, rho)[0]
                    Errors[i]   = np.log10(np.mean((Yh_t-Y_t)**2))

                print('Errors', np.mean(Errors), U_tv[0,:3])
                minn           += [np.mean(Errors)]
                
                # check wether to icrease/decrease noise
                if kk == 1 and minn[1] > minn[0]:
                    sigma_n   = sigma_n/(np.sqrt(10)**(3))
                    sign      = -1
                
                argg = np.argsort(sigma_ns)
                min1 = np.array(minn)
                
                # check whether the max number of iterations or convergence has been reached
                if np.argmin(min1[argg]) != 0 and np.argmin(min1[argg]) != kk:
                    non_max = False
                
                if kk == max_iterations - 1 : non_max = False
                    
                kk +=1

            # save optimal noise

            min_arg       = np.argmin(min1)
            minimum[j,:3] = hypers[min_arg][0]

            seed         = 0                        #to be able to recreate the data
            rnd1         = np.random.RandomState(seed)
            sigma_n      = sigma_ns[min_arg] 
            minimum[j,3] = sigma_n*1
                    
            Woutt[j]   = train_save_n(U_washout, U_tvv[min_arg], U[N_washout+1:N_washout+N_train],
                                      minimum[j,2],10**minimum[j,1], minimum[j,0], 0)
            Winn      += [Win]
            Ws        += [W]


        if not load:
            fln = './data/48_ESN_Re30_enc_base_double_' + str(N_units) + '_' +str(N_latent)+'_'+str(N_trains)+'.mat'
            with open(fln,'wb') as f:  # need 'wb' in Python3
                savemat(f, {"norm": norm})
                savemat(f, {'hyperparameters':10**minimum[:,:]})
                savemat(f, {"Win": Winn})
                savemat(f, {'W': Ws})
                savemat(f, {"Wout": Woutt})
            print(fln)