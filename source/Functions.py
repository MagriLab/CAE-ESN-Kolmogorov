#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py


# ## ODE-side stuff

# In[4]:


class PerPad2D(tf.keras.layers.Layer):
    """
    Periodic Padding layer
    """
    def __init__(self, padding=1, asym=False, **kwargs):
        self.padding = padding
        self.asym    = asym
        super(PerPad2D, self).__init__(**kwargs)
        
    def get_config(self): #needed to be able to save and load the model with this layer
        config = super(PerPad2D, self).get_config()
        config.update({
            'padding': self.padding,
            'asym': self.asym,
        })
        return config

    def call(self, x):
        return periodic_padding(x, self.padding, self.asym)

def periodic_padding(image, padding=1, asym=False):
    '''
    Create a periodic padding (same of np.pad('wrap')) around the image, 
    to mimic periodic boundary conditions.
    When asym=True on the right and lower edges an additional column/row is added
    '''
        
    if asym:
        lower_pad = image[:,:padding+1,:]
    else:
        lower_pad = image[:,:padding,:]
    
    if padding != 0:
        upper_pad     = image[:,-padding:,:]
        partial_image = tf.concat([upper_pad, image, lower_pad], axis=1)
    else:
        partial_image = tf.concat([image, lower_pad], axis=1)
        
    if asym:
        right_pad = partial_image[:,:,:padding+1] 
    else:
        right_pad = partial_image[:,:,:padding]
    
    if padding != 0:
        left_pad = partial_image[:,:,-padding:]
        padded_image = tf.concat([left_pad, partial_image, right_pad], axis=2)
    else:
        padded_image = tf.concat([partial_image, right_pad], axis=2)

    return padded_image

def matsmul(x):
    return np.dot(x.T, x)


# In[1]:


## ESN with bias architecture

def step_single(x_pre, u, sigma_in, rho):
    """ Advances one ESN time step.
        Args:
            x_pre: reservoir state
            u: input
        Returns:
            new augmented state (new state with bias_out appended)
    """
    # input is normalized and input bias added
    u_augmented = np.hstack((u/norm, bias_in))
    # hyperparameters are explicit here
    x_post      = np.tanh(Win.dot(u_augmented*sigma_in) + W.dot(rho*x_pre)) 
    # output bias added
    x_augmented = np.concatenate((x_post, bias_out))
    
    return x_augmented

def step_multi(x_pre, u, sigma_in, rho):
    """ Advances one ESN time step.
        Args:
            x_pre: reservoir state
            u: input
        Returns:
            new augmented state (new state with bias_out appended)
    """
    # input is normalized and input bias added
    u_augmented = np.hstack((u/norm, bias_in))
    # hyperparameters are explicit here
    x_post      = np.tanh(sparse_dot_mkl.dot_product_mkl(Win,u_augmented*sigma_in) + \
                                    sparse_dot_mkl.dot_product_mkl(W, rho*x_pre))
    # output bias added
    x_augmented = np.concatenate((x_post, bias_out))
    
    return x_augmented

def open_loop(U, x0, sigma_in, rho):
    """ Advances ESN in open-loop.
        Args:
            U: input time series
            x0: initial reservoir state
        Returns:
            time series of augmented reservoir states
    """
    N     = U.shape[0]
    Xa    = np.empty((N+1, N_units+1))
    Xa[0] = np.concatenate((x0,bias_out))#, 0*U[0]/norm))
#     tt = time.time()
    for i in np.arange(1,N+1):
        Xa[i] = step(Xa[i-1,:N_units], U[i-1], sigma_in, rho)
#     print('open_loop:', (time.time()-tt)/N)

    return Xa

def closed_loop(N, x0, Wout, sigma_in, rho):
    """ Advances ESN in closed-loop.
        Args:
            N: number of time steps
            x0: initial reservoir state
            Wout: output matrix
        Returns:
            time series of prediction
            final augmented reservoir state
    """
    xa = x0.copy()
    Yh = np.empty((N+1, dim))
    Yh[0] = np.dot(xa, Wout)
    for i in np.arange(1,N+1):
        xa    = step(xa[:N_units], Yh[i-1], sigma_in, rho)
        Yh[i] = np.dot(xa, Wout) #np.linalg.multi_dot([xa, Wout]) 

    return Yh, xa

def train_n(U_washout, U_train, Y_train, tikh, sigma_in, rho):
    """ Trains ESN.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """
        
    ## washout phase
    xf = open_loop(U_washout, np.zeros(N_units), sigma_in, rho)[-1,:N_units]
    
    ## splitting training in N_splits to save memory
    LHS = 0
    RHS = 0
#     N_splits = 4
    N_len = (U_train.shape[0]-1)//N_splits
    
    for i in range(N_splits):
        t1  = time.time()
        ## open-loop train phase
        Xa1 = open_loop(U_train[i*N_len:(i+1)*N_len], xf, sigma_in, rho)[1:]
        xf  = Xa1[-1,:N_units].copy()
        if i == 0 and k==0: print('open_loop time:', (time.time()-t1)*N_splits)

        t1  = time.time()   
        LHS += np.dot(Xa1.T, Xa1) 
        RHS += np.dot(Xa1.T, Y_train[0,i*N_len:(i+1)*N_len])
        if i == 0 and k==0: print('matrix multiplication time:', (time.time()-t1)*N_splits)
    
    if N_splits > 1:# to cover the last part of the data that didn't make into the even splits
        Xa1 = open_loop(U_train[(i+1)*N_len:], xf, sigma_in, rho)[1:]
        LHS += np.dot(Xa1.T, Xa1) 
        RHS += np.dot(Xa1.T, Y_train[0,(i+1)*N_len:])
    
    Wout = np.empty((len(tikh),len(noises),N_units+1,dim))
    for jj in range(len(noises)):
        for j in range(len(tikh)):
            t1   = time.time()
            if j == 0: #add tikhonov to the diagonal (fast way that requires less memory)
                LHS.ravel()[::LHS.shape[1]+1] += tikh[j]
            else:
                LHS.ravel()[::LHS.shape[1]+1] += tikh[j] - tikh[j-1]
            Wout[j, jj] = np.linalg.solve(LHS,RHS)
#             Wout[j, jj] = np.linalg.solve(LHS + tikh[j]*np.eye(N_units+1),
#                                               RHS)#, assume_a='pos')
            
            if j==0 and k==0: print('linear system time:', time.time() - t1)

    return Xa1, Wout, LHS, RHS

def train_save_n(U_washout, U_train, Y_train, tikh, sigma_in, rho, noise):
    """ Trains ESN.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """
    
    ## washout phase
#     xf_washout = open_loop(U_washout, np.zeros(N_units), sigma_in, rho)[-1,:N_units]
    
    xf = open_loop(U_washout, np.zeros(N_units), sigma_in, rho)[-1,:N_units]
    
    ## splitting training in N_splits to save memory
    LHS = 0
    RHS = 0
#     N_splits = 4
    N_len = (U_train.shape[0]-1)//N_splits
    
    for i in range(N_splits):
        t1  = time.time()
        ## open-loop train phase
        Xa1 = open_loop(U_train[i*N_len:(i+1)*N_len], xf, sigma_in, rho)[1:]
        xf  = Xa1[-1,:N_units].copy()
        #if i == 0: print('open_loop time:', (time.time()-t1)*N_splits)

        t1  = time.time()   
        LHS += np.dot(Xa1.T, Xa1) 
        RHS += np.dot(Xa1.T, Y_train[i*N_len:(i+1)*N_len])
        #if i == 0: print('matrix multiplication time:', (time.time()-t1)*N_splits)
    
    if N_splits > 1:# to cover the last part of the data that didn't make into the even splits
        Xa1 = open_loop(U_train[(i+1)*N_len:], xf, sigma_in, rho)[1:]
        LHS += np.dot(Xa1.T, Xa1) 
        RHS += np.dot(Xa1.T, Y_train[(i+1)*N_len:])

    ## open-loop train phase
#     Xa = open_loop(U_train, xf_washout, sigma_in, rho)
    
#     sh_0      = Xa.shape[0]
#     state_std = np.std(Xa,axis=0)
    
#     for i in range(N_units+1):
#             Xa[:,i] = Xa[:,i].copy() + rnd.normal(0, noise*state_std[i], sh_0)
    
    ## Ridge Regression
#     LHS  = np.dot(Xa[1:].T, Xa[1:])
#     sh_0 = Y_train.shape[0]
    
#     for i in range(N_latent):
#             Y_train[:,i] = Y_train[:,i] + rnd.normal(0, noise*data_std[i], sh_0)
#     RHS  = np.dot(Xa[1:].T, Y_train)
    LHS.ravel()[::LHS.shape[1]+1] += tikh
    Wout = np.linalg.solve(LHS, RHS)
    
#     print(Wout[0,:3])

    return Wout