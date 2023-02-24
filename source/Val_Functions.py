#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import h5py


# ## Validation Strategies

# In[2]:


#Objective Functions to minimize with Bayesian Optimization


def RVC_Noise(x):
    # chaotic Recycle Validation
    
    global rho, sigma_in, tikh_opt, k, ti, noise_opt
    rho      = x[0]
    sigma_in = round(10**x[1],3)
        
    #print(ti - time.time())
    ti       = time.time()
        
    lenn     = tikh.size
    len1     = noises.size
    Mean     = np.zeros((lenn, len1))
#     print(Mean.shape)
    
    #Train using tv: training+val
    Xa_train, Wout = train_n(U_washout, U_tv, Y_tv, tikh, sigma_in, rho)[:2]

    #Different Folds in the validation set
    t1   = time.time()
    for i in range(N_fo):

        p      = N_in + i*N_fw
        Y_val  = U[N_washout + p : N_washout + p + N_val].copy()
        U_wash = U[            p : N_washout + p        ].copy()
        
    #         #Train: remove the validation interval
#         Xa1    = Xa_train[p+1:p+N_val+1]
#         Y_t1   = Y_tv[p:p+N_val] #Xa_train and Y_tv indices are shifted by one

#         LHS   = LHS0 - np.dot(Xa1.T, Xa1)
#         RHS   = RHS0 - np.dot(Xa1.T, Y_t1)

#         xf          = Xa_train[p-1].copy()
        xf          = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]
        
        for jj in range(len1):
            for j in range(lenn):
    #             Wout[j]  = np.linalg.solve(LHS + tikh[j]*np.eye(N_units+1), RHS)

                #Validate
                Yh_val      = closed_loop(N_val-1, xf, Wout[j, jj], sigma_in, rho)[0]

                Mean[j,jj] += np.log10(np.mean((Y_val-Yh_val)**2))
                
#                 if np.log10(np.mean((Y_val-Yh_val)**2)) < -5:
#                     plt.plot(Y_val[:,-10:], 'w')
#                     plt.plot(Yh_val[:,-10:], '--r')
#                     plt.show()
        
#             if (i%10) == 0:
#                 plt.plot(Yh_val[:,0], label=str(tikh[j]))
#         if (i%10) == 0:
#             plt.plot(Y_val[:,0], 'k')
#             plt.ylim(Y_val[:,0].min()-0.5, Y_val[:,0].max()+0.5)
#             plt.legend()
#             plt.show()

            # prevent from diverging to infinity: put MSE equal to 10^10
                if np.isnan(Mean[j,jj]) or np.isinf(Mean[j,jj]):
                    Mean[j,jj] = 10*N_fo
                
    if k==0: print('closed-loop time:', time.time() - t1)
#     a           = np.argmin(Mean)
    a = np.unravel_index(Mean.argmin(), Mean.shape)
    tikh_opt[k] = tikh[a[0]]
    noise_opt[k]= noises[a[1]]
#     print(Wout[a, :3,:3])
    k          +=1
#     print(k,'Mean:', Mean.flatten()/N_fo)
    print(k,'Par :', rho,sigma_in, tikh[a[0]], noises[a[1]], Mean[a]/N_fo)
    print('')


    return Mean[a]/N_fo

def RVC_Noise_PH(x):
    # chaotic Recycle Validation
    
    global rho, sigma_in, tikh_opt, k, ti, noise_opt
    rho      = x[0]
    sigma_in = round(10**x[1],3)
        
    print(ti - time.time())
    ti       = time.time()
        
    lenn     = tikh.size
    len1     = noises.size
    Mean     = np.zeros((lenn, len1))
#     print(Mean.shape)
    
    #Train using tv: training+val
    Wout = train_n(U_washout, U_tv, Y_tv, tikh, sigma_in, rho)[1]

    #Different Folds in the validation set
    for i in range(N_fo):

        p      = N_in + i*N_fw
        Y_val  = U[N_washout + p : N_washout + p + N_val].copy()
        U_wash = U[            p : N_washout + p        ].copy()
        
    #         #Train: remove the validation interval
#         Xa1    = Xa_train[p+1:p+N_val+1]
#         Y_t1   = Y_tv[p:p+N_val] #Xa_train and Y_tv indices are shifted by one

#         LHS   = LHS0 - np.dot(Xa1.T, Xa1)
#         RHS   = RHS0 - np.dot(Xa1.T, Y_t1)

#         xf          = Xa_train[p-1].copy()
        xf          = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]
        
        for jj in range(len1):
            for j in range(lenn):
    #             Wout[j]  = np.linalg.solve(LHS + tikh[j]*np.eye(N_units+1), RHS)

                #Validate
                Yh_val      = closed_loop(N_val-1, xf, Wout[j, jj], sigma_in, rho)[0]
                
                Y_err       = np.sqrt(np.mean((Y_val-Yh_val)**2,axis=1))/sigma_ph
                PH          = np.argmax(Y_err>threshold_ph)
                if PH == 0 and Y_err[0]<threshold_ph: PH=N_val
                Mean[j,jj] += -PH #np.log10(np.mean((Y_val-Yh_val)**2))
                
#                 print(-np.argmax(Y_err>threshold_ph)*dt)
#                 if np.argmax(Y_err>threshold_ph) > N_val*.75:
#                     plt.axhline(threshold_ph)
#                     plt.plot(np.arange(N_val)*dt,Y_err, 'w')
#                     plt.ylim(0.,threshold_ph*1.5)
#                     plt.show()
        
#             if (i%10) == 0:
#                 plt.plot(Yh_val[:,0], label=str(tikh[j]))
#         if (i%10) == 0:
#             plt.plot(Y_val[:,0], 'k')
#             plt.ylim(Y_val[:,0].min()-0.5, Y_val[:,0].max()+0.5)
#             plt.legend()
#             plt.show()

            # prevent from diverging to infinity: put MSE equal to 10^10
#                 if np.isnan(Mean[j,jj]) or np.isinf(Mean[j,jj]):
#                     Mean[j,jj] = 10*N_fo
                
#     a           = np.argmin(Mean)
    a           = np.unravel_index(Mean.argmin(), Mean.shape)
    tikh_opt[k] = tikh[a[0]]
    noise_opt[k]= noises[a[1]]
#     print(Wout[a, :3,:3])
    k          +=1
#     print(k,'Mean:', Mean.flatten()/N_fo)
    print(k,'Par :', rho,sigma_in, tikh[a[0]], noises[a[1]], Mean[a]/N_fo*dt)


    return Mean[a]/N_fo*dt