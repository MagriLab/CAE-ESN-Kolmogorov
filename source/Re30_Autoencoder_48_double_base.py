#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = "1" #set cores for numpy
import numpy as np
import tensorflow as tf
import json
tf.get_logger().setLevel('ERROR') #no info and warnings are printed 
tf.config.threading.set_inter_op_parallelism_threads(1) #set cores for TF
tf.config.threading.set_intra_op_parallelism_threads(1)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices([], 'GPU') #runs the code without GPU
import matplotlib.pyplot as plt
import h5py
import time
from pathlib import Path
import matplotlib as mpl
plt.style.use('dark_background')

#Latex
mpl.rc('text', usetex = True)
mpl.rc('font', family = 'serif')


# ## Data Handling

# In[2]:


#load data
upsample   = 1
Re         = 30
data_len   = 300000
transient  = 10000//upsample

U        = 0 

Nx       = 48
Nu       = 2
# file obtained from Gen_data.ipynb
fln      = '/rds/user/ar994/hpc-work/Kolmogorov_0.1_48_30.0_100100_32.h5'
hf       = h5py.File(fln,'r')
dt       = 0.1
U        = np.array(hf.get('U')).astype('float32')[transient:transient+data_len:upsample]
hf.close()


# In[3]:


N_x     = U.shape[1]
N_y     = U.shape[2]

print('U-shape:',U.shape, dt)

def split_data(U, b_size, n_batches):
    
    '''
    Splits the data in batches. Each batch is created by sampling the signal with interval
    equal to n_batches.
    U0 and U1 are stored to compute the time-derivative with a first-order numerical scheme.
    '''
    U0     = U[::skip].copy()
    U1     = U[1::skip].copy()
    data   = np.zeros((n_batches, b_size*2, U.shape[1], U.shape[2], U.shape[3]))    
    for j in range(n_batches):
        data[j,:b_size] = U0[j::n_batches].copy()
        data[j,b_size:] = U1[j::n_batches].copy()

    return data

b_size      = 50   #batch_size
n_batches   = 500  #number of batches
val_batches = 50   #validation batches
skip        = 10   #we sample the timeseries every skip timesteps


print('Train Data%  :',b_size*n_batches*skip/U.shape[0]) #how much of the data we are using for training
print('Val   Data%  :',b_size*val_batches*skip/U.shape[0])

# training data
U_tt        = np.array(U[:b_size*n_batches*skip].copy())                 #to be used for random batches
U_train     = split_data(U_tt, b_size, n_batches).astype('float32') #to be used for randomly shuffled batches
# validation data
U_vv        = np.array(U[-b_size*val_batches*skip:].copy())
U_val       = split_data(U_vv, b_size, val_batches).astype('float32')             
rng = np.random.default_rng() #random generator for later shuffling


# ## Autoencoder functions

# In[23]:


def model(inputs, enc_mods, dec_mods, is_train=False):
    
    '''
    Multiscale autoencoder, taken from Hasegawa 2020. The contribution of the CNNs at different
    scales are simply summed.
    '''
        
    # sum of the contributions of the different encoders/decoders
    encoded_0 = 0
    for enc_mod in enc_mods:
        encoded_0 += enc_mod(inputs, training=is_train)
        
            
    decoded_0 = 0
    for dec_mod in dec_mods:
        decoded_0 += dec_mod(encoded_0, training=is_train)
        
    return encoded_0, decoded_0

@tf.function #this creates the tf graph
def train_step(inputs, enc_mods, dec_mods, train=True):
    
    """
    Trains the model by minimizing the loss between input and output
    """
    
    # autoencoded field 
    if is_der:
        decoded_0  = model(inputs, enc_mods, dec_mods, is_train=train)[-1]
    else:
        decoded_0  = model(inputs[:b_size], enc_mods, dec_mods, is_train=train)[-1]

    
#     decoded_0  = model(inputs, enc_mods, dec_mods, Dense, is_train=train)[-2:]

    # loss with respect to the data
    loss      = (Loss_Mse(inputs[:b_size], decoded_0[:b_size]))
    loss      = (loss)/U_std
    
    if is_der:
        inp_der   = (inputs[:b_size]      - inputs[b_size:]   )  /dt
        dec_der   = (decoded_0[:b_size]   - decoded_0[b_size:])  /dt

        loss_der  = Loss_Mse(inp_der, dec_der)
        loss_der  = (loss_der)/Uder_std
    
        losss     = loss_der + loss
    else:
        losss     = loss
        loss_der  = 1e-12
    
    # compute and apply gradients inside tf.function environment for computational efficiency
    if train:
        # create a variable with all the weights to perform gradient descent on
        # appending lists is done by plus sign
        varss    = []
        for enc_mod in enc_mods:
            varss  += enc_mod.trainable_weights
        for dec_mod in dec_mods:
            varss +=  dec_mod.trainable_weights
        
        #compute and apply gradients
        grads   = tf.gradients(losss, varss)
        optimizer.apply_gradients(zip(grads, varss))
    
    return loss, loss_der


# In[24]:


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


# ## Autoencoder parameters

# In[25]:


# define the model
# we do not have pooling and upsampling, instead we use stride=2 and padding


n_fil         = [6,12,24,1]                  #size of filters in encoder
n_dec         = [24,12,6,3]                  #size of filters in decoder
N_parallel    = 3                          #number of CNN for multiscale
ker_size      = [(5,5), (3,3), (7,7)]      #kernel sizes
N_layers      = 4                          #number of layers in every CNN
stride        = 2                          #stride for the kernel
act           = 'tanh'                     #activation function

is_der        = False   #Flag to include derivative loss in training
if is_der:
    der_string = 'der_'
else:
    der_string = 'base_'
    
is_double    = True     #flag to generate a more complex autoencoder, in which each convolution with 

                        #stride=2 is followed by a stride=1 conv 
                        #(better performance, more difficult training)
if is_double:
    dbl_string = 'double_'
else:
    dbl_string = 'single_'

load          = False #flag to load already trained model
if load:
    #to load already existing model to continue training
    N_latent = 9
    path = './data/48_RE30_'+der_string+dbl_string+str(ker_size)+'_'+act+'_'+str(N_latent)
    Loss_Mse    = tf.keras.losses.MeanSquaredError()
    
    #Load best model
    enc_mods = [None]*N_parallel
    dec_mods = [None]*N_parallel
    for i in range(N_parallel):
        enc_mods[i] = tf.keras.models.load_model(path + '/enc_mod'+str(ker_size[i])+                                '_'+str(N_latent)+'.h5', custom_objects={"PerPad2D": PerPad2D})
        dec_mods[i] = tf.keras.models.load_model(path + '/dec_mod'+str(ker_size[i])+                                '_'+str(N_latent)+'.h5', custom_objects={"PerPad2D": PerPad2D})
    
else:

    pad_enc       = 'valid'         #no padding in the conv layer
    pad_dec       = 'valid'
    asyms_enc     = [True,True,True,True,True]    #these have to be manually depending on input and kernel size
    p_size        = [1,0,2]          #stride = 2 padding          
    p_fin         = [2,1,3]          #stride = 1 padding
    if is_double:
        p_dec         = 1                #initial padding in the decoder
    else:
        p_dec         = 1                #initial padding in the decoder
    
    p_crop        = [6,12,24,48]     #size of cropping in the decoder

    #initialize the encoders and decoders with different kernel sizes    
    enc_mods      = [None]*(N_parallel)
    dec_mods      = [None]*(N_parallel)    
    for i in range(N_parallel):
        enc_mods[i] = tf.keras.Sequential(name='Enc_' + str(i))
        dec_mods[i] = tf.keras.Sequential(name='Dec_' + str(i))


    #generate encoder layers    
    for j in range(N_parallel):
        for i in range(N_layers):
       

            #stride=2 padding and conv
            enc_mods[j].add(PerPad2D(padding=p_size[j], asym=asyms_enc[i],
                                              name='Enc_' + str(j)+'_PerPad_'+str(i)))
            enc_mods[j].add(tf.keras.layers.Conv2D(filters = n_fil[i], kernel_size=ker_size[j],
                                          activation=act, padding=pad_enc, strides=stride,
                            name='Enc_' + str(j)+'_ConvLayer_'+str(i)))

            #stride=1 padding and conv
            if is_double and i<N_layers-1:
                enc_mods[j].add(PerPad2D(padding=p_fin[j], asym=False,
                                                          name='Enc_'+str(j)+'_Add_PerPad1_'+str(i)))
                enc_mods[j].add(tf.keras.layers.Conv2D(filters=n_fil[i],
                                                        kernel_size=ker_size[j], 
                                                    activation=act,padding=pad_dec,strides=1,
                                                        name='Enc_'+str(j)+'_Add_Layer1_'+str(i)))        


    N_1 = enc_mods[-1](U_train[0]).shape
    N_latent = N_1[-3]*N_1[-2]*N_1[-1]


    #generate decoder layers            
    for j in range(N_parallel):

        #reshape layer in case the input is flattened
        dec_mods[j].add(tf.keras.layers.Reshape((N_1[-3], N_1[-2], N_1[-1]), 
                        name='Dec_' + str(j)+'_Reshape'))
        for i in range(N_layers):

            #initial padding of latent space
            if i==0: 
                dec_mods[j].add(PerPad2D(padding=p_dec, asym=False,
                                              name='Dec_' + str(j)+'_PerPad_'+str(i))) 

            if is_double:
                #stride=2 transpose conv
                dec_mods[j].add(tf.keras.layers.Conv2DTranspose(filters = n_dec[i],#n_fil[N_layers-i-1],
                                               output_padding=None,kernel_size=ker_size[j],
                                              activation=act, padding=pad_dec, strides=stride,
                                    name='Dec_' + str(j)+'_ConvLayer_'+str(i)))
                if  i<N_layers-1:       
                    dec_mods[j].add(tf.keras.layers.Conv2D(filters=n_dec[i],
                                                kernel_size=ker_size[j], 
                                               activation=act,padding=pad_dec,strides=1,
                                              name='Dec_' + str(j)+'_ConvLayer1_'+str(i)))
            else:
            #stride=2 transpose conv
                dec_mods[j].add(tf.keras.layers.Conv2DTranspose(filters = n_dec[i],#n_fil[N_layers-i-1],
                                                   output_padding=None,kernel_size=ker_size[j],
                                                  activation=act, padding=pad_dec, strides=stride,
                                        name='Dec_' + str(j)+'_ConvLayer_'+str(i)))
                #either convolutional with stride=1 or cropping the center of the image
                if  i<N_layers-1:
#                     print(p_crop[i]+p_dec*3)
                    dec_mods[j].add(tf.keras.layers.CenterCrop(p_crop[i]+p_dec*2,
                                           p_crop[i]+p_dec*2,
                                        name='Dec_' + str(j)+'_Crop_'+str(i)))



        #crop and final linear convolution with stride=1
        dec_mods[j].add(tf.keras.layers.CenterCrop(p_crop[i] + 2*p_fin[j],
                                                       p_crop[i]+ 2*p_fin[j],
                                name='Dec_' + str(j)+'_Crop_'+str(i)))
        dec_mods[j].add(tf.keras.layers.Conv2D(filters=U.shape[3],
                                                kernel_size=ker_size[j], 
                                                activation='linear',padding=pad_dec,strides=1,
                                                  name='Dec_' + str(j)+'_Final_Layer'))


    # run the model once to check dimensions
    enc0, dec0 = model(U_train[0], enc_mods, dec_mods)
    print('latent   space size:', N_latent)
    print('physical space size:', U[0].flatten().shape)
    print('')
    for j in range(3):
        enc_mods[j].summary()
    for j in range(3):
        dec_mods[j].summary()
    
# DFDF

# In[26]:


#computing the normalization factor in fron of the losses
U2_mean     = np.mean(U**2,axis=(1,2,3))
U_std       = np.std(U2_mean) 

U_der       = np.zeros(U.shape)
for i in range(U.shape[0]-2):
    U_der[i] = (U[i+2] - U[i])/2/dt
Uder2_mean  = np.mean(U_der[:-2]**2,axis=(1,2,3))
Uder_std    = np.std(Uder2_mean)

#visual check that they have similar order of magnitude in the std
plt.rcParams["figure.figsize"] = (15,4)
plt.rcParams["font.size"] = 20
plt.subplot(121)
plt.ylim(0.0001,1)
plt.hist((U2_mean)/U_std, bins=100, density=True, log=True, histtype='step')
plt.subplot(122)
plt.ylim(0.0001,1)
plt.hist(Uder2_mean/Uder_std,bins=100, density=True, log=True, histtype='step')
plt.close()
print(U_std, Uder_std)


# ## Training the autoencoder

# In[27]:


plt.rcParams["figure.figsize"] = (15,4)
plt.rcParams["font.size"]  = 20

Loss_Mse    = tf.keras.losses.MeanSquaredError()

n_epochs    = 2001 #number of epochs

#define optimizer and initial learning rate   
optimizer   = tf.keras.optimizers.Adam(amsgrad=True) #amsgrad True for better convergence

if not load:
    l_rate = 0.002
else:
    ## Load optimizer parameters from previous run to continue training
    hf = h5py.File(path + '/opt_weights.h5','r')
    l_rate      = np.array(hf.get('l_rate'))
    leng        = np.array(hf.get('length'))
    min_weights = [None for i in range(leng+1)]
    for i in range(leng+1):
         min_weights[i] = np.array(hf.get('weights_'+str(i)))
    hf.close()
            
optimizer.learning_rate = l_rate

lrate_update = True #flag for l_rate updating
lrate_mult   = 0.75
N_lr         = 100  #number of epochs before which the l_rate is not updated

# quantities to check and store the training and validation loss and the training goes on
old_loss      = np.zeros(n_epochs) #needed to evaluate training loss convergence
tloss_plot    = np.zeros(n_epochs) #training loss
vloss_plot    = np.zeros(n_epochs) #validation loss
tloss1_plot   = np.zeros(n_epochs) #training_der loss
vloss1_plot   = np.zeros(n_epochs) #validation_der loss
old_loss[0]  = 1e6 #initial value has to be high
N_check      = 5   #each N_check epochs we check convergence and validation loss
patience     = 200 #if the val_loss has not gone down in the last patience epochs, early stop
last_save    = patience

t            = 1 # initial (not important value) to monitor the time of the training

for epoch in range(n_epochs):
    
    if epoch - last_save > patience: break #early stop
                
    #Perform gradient descent for all the batches every epoch
    loss_0 = 0
    loss_1 = 0
    rng.shuffle(U_train, axis=0) #shuffle batches
    for j in range(n_batches):
            loss, loss1  = train_step(U_train[j], enc_mods, dec_mods)
            loss_0 += loss
            loss_1 += loss1
    
    #save training loss each epoch
    loss_0             = loss_0.numpy()/n_batches
    loss_1             = loss_1.numpy()/n_batches
    tloss_plot[epoch]  = loss_0
    tloss1_plot[epoch] = loss_1
    
    if load and epoch == 0:
        #loading weights in the first epoch for model if loading trained network
        print('LOADING MINIMUM')
        for i in range(N_parallel):
            enc_mods[i].load_weights(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
            dec_mods[i].load_weights(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
            
        optimizer.set_weights(min_weights)
        
        loss_0 = 0
        loss_1 = 0
        for j in range(n_batches):
            loss, loss1 = train_step(U_train[j], enc_mods, dec_mods, train=False)
            loss_0     += loss
            loss_1     += loss1
        loss_0         = loss_0.numpy()/n_batches
        loss_1         = loss_1.numpy()/n_batches 
        tloss_plot[0]  = 0
        tloss1_plot[0] = 0        
    
    # every N epochs checks the convergence of the training loss and val loss
    if (epoch%N_check==0):
        
        #Compute Validation Loss
        loss_val        = 0
        loss_val1       = 0
        for j in range(val_batches):
            loss, loss1 = train_step(U_val[j], enc_mods, dec_mods,
                                    train=False)
            loss_val   += loss
            loss_val1  += loss1
        loss_val   = loss_val.numpy()/val_batches
        loss_val1  = loss_val1.numpy()/val_batches
        vloss_plot[epoch]  = loss_val
        vloss1_plot[epoch] = loss_val1
        
        # Halves the learning rate if the training loss is not going down with respect to 
        # N_lr epochs before
        if epoch > N_lr:
            #check if the training loss is smaller than the average training loss N_lr epochs ago
            tt_loss   = tloss_plot[epoch-N_lr//2:epoch] + tloss1_plot[epoch-N_lr//2:epoch]
            deviation = np.mean(tt_loss)
            if lrate_update and deviation > old_loss[epoch-N_lr]:
                #if it is larger, load optimal val loss weights and decrease learning rate
                print('LOADING MINIMUM')
                for i in range(N_parallel):
                    enc_mods[i].load_weights(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
                    dec_mods[i].load_weights(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')

                optimizer.learning_rate = optimizer.learning_rate*lrate_mult
                optimizer.set_weights(min_weights)
                print('LEARNING RATE CHANGE', optimizer.learning_rate.numpy(), deviation)
                old_loss[epoch-N_lr:epoch] = 1e6 #so that l_rate is not changed for N_lr steps
        
        #store current loss
        old_loss[epoch] = loss_0 + loss_1
        
        #save best model (the one with minimum validation loss)
        if epoch > 1 and         loss_val + loss_val1 < (vloss_plot[:epoch-1][np.nonzero(vloss_plot[:epoch-1])] +                                 vloss1_plot[:epoch-1][np.nonzero(vloss_plot[:epoch-1])]).min():
            
            #saving the model weights
            print('Saving Model..')
            path = './data/48_RE30_'+der_string+dbl_string+str(ker_size)+'_'+act+'_'+str(N_latent)
            Path(path).mkdir(parents=True, exist_ok=True) #creates directory even when it exists
            for i in range(N_parallel):
                enc_mods[i].save(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5')
                dec_mods[i].save(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5')
                enc_mods[i].save_weights(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
                dec_mods[i].save_weights(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
            
            #saving optimizer parameters
            min_weights = optimizer.get_weights()
            hf = h5py.File(path + '/opt_weights.h5','w')
            for i in range(len(min_weights)):
                hf.create_dataset('weights_'+str(i),data=min_weights[i])
            hf.create_dataset('length', data=i)
            hf.create_dataset('l_rate', data=optimizer.learning_rate)  
            hf.close()
            
            last_save = epoch

        # Print loss values and training time (per epoch)
        print('Epoch', epoch, '; TLoss_der', loss_1,   '; TLoss', loss_0,  
              '; TLoss', loss_0*U_std)
        print('Epoch', epoch, '; VLoss_der', loss_val1,'; VLoss', loss_val, 
              '; VLoss', loss_val*U_std)
        print('Epoch', epoch, '; Ratio', (loss_val+loss_val1)/(loss_0+loss_1),
              '; time', (time.time()-t)/N_check)
        print('')
        
        t = time.time()
        
    if (epoch%200==0) and epoch != 0:    
        #plot convergence of training and validation loss
        plt.subplot(1,2,1)
        plt.title('MSE convergence')
        plt.yscale('log')
        plt.grid(True, axis="both", which='both', ls="-", alpha=0.3)
        plt.plot(tloss_plot[np.nonzero(tloss_plot)], 'y', label='Train loss')
        plt.plot(np.arange(np.nonzero(vloss_plot)[0].shape[0])*N_check,
                 vloss_plot[np.nonzero(vloss_plot)], label='Val loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.subplot(1,2,2)
        plt.title('MSE-Der convergence')
        plt.yscale('log')
        plt.grid(True, axis="both", which='both', ls="-", alpha=0.3)
        plt.plot(tloss1_plot[np.nonzero(tloss_plot)], 'y', label='Train loss')
        plt.plot(np.arange(np.nonzero(vloss1_plot)[0].shape[0])*N_check,
                 vloss1_plot[np.nonzero(vloss1_plot)], label='Val loss')
        plt.xlabel('epochs')
        
        plt.tight_layout()
        plt.close()


# In[ ]:


#save training and validation loss convergence
if not load:
    
        hf = h5py.File(path + '/loss_conv.h5','w')
        hf.create_dataset('t_loss',  data=tloss_plot[np.nonzero(tloss_plot)])
        hf.create_dataset('v_loss',  data=vloss_plot[np.nonzero(vloss_plot)]) 
        hf.create_dataset('t_loss1', data=tloss1_plot[np.nonzero(tloss_plot)])
        hf.create_dataset('v_loss1', data=vloss1_plot[np.nonzero(vloss_plot)]) 
        hf.create_dataset('N_check', data=N_check)
        hf.close()
        
        plt.rcParams["figure.figsize"] = (10,5)
        plt.rcParams["font.size"]  = 20
    
        #plot convergence of training and validation loss
        plt.figure()
        plt.title('Loss convergence')
        plt.yscale('log')
        plt.grid(True, axis="both", which='both', ls="-", alpha=0.3)
        plt.plot(tloss_plot[np.nonzero(tloss_plot)], 'y', label='Train loss')
        plt.plot(np.arange(np.nonzero(vloss_plot)[0].shape[0])*N_check,
                 vloss_plot[np.nonzero(vloss_plot)], label='Val loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.tight_layout(pad=0.1)
        plt.savefig(path+'/loss_convergence.pdf')
        plt.close()

