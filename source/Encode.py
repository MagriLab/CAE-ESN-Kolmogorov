
import os
os.environ["OMP_NUM_THREADS"] = "1" #set cores for numpy
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR') #no info and warnings are printed 
tf.config.threading.set_inter_op_parallelism_threads(1) #set cores for TF
tf.config.threading.set_intra_op_parallelism_threads(1)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices([], 'GPU') #runs the code without GPU
import h5py

exec(open("Functions.py").read())

def model(inputs, enc_mods, dec_mods, is_train=False):
    
    '''
    Multiscale autoencoder, taken from Hasegawa 2020. The contribution of the CNNs at different
    scales are simply summed.
    '''
        
    # sum of the contributions of the different CNNs
    encoded = 0
    for enc_mod in enc_mods:
        encoded += enc_mod(inputs, training=is_train)
            
    decoded = 0
    for dec_mod in dec_mods:
        decoded += dec_mod(encoded, training=is_train)
        
    return encoded, decoded


# save the encoded data for the ESN (too much memory used for GPU)
N_pos     = 5000 #split in k interval of N_pos length needed to process long timeseries
k         = 75
transient = 10000
N_len = k*N_pos
fln      = '/data/ar994/Python/data/Kolmogorov/Kolmogorov_0.1_48_30.0_100100_32.h5' #change with relevant path
hf       = h5py.File(fln,'r')
dt       = 0.1
U        = np.array(hf.get('U')[transient:transient+N_len], dtype=np.float32)
hf.close()

N_x      = U.shape[1]
N_y      = U.shape[2]


Latents    = [18]
Re         = 30
N_parallel = 3
ker_size   = [(3,3), (5,5), (7,7)]      #kernel sizes

for N_latent in Latents:
    path = './../tutorial/data/48_RE30_'+str(N_latent) #change with relevant path
    a = [None]*N_parallel
    b = [None]*N_parallel
    for i in range(N_parallel):
        a[i] = tf.keras.models.load_model(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5', 
                                              custom_objects={"PerPad2D": PerPad2D})
    for i in range(N_parallel):
        b[i] = tf.keras.models.load_model(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5',
                                              custom_objects={"PerPad2D": PerPad2D})

    N_1   = [3,3,N_latent//9]
    U_enc = np.zeros((N_len, N_1[0], N_1[1], N_1[2]))
    #encode all the data to provide time series in latent space for the ESN
    for i in range(k):
        U_enc[i*N_pos:(i+1)*N_pos]= model(U[i*N_pos:(i+1)*N_pos], a, b)[0]

    fln = './../tutorial/data/48_Encoded_data_Re'+str(Re)+'_' \
                + str(N_latent) +'.h5'                                   #change with relevant path
    hf = h5py.File(fln,'w')
    hf.create_dataset('U_enc'      ,data=U_enc)  
    hf.close()
    print(fln)