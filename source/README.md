# Source files

This folder contains the scripts used to train the CAE-ESN described in https://arxiv.org/abs/2211.11379.

The structure is organised as follows:

- Data needs to be generated from ../tutorial/Gen_data.ipynb by solving the Kolmogorov flow using KolSol (https://github.com/MagriLab/KolSol), a pseudospectral solver implemented in pytorch.

- Once the flowfield time series is available, the autoencoder is trained using Re30_Autoencoder_48_double_base.py. The autoencoder provides the mapping to (and back from) the low-dimensional latent space.

- Then, we compute the encoded time series produced by the autoencoder in Encode.py. We need the encoded time series to train the echo state network.

- Finally, the echo state networks are trained in Train_ESN_RVC_Re30_48_base.py to predict the latent dynamics.

As discussed in the paper, the convolutional autoencoder and ESN are trained separately, and then used together for the prediction in the test set. 
The implementation for the CAE-ESN used in the test set can be found in ../tutorial/CAE-ESN.py.


- Without inputs from other files, Kolmogorov_Lyapunov_Spectrum.py computes the first N_exp Lyapunov exponents of the flow.


Comments are provided within each script.