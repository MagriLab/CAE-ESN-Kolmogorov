# Tutorial

Running 2.-4., which consists of training and testing the architecture, requires ~20min with a dedicated GPU.

The structure is organised as follows:

1. Gen_data.ipynb produces the dataset by solving the Kolmogorov flow using KolSol (https://github.com/MagriLab/KolSol), a pseudospectral solver implemented in pytorch. Alternatively, the dataset can be downloaded from (https://zenodo.org/record/7698307#.ZAMyrtLP1mg).

2. Autoencoder.ipynb (i) trains the multiscale convolutional autoencoder (implemented in tensorflow 2.x), (ii) shows the performance in the test set of the reconstruction and (iii) saves the encoded data for the ESN to be trained on.

3. ESN.ipynb (i) trains the echo state network to predict the latent space dynamics and (ii) shows the performance of the ESN in predicting the latent dynamics in the test set.

4. CAE-ESN.ipynb shows (i) a comparison with Proper Orthogonal Decomposition (also known as Principal Component Analysis), (ii) the prediction in time of the flowfield in the test set.


Detailed comments are provided within each script.
