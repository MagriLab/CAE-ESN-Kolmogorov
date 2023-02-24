# Tutorial
The structure is organised as follows:

- Gen_data.ipynb produces the dataset by solving the Kolmogorov flow using KolSol (https://github.com/MagriLab/KolSol), a pseudospectral solver implemented in pytorch.

- Autoencoder.ipynb (i) trains the multiscale convolutional autoencoder (implemented in tensorflow 2.x), (ii) shows the performance in the test set of the reconstruction and (iii) saves the encoded data for the ESN to be trained on.

- ESN.ipynb (i) trains the echo state network to predict the latent space dynamics and (ii) shows the performance of the ESN in predicting the latent dynamics in the test set.

- CAE-ESN.ipynb shows the combined architecture for the prediction of the flowfield in the test set.


Detailed comments are provided within each script.