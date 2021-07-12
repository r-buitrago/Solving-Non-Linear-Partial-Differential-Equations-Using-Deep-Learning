# Solving-Non-Linear-Partial-Differential-Equations-Using-Deep-Learning
Implementation of the Deep BSDE method to innovative models in Financial Mathematics

Here I present the code that I have used in my final bachelor thesis in Mathematics, in which I implement the Deep BSDE algorithm (https://doi.org/10.1073/pnas.1718942115) to solve variants of the Black-Scholes equation. This algorithm takes advantage of the stochastic nature of the PDE's to use Deep Learning to solve them. 

In this repository you can find the solution to two PDE's: the standard Black-Scholes equation (deep_bsde_BS_standard.ipynb), and the Black-Scholes equation with default risk (deep_bsde_BS_defaultrisk.ipynb). You can check out the modelling of the latter equation in my thesis, which I have also uploaded to this repository.

In any of the before mentioned files, the key variables are the functions sigma, mu, f and g. Those variables are taken from equation 63 of my thesis. To solve any parabolic semilinear PDE, you can change them according to the equation 63, and you will have the PDE solution's initial values and gradients. 

The files I have uploaded are Python notebooks (.ipynb), which call the python files (.py) that are present in the repository. If you download everything in the same folder and run the notebooks in Anaconda or any similar software, it will work fine. 
