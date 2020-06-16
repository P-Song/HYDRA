# HYDRA: Hybrid deep magnetic resonance fingerprinting
========================================================================


The software is for the paper "HYDRA: Hybrid deep magnetic resonance fingerprinting". The source codes are freely available for research and study purposes.

Purpose: 
    Magnetic resonance fingerprinting (MRF) methods typically rely on dictionary matching to map the temporal MRF signals to quantitative tissue parameters. 
    Such approaches suffer from inherent discretization errors, as well as high computational complexity as the dictionary size grows. 
    To alleviate these issues, we propose a HYbrid Deep magnetic ResonAnce fingerprinting (HYDRA) approach, referred to as HYDRA.

Methods: 
    HYDRA involves two stages: a model-based signature restoration phase and a learningbased parameter restoration phase. 
    Signal restoration is implemented using low-rank based de-aliasing techniques while parameter restoration is performed 
    using a deep nonlocal residual convolutional neural network. The designed network is trained on synthesized MRF data simulated with 
    the Bloch equations and fast imaging with steady-state precession (FISP) sequences. 
    In test mode, it takes a temporal MRF signal as input and produces the corresponding tissue parameters.


Reference:
----------------------------
If you use the source codes, please refer to the following papers for details and thanks for your citation.

[1] Pingfan Song, Yonina C. Eldar, Gal Mazor, Miguel R. D. Rodrigues, "HYDRA: Hybrid Deep Magnetic Resonance Fingerprinting", Medical Physics, 2019, doi: 10.1002/mp.13727. 

[2] Pingfan Song, Yonina C. Eldar, Gal Mazor, Miguel R. D. Rodrigues, "Magnetic Resonance Fingerprinting Using a Residual Convolutional Neural Network", ICASSP, pp. 1040-1044. IEEE, 2019.

[3] Gal Mazor, Lior Weizman, Assaf Tal, Yonina C. Eldar. "Low‐rank magnetic resonance fingerprinting." Medical physics 45, no. 9 (2018): 4066-4084.


Usage:
----------------------------
	- Run the code 'Gen_D_LUT.m' to generate simulated dictionary and look-up-table.

	- Run the code 'HYDRA_Step1.m' to perform the first step of HYDRA: reconstruct temporal signatures via using adapted FLOR algorithm. After the temporal signatures are reconstructed, they are input into the trained neural network to restore the parameter maps, i.e. the second step of HYDRA. Please refer to 'MRF_FullNL_ResCNN_T1T2_L1000_Test.py' in the upper folder for the second step.


Codes written & compiled by:
----------------------------
Pingfan Song 

Electronic and Electrical Engineering, Imperial College London, UK.

p.song@imperial.ac.uk, songpingfan@gmail.com












