 
# coding: utf-8

'''

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

[2] Pingfan Song, Yonina C. Eldar, Gal Mazor, Miguel R. D. Rodrigues, “Multimodal Image Super-Resolution via Joint Sparse Representations ...", IEEE Transactions on Computational Imaging, DOI: 10.1109/TCI.2019.2916502.–PingfanSong, Miguel Rodrigues, et al., "Magnetic Resonance Fingerprinting Using a Residual Convolutional Neural Network", ICASSP, pp. 1040-1044. IEEE, 2019.


Usage:
----------------------------
	- Run the code 'MRF_FullNL_ResCNN_T1T2_L1000_Train' to train the designed nonlocal residual CNN.

    - Run the code 'MRF_FullNL_ResCNN_T1T2_L1000_Test' to test the network on following cases:
        
        case 1: Testing on the synthetic dataset for comparing parameter restoration performance, i.e. testing on simulated MRF temporal signals.
            
        case 2: Testing on the anatomical dataset with full k-space sampling for comparing parameter restoration performance.
           
        case 3: Testing on the anatomical dataset with k-space subsampling factor 15% using Gaussian patterns. 
        
        case 4: Testing on the anatomical dataset with k-space subsampling factor 9% using Spiral patterns. 


Codes written & compiled by:
----------------------------
Pingfan Song 

Electronic and Electrical Engineering, Imperial College London, UK.

p.song@imperial.ac.uk, songpingfan@gmail.com

'''

# In[1]:

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Embedding, Input
from keras.layers.merge import add
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.constraints import maxnorm
from keras import regularizers
from keras.optimizers import *
from keras.models import model_from_json # load model from .json file
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import normalize
import os
keras.__version__
import scipy.io
import keras.backend as K
import time
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from non_local import non_local_block


# In[1]: set GPU resource quota
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))


# In[34]:


def psnr(target,ref, peak_val=1.):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)

    diff = ref_data - target_data
#     print(diff.shape)
    diff = diff.flatten('C')

    rmse = np.sqrt(np.mean(diff ** 2.))
    
    psnr = 20 * np.log10(peak_val / rmse)

    return psnr

def snr(target,ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    
    diff = ref_data - target_data
#     print(diff.shape)
    diff = diff.flatten('C')
    rmse = np.sqrt(np.mean(diff ** 2.))
    
    target_data = target_data.flatten('C')
    power = np.sqrt(np.mean(target_data ** 2.))
    
    snr = 20*np.log10(power/rmse);
    return snr

def rmse(target,ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    
    diff = ref_data - target_data
#     print(diff.shape)
    diff = diff.flatten('C')
    rmse = np.sqrt(np.mean(diff ** 2.))
    return rmse

def mre(target,ref): # mean relative error
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    
    meanRef = np.mean(ref_data.flatten('C'))
    if meanRef != 0:
        diff = np.abs((ref_data - target_data)/meanRef)
    else:
        diff = np.abs(ref_data - target_data)
        
#    print(diff.shape)
    diff = diff.flatten('C')
   
    mre = np.mean(diff)
    return mre

#%%
    


#%% Case 1
# Testing on the synthetic dataset for comparing parameter restoration performance, i.e. testing on simulated MRF temporal signals.


MRFData = scipy.io.loadmat('D_LUT_L1000_TE10_TestRandom.mat') #
Label = MRFData['LUT']
X = MRFData['D']
X = X[:,0::1] # fully-sampled from 1000 time points;
X = normalize(X, norm = 'l2', axis=1)# L2 normalization along time dimention
X = np.expand_dims(X, axis=2)
plt.figure()
Xpart = X[300:16000:2000,:,0]
print(Xpart.shape)
plt.plot(np.real(np.transpose(Xpart)))
plt.show()

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
#model.load_weights("model.h5") # load saved weights in final epoch.
model.load_weights("weights.best.hdf5") # load saved weights from the checkpoint.
print("Loaded model from disk")


#%%

# calculate predictions
Tstart = time.clock()
predictions = model.predict(X)
Tend = time.clock()
Tcost = Tend - Tstart

# compute correlation coefficients
coeff_T1 = np.corrcoef(Label[:,0],predictions[:,0])
coeff_T1 = coeff_T1[0,1]
coeff_T2 = np.corrcoef(Label[:,1],predictions[:,1])
coeff_T2 = coeff_T2[0,1]

# compute RMSE
PSNR_T1 = psnr(predictions[:,0],Label[:,0],5000)
PSNR_T2 = psnr(predictions[:,1],Label[:,1],2000)

SNR_T1 = snr(predictions[:,0],Label[:,0])
SNR_T2 = snr(predictions[:,1],Label[:,1])

RMSE_T1 = rmse(predictions[:,0],Label[:,0])
RMSE_T2 = rmse(predictions[:,1],Label[:,1])

print('{:0.2f} / {:0.2f}'.format( PSNR_T1 ,  PSNR_T2 ))
print('{:0.2f} / {:0.2f}'.format( SNR_T1 ,  SNR_T2 ))
print('{:0.2f} / {:0.2f}'.format( RMSE_T1 ,  RMSE_T2 ))
print('{:0.8f} / {:0.8f}'.format( coeff_T1 ,  coeff_T2 ))


#%%
FileName = 'HYDRA_Test_1D_synthetic.npz'
np.savez(FileName,PSNR_T1 = PSNR_T1,PSNR_T2 = PSNR_T2,SNR_T1 = SNR_T1,SNR_T2 = SNR_T2,
         RMSE_T1 = RMSE_T1, RMSE_T2 = RMSE_T2, coeff_T1 = coeff_T1, coeff_T2 = coeff_T2,
         Label = Label, predictions = predictions, Tcost = Tcost)

#%% load reconstructed 1D synthetic data 
Results=np.load(FileName)
print(Results.keys())
print('{:0.2f} / {:0.2f}'.format(Results['PSNR_T1'], Results['PSNR_T2']))
print('{:0.2f} / {:0.2f}'.format(Results['SNR_T1'], Results['SNR_T2']))
print('{:0.2f} / {:0.2f}'.format(Results['RMSE_T1'], Results['RMSE_T2']))
print('{:0.8f} / {:0.8f}'.format(Results['coeff_T1'], Results['coeff_T2']))


T1 = Results['predictions'][:,0]
T1 = T1.flatten()
T2 = Results['predictions'][:,1]
T2 = T2.flatten()
#T1 = np.squeeze(T1)
#T2 = np.squeeze(T2)

FigNameT1 = "T1_CNN_1Dsimu.png"
FigNameT2 = "T2_CNN_1Dsimu.png"
FigNameT1res = "T1_res_CNN_1Dsimu.png"
FigNameT2res = "T2_res_CNN_1Dsimu.png"
FigNameT1corr = "T1_corr_CNN_1Dsimu.png"
FigNameT2corr = "T2_corr_CNN_1Dsimu.png"
FigNameT1error = "T1_error_CNN_1Dsimu.png"
FigNameT2error = "T2_error_CNN_1Dsimu.png"

#%% 
# show reconstruction

ind_T1 = np.argsort(Label[:,0])
Label_T1 = Label[ind_T1,0]
predictions_T1 = predictions[ind_T1,0]

ind_T2 = np.argsort(Label[:,1])
Label_T2 = Label[ind_T2[20:80000:1],1]
predictions_T2 = predictions[ind_T2[20:80000:1],1]
ind_T2 = np.argsort(Label[:,1])
Label_T2 = Label[ind_T2,1]
predictions_T2 = predictions[ind_T2,1]


plt.figure(figsize = (3,3))
plt.plot(Label_T1, predictions_T1,'r.',label='Estimation')
plt.plot(Label_T1, Label_T1,'b-',label='Reference')
#plt.title('T1_Corr')
plt.grid(True)
plt.xlim((0, 5000))
plt.ylim((0, 5000))
plt.xlabel('Reference T1 (ms)')
plt.ylabel('Estimated T1 (ms)')
plt.legend(loc='best')
plt.savefig(FigNameT1corr,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

plt.figure(figsize = (3,3))
plt.plot(Label_T2, predictions_T2,'r.',label='Estimation')
plt.plot(Label_T2, Label_T2,'b-',label='Reference')
#plt.title('T2_Corr')
plt.grid(True)
plt.xlim((0, 2000))
plt.ylim((0, 2000))
plt.xlabel('Reference T2 (ms)')
plt.ylabel('Estimated T2 (ms)')
plt.legend(loc='best')
plt.savefig(FigNameT2corr,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


#%%
# show error maps

plt.figure(figsize = (3,3))
plt.plot(Label_T1, predictions_T1-Label_T1,'r.',label='Estimation')
plt.grid(True)
plt.xlim((0, 5000))
plt.ylim((-100, 100))
plt.xlabel('Reference T1 (ms)')
plt.ylabel('Error of estimated T1 (ms)')
#plt.legend(loc='best')
plt.savefig(FigNameT1error,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


plt.figure(figsize = (3,3))
plt.plot(Label_T2, predictions_T2-Label_T2,'r.',label='Estimation')
plt.grid(True)
plt.xlim((0, 2000))
plt.ylim((-40, 40))
plt.xlabel('Reference T2 (ms)')
plt.ylabel('Error of estimated T2 (ms)')
#plt.legend(loc='best')
plt.savefig(FigNameT2error,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()



#%%


# In[22]: Case 2

# Testing on the anatomical dataset with full k-space sampling for comparing parameter restoration performance.

# In specific, testing on a stack of multi-contrast images. Each pixel position leads to a MRF temporal signal.


MRFData = scipy.io.loadmat('MRF_ImageStack_N128_L1000_TE10_Ratio0.15.mat')  #MRFData = scipy.io.loadmat('Groundtruth_T1_T2.mat')
print(MRFData.keys())
T1_true = MRFData['T1_128']
T1_true = T1_true[:,:,np.newaxis]
T2_true = MRFData['T2_128']
T2_true = T2_true[:,:,np.newaxis]
print(T1_true.shape, T2_true.shape)
Label = np.concatenate([T1_true, T2_true], axis=2)
Label = Label.reshape((128*128,-1))
print(Label[0:16000:1000].T)
print(Label.shape, Label.dtype)

#X = MRFData['X_estimated_old_mrf']
X = MRFData['X_fullysamp']
print(X.shape, X.dtype)
X = X.reshape((128*128,-1))
X = np.real(X)
#X = X[:,1::5] # sub-sampled from 1000 time points;

# remove those signature with too small value
NormX = np.zeros(X.shape[0])
NormX_index = np.empty(X.shape[0]) # index of valid values
NormX_index[:] = np.nan
print(NormX.shape, NormX_index.shape)
for i in range(0, X.shape[0]):
    NormX[i] = np.sum(X[i,:]**2)
    if NormX[i] < 1 : #20:
        X[i,:] = 0
        NormX_index[i] = i

np.set_printoptions(precision=2)
NormX_index = NormX_index[~np.isnan(NormX_index)]
NormX_index = NormX_index.astype('int32') # arrays used as indices must be of integer (or boolean) type
X = normalize(X, norm = 'l2', axis=1)# L2 normalization along time dimention
X = np.expand_dims(X, axis=2)
print(X.shape, X.dtype)


# In[23]:
# show true T1, T2

#MRFData = MRFData = scipy.io.loadmat('Groundtruth_T1_T2.mat')
#T1_true = MRFData['T1_128']
#T2_true = MRFData['T2_128']
#print(T1_true.shape, T2_true.shape)

T1max = 4500
T2max = 2500


mycmap = 'jet' # 'gray'
mydpi = 200

plt.figure()
plt.imshow(T1_true, cmap = mycmap)
plt.colorbar()
plt.clim(0,T1max)
plt.axis('off') 
plt.title('T1_true')
plt.grid(True)
plt.savefig("T1_true.png",bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

plt.figure()
plt.imshow(T2_true, cmap = mycmap)
plt.colorbar()
plt.clim(0,T2max)
plt.axis('off') 
plt.title('T2_true')
plt.grid(True)
plt.savefig("T2_true.png",bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()



plt.figure()
Xpart = X[300:16000:2000,:,0]
print(Xpart.shape)
plt.plot(np.real(np.transpose(Xpart)))
plt.show()


# In[30]:


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
#model.load_weights("model.h5") # load saved weights in final epoch.
model.load_weights("weights.best.hdf5") # load saved weights from the checkpoint.
print("Loaded model from disk")


# In[31]:

# calculate predictions
Tstart = time.clock()
predictions = model.predict(X)
Tend = time.clock()
Tcost = Tend - Tstart
print(predictions.shape)

predictions[NormX_index,:] = 0
print(predictions[0:200:10,:].T)
print(Label[0:200:10,:].T)

predictions = predictions.reshape((128,128,2))
print(predictions.shape, predictions.dtype)



# In[35]:


T1max = 4500
T2max = 2500

T1_true = np.squeeze(T1_true)
T2_true = np.squeeze(T2_true)

T1 = predictions[:,:,0]
T1[np.where((T1<0))] = 0
T1[np.where((T1>T1max))] = T1max
T2 = predictions[:,:,1]
T2[np.where((T2<0))] = 0
T2[np.where((T2>T2max))] = T2max

## remove invalid elements referring to the label.
#T1 = T1 * (T1_true > 0)
#T2 = T2 * (T2_true > 0)

PSNR_T1 = psnr(T1,T1_true,T1max)
PSNR_T2 = psnr(T2,T2_true,T2max)

SNR_T1 = snr(T1,T1_true)
SNR_T2 = snr(T2,T2_true)

RMSE_T1 = rmse(T1,T1_true)
RMSE_T2 = rmse(T2,T2_true)

# compute correlation coefficients
Label = Label.reshape((128*128,-1))
T1 = T1[:,:,np.newaxis]
T2 = T2[:,:,np.newaxis]
predictions = np.concatenate([T1, T2], axis=2)
predictions = predictions.reshape((128*128,-1))

coeff_T1 = np.corrcoef(Label[:,0],predictions[:,0])
coeff_T1 = coeff_T1[0,1]
coeff_T2 = np.corrcoef(Label[:,1],predictions[:,1])
coeff_T2 = coeff_T2[0,1]


print('{:0.2f} / {:0.2f}'.format( PSNR_T1 ,  PSNR_T2 ))
print('{:0.2f} / {:0.2f}'.format( SNR_T1 ,  SNR_T2 ))
print('{:0.2f} / {:0.2f}'.format( RMSE_T1 ,  RMSE_T2 ))
print('{:0.8f} / {:0.8f}'.format( coeff_T1, coeff_T2))



# save results
FileName = 'HYDRA_Test_2D_Anatomical_FullSample.npz'
np.savez(FileName,PSNR_T1 = PSNR_T1,PSNR_T2 = PSNR_T2,SNR_T1 = SNR_T1,SNR_T2 = SNR_T2,
         RMSE_T1 = RMSE_T1, RMSE_T2 = RMSE_T2, coeff_T1 = coeff_T1, coeff_T2 = coeff_T2,
         T1 = T1, T2 = T2, T1_true = T1_true,T2_true = T2_true, Tcost = Tcost)


# In[38]:

Results=np.load(FileName)
print('{:0.2f} / {:0.2f}'.format(Results['PSNR_T1'], Results['PSNR_T2']))
print('{:0.2f} / {:0.2f}'.format(Results['SNR_T1'], Results['SNR_T2']))
print('{:0.2f} / {:0.2f}'.format(Results['RMSE_T1'], Results['RMSE_T2']))
print('{:0.8f} / {:0.8f}'.format(Results['coeff_T1'], Results['coeff_T2']))
#print(Results['val_loss'][-10:-1],Results['loss'][-10:-1])

T1 = Results['T1']
T2 = Results['T2']

T1 = np.squeeze(T1)
T2 = np.squeeze(T2)



FigNameT1 = "T1_CNN_FullSample.png"
FigNameT2 = "T2_CNN_FullSample.png"
FigNameT1res = "T1_res_CNN_FullSample.png"
FigNameT2res = "T2_res_CNN_FullSample.png"
FigNameT1corr = "T1_corr_CNN_FullSample.png"
FigNameT2corr = "T2_corr_CNN_FullSample.png"
FigNameT1error = "T1_error_CNN_FullSample.png"
FigNameT2error = "T2_error_CNN_FullSample.png"

#%%

mycmap = 'jet' # 'gray'
mydpi = 200
plt.figure()
plt.imshow(T1, cmap = mycmap)
plt.colorbar()
plt.clim(0,T1max)
plt.axis('off') 
#plt.title('T1_Rec')
plt.grid(True)
plt.savefig(FigNameT1,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

plt.figure()
plt.imshow(T2, cmap = mycmap)
plt.colorbar()
plt.clim(0,T2max)
plt.axis('off') 
#plt.title('T2_Rec')
plt.grid(True)
plt.savefig(FigNameT2,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


plt.figure()
plt.imshow(np.abs(T1_true-T1), cmap = mycmap)
plt.colorbar()
plt.clim(0,20)
plt.axis('off') 
#plt.title('T1_residual')
plt.grid(True)
plt.savefig(FigNameT1res,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

plt.figure()
plt.imshow(np.abs(T2_true-T2), cmap = mycmap)
plt.colorbar()
plt.clim(0,10)
plt.axis('off') 
#plt.title('T2_residual')
plt.grid(True)
plt.savefig(FigNameT2res,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


#%% show correlation coefficients

ind_T1 = np.argsort(T1_true.flatten())
temp = T1_true.flatten();
Label_T1 = temp[ind_T1]
temp = T1.flatten();
predictions_T1 = temp[ind_T1]

ind_T2 = np.argsort(T2_true.flatten())
temp = T2_true.flatten();
Label_T2 = temp[ind_T2]
temp = T2.flatten();
predictions_T2 = temp[ind_T2]

#%% 
plt.figure(figsize = (3,3))
plt.plot(Label_T1, predictions_T1,'r.',label='Estimation')
plt.plot(Label_T1, Label_T1,'b-',label='Reference')
#plt.title('T1_Corr')
plt.grid(True)
plt.xlim((0, 5000))
plt.ylim((0, 5000))
plt.xlabel('Reference T1 (ms)')
plt.ylabel('Estimated T1 (ms)')
plt.legend(loc='best')
plt.savefig(FigNameT1corr,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

plt.figure(figsize = (3,3))
plt.plot(Label_T2, predictions_T2,'r.',label='Estimation')
plt.plot(Label_T2, Label_T2,'b-',label='Reference')
#plt.title('T2_Corr')
plt.grid(True)
plt.xlim((0, 2000))
plt.ylim((0, 2000))
plt.xlabel('Reference T2 (ms)')
plt.ylabel('Estimated T2 (ms)')
plt.legend(loc='best')
plt.savefig(FigNameT2corr,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


#%%
# show error maps

mydpi = 200

plt.figure(figsize = (3,3))
plt.plot(Label_T1, predictions_T1-Label_T1,'r.',label='Estimation')
plt.grid(True)
plt.xlim((0, 5000))
plt.ylim((-100, 100))
plt.xlabel('Reference T1 (ms)')
plt.ylabel('Error of estimated T1 (ms)')
#plt.legend(loc='best')
plt.savefig(FigNameT1error,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


plt.figure(figsize = (3,3))
plt.plot(Label_T2, predictions_T2-Label_T2,'r.',label='Estimation')
plt.grid(True)
plt.xlim((0, 2000))
plt.ylim((-40, 40))
plt.xlabel('Reference T2 (ms)')
plt.ylabel('Error of estimated T2 (ms)')
#plt.legend(loc='best')
plt.savefig(FigNameT2error,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()




#%%




# In[ ]:
# case 3: Testing on the anatomical dataset with k-space subsampling factor 15% using Gaussian patterns. 
       

MRFData = scipy.io.loadmat('Groundtruth_T1_T2.mat')
print(MRFData.keys())
T1_true = MRFData['T1_128']
T1_true = T1_true[:,:,np.newaxis]
T2_true = MRFData['T2_128']
T2_true = T2_true[:,:,np.newaxis]
print(T1_true.shape, T2_true.shape)
Label = np.concatenate([T1_true, T2_true], axis=2)
Label = Label.reshape((128*128,-1))
print(Label[0:16000:1000].T)
print(Label.shape, Label.dtype)

MRFData_Est = scipy.io.loadmat('X_FLOR_Gaussian_Ratio0_15_L1000.mat') # Gaussian pattern
print(MRFData_Est.keys())
X = MRFData_Est['X_estimated_flor']


print(X.shape, X.dtype)
X = X.reshape((128*128,-1))
X = np.real(X)

# remove those signature with too small value
NormX = np.zeros(X.shape[0])
NormX_index = np.empty(X.shape[0]) # index of valid values
NormX_index[:] = np.nan
print(NormX.shape, NormX_index.shape)
for i in range(0, X.shape[0]):
    NormX[i] = np.sum(X[i,:]**2)
    if NormX[i] < 10: #8: #10: # 1: #20: 125
        X[i,:] = 0
        NormX_index[i] = i

np.set_printoptions(precision=2)
NormX_index = NormX_index[~np.isnan(NormX_index)]
NormX_index = NormX_index.astype('int32') # arrays used as indices must be of integer (or boolean) type
X = normalize(X, norm = 'l2', axis=1)# L2 normalization along time dimention
X = np.expand_dims(X, axis=2)
print(X.shape, X.dtype)


# In[23]:

plt.figure()
Xpart = X[300:16000:2000,:,0]
print(Xpart.shape)
plt.plot(np.real(np.transpose(Xpart)))
plt.show()


#%%
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
#model.load_weights("model.h5") # load saved weights in final epoch.
model.load_weights("weights.best.hdf5") # load saved weights from the checkpoint.
print("Loaded model from disk")


# In[31]:


# calculate predictions
Tstart = time.clock()
predictions = model.predict(X)
Tend = time.clock()
Tcost = Tend - Tstart

predictions[NormX_index,:] = 0
print(predictions.shape)
print(predictions[0:200:10,:].T)
print(Label[0:200:10,:].T)

predictions = predictions.reshape((128,128,2))


# In[35]:


T1max = 4500
T2max = 2500

T1_true = np.squeeze(T1_true)
T2_true = np.squeeze(T2_true)

T1 = predictions[:,:,0]
T1[np.where((T1<0))] = 0
T1[np.where((T1>T1max))] = T1max
T2 = predictions[:,:,1]
T2[np.where((T2<0))] = 0
T2[np.where((T2>T2max))] = T2max

# remove invalid elements referring to the label.
T1 = T1 * (T1_true > 0)
T2 = T2 * (T2_true > 0)

print(T1_true.shape, T1.shape)

PSNR_T1 = psnr(T1,T1_true,T1max)
PSNR_T2 = psnr(T2,T2_true,T2max)

SNR_T1 = snr(T1,T1_true)
SNR_T2 = snr(T2,T2_true)

RMSE_T1 = rmse(T1,T1_true)
RMSE_T2 = rmse(T2,T2_true)

MRE_T1 = mre(T1,T1_true)
MRE_T2 = mre(T2,T2_true)


# compute correlation coefficients
Label = Label.reshape((128*128,-1))
T1 = T1[:,:,np.newaxis]
T2 = T2[:,:,np.newaxis]
predictions = np.concatenate([T1, T2], axis=2)
predictions = predictions.reshape((128*128,-1))

coeff_T1 = np.corrcoef(Label[:,0],predictions[:,0])
coeff_T1 = coeff_T1[0,1]
coeff_T2 = np.corrcoef(Label[:,1],predictions[:,1])
coeff_T2 = coeff_T2[0,1]


print('{:0.2f} / {:0.2f}'.format( PSNR_T1 ,  PSNR_T2 ))
print('{:0.2f} / {:0.2f}'.format( SNR_T1 ,  SNR_T2 ))
print('{:0.2f} / {:0.2f}'.format( RMSE_T1 ,  RMSE_T2 ))
print('{:0.8f} / {:0.8f}'.format( coeff_T1, coeff_T2))
print('{:0.2f} / {:0.2f}'.format( MRE_T1 ,  MRE_T2 ))


# save results
FileName = 'HYDRA_Test_2D_Anatomical_SubSample.npz'
np.savez(FileName,PSNR_T1 = PSNR_T1,PSNR_T2 = PSNR_T2,SNR_T1 = SNR_T1,SNR_T2 = SNR_T2,
         RMSE_T1 = RMSE_T1, RMSE_T2 = RMSE_T2, coeff_T1 = coeff_T1, coeff_T2 = coeff_T2,
         T1 = T1, T2 = T2, T1_true = T1_true,T2_true = T2_true, Tcost = Tcost)


# In[38]:

Results=np.load(FileName)
print('{:0.2f} / {:0.2f}'.format(Results['PSNR_T1'], Results['PSNR_T2']))
print('{:0.2f} / {:0.2f}'.format(Results['SNR_T1'], Results['SNR_T2']))
print('{:0.2f} / {:0.2f}'.format(Results['RMSE_T1'], Results['RMSE_T2']))
print('{:0.8f} / {:0.8f}'.format(Results['coeff_T1'], Results['coeff_T2']))
#print(Results['val_loss'][-10:-1],Results['loss'][-10:-1])

T1 = Results['T1']
T2 = Results['T2']

T1 = np.squeeze(T1)
T2 = np.squeeze(T2)



FigNameT1 = "T1_CNN_SubSample.png"
FigNameT2 = "T2_CNN_SubSample.png"
FigNameT1res = "T1_res_CNN_SubSample.png"
FigNameT2res = "T2_res_CNN_SubSample.png"
FigNameT1corr = "T1_corr_CNN_SubSample.png"
FigNameT2corr = "T2_corr_CNN_SubSample.png"
FigNameT1error = "T1_error_CNN_SubSample.png"
FigNameT2error = "T2_error_CNN_SubSample.png"

# In[36]:

mycmap = 'jet' # 'gray'
mydpi = 200
plt.figure()
plt.imshow(T1, cmap = mycmap)
plt.colorbar()
plt.clim(0,T1max)
plt.axis('off') 
#plt.title('T1_Rec')
plt.grid(True)
plt.savefig(FigNameT1,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

plt.figure()
plt.imshow(T2, cmap = mycmap)
plt.colorbar()
plt.clim(0,T2max)
plt.axis('off') 
#plt.title('T2_Rec')
plt.grid(True)
plt.savefig(FigNameT2,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


plt.figure()
plt.imshow(np.abs(T1_true-T1), cmap = mycmap)
plt.colorbar()
plt.clim(0,200)
plt.axis('off') 
#plt.title('T1_residual')
plt.grid(True)
plt.savefig(FigNameT1res,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

plt.figure()
plt.imshow(np.abs(T2_true-T2), cmap = mycmap)
plt.colorbar()
plt.clim(0,100)
plt.axis('off') 
#plt.title('T2_residual')
plt.grid(True)
plt.savefig(FigNameT2res,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

#%% show correlation coefficients

ind_T1 = np.argsort(T1_true.flatten())
temp = T1_true.flatten();
Label_T1 = temp[ind_T1]
temp = T1.flatten();
predictions_T1 = temp[ind_T1]

ind_T2 = np.argsort(T2_true.flatten())
temp = T2_true.flatten();
Label_T2 = temp[ind_T2]
temp = T2.flatten();
predictions_T2 = temp[ind_T2]

#%% 
plt.figure(figsize = (3,3))
plt.plot(Label_T1, predictions_T1,'r.',label='Estimation')
plt.plot(Label_T1, Label_T1,'b-',label='Reference')
#plt.title('T1_Corr')
plt.grid(True)
plt.xlim((0, 5000))
plt.ylim((0, 5000))
plt.xlabel('Reference T1 (ms)')
plt.ylabel('Estimated T1 (ms)')
plt.legend(loc='best')
plt.savefig(FigNameT1corr,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

plt.figure(figsize = (3,3))
plt.plot(Label_T2, predictions_T2,'r.',label='Estimation')
plt.plot(Label_T2, Label_T2,'b-',label='Reference')
#plt.title('T2_Corr')
plt.grid(True)
plt.xlim((0, 2000))
plt.ylim((0, 2000))
plt.xlabel('Reference T2 (ms)')
plt.ylabel('Estimated T2 (ms)')
plt.legend(loc='best')
plt.savefig(FigNameT2corr,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


#%%
# show error maps

plt.figure(figsize = (3,3))
plt.plot(Label_T1, predictions_T1-Label_T1,'r.',label='Estimation')
plt.grid(True)
plt.xlim((0, 5000))
plt.ylim((-100, 100))
plt.xlabel('Reference T1 (ms)')
plt.ylabel('Error of estimated T1 (ms)')
#plt.legend(loc='best')
plt.savefig(FigNameT1error,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


plt.figure(figsize = (3,3))
plt.plot(Label_T2, predictions_T2-Label_T2,'r.',label='Estimation')
plt.grid(True)
plt.xlim((0, 2000))
plt.ylim((-40, 40))
plt.xlabel('Reference T2 (ms)')
plt.ylabel('Error of estimated T2 (ms)')
#plt.legend(loc='best')
plt.savefig(FigNameT2error,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


# In[37]:





# In[ ]:

# case 4: Testing on the anatomical dataset with k-space subsampling factor 9% using Spiral patterns. 

MRFData = scipy.io.loadmat('Groundtruth_T1_T2.mat')
print(MRFData.keys())
T1_true = MRFData['T1_128']
T1_true = T1_true[:,:,np.newaxis]
T2_true = MRFData['T2_128']
T2_true = T2_true[:,:,np.newaxis]
print(T1_true.shape, T2_true.shape)
Label = np.concatenate([T1_true, T2_true], axis=2)
Label = Label.reshape((128*128,-1))
print(Label[0:16000:1000].T)
print(Label.shape, Label.dtype)

MRFData_Est = scipy.io.loadmat('X_FLOR_Spiral_Ratio0_09_L1000.mat') # Spiral pattern
print(MRFData_Est.keys())
X = MRFData_Est['X_estimated_flor']


print(X.shape, X.dtype)
X = X.reshape((128*128,-1))
X = np.real(X)

# remove those signature with too small value
NormX = np.zeros(X.shape[0])
NormX_index = np.empty(X.shape[0]) # index of valid values
NormX_index[:] = np.nan
print(NormX.shape, NormX_index.shape)
for i in range(0, X.shape[0]):
    NormX[i] = np.sum(X[i,:]**2)
    if NormX[i] < 150: #8: #10: # 1: #20: 125
        X[i,:] = 0
        NormX_index[i] = i

np.set_printoptions(precision=2)
NormX_index = NormX_index[~np.isnan(NormX_index)]
NormX_index = NormX_index.astype('int32') # arrays used as indices must be of integer (or boolean) type
X = normalize(X, norm = 'l2', axis=1)# L2 normalization along time dimention
X = np.expand_dims(X, axis=2)
print(X.shape, X.dtype)


# In[23]:

plt.figure()
Xpart = X[300:16000:2000,:,0]
print(Xpart.shape)
plt.plot(np.real(np.transpose(Xpart)))
plt.show()


#%%
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
#model.load_weights("model.h5") # load saved weights in final epoch.
model.load_weights("weights.best.hdf5") # load saved weights from the checkpoint.
print("Loaded model from disk")


# In[31]:


# calculate predictions
Tstart = time.clock()
predictions = model.predict(X)
Tend = time.clock()
Tcost = Tend - Tstart

predictions[NormX_index,:] = 0
print(predictions.shape)
print(predictions[0:200:10,:].T)
print(Label[0:200:10,:].T)

predictions = predictions.reshape((128,128,2))


# In[35]:


T1max = 4500
T2max = 2500

T1_true = np.squeeze(T1_true)
T2_true = np.squeeze(T2_true)

T1 = predictions[:,:,0]
T1[np.where((T1<0))] = 0
T1[np.where((T1>T1max))] = T1max
T2 = predictions[:,:,1]
T2[np.where((T2<0))] = 0
T2[np.where((T2>T2max))] = T2max

# remove invalid elements referring to the label.
T1 = T1 * (T1_true > 0)
T2 = T2 * (T2_true > 0)

print(T1_true.shape, T1.shape)

PSNR_T1 = psnr(T1,T1_true,T1max)
PSNR_T2 = psnr(T2,T2_true,T2max)

SNR_T1 = snr(T1,T1_true)
SNR_T2 = snr(T2,T2_true)

RMSE_T1 = rmse(T1,T1_true)
RMSE_T2 = rmse(T2,T2_true)

MRE_T1 = mre(T1,T1_true)
MRE_T2 = mre(T2,T2_true)


# compute correlation coefficients
Label = Label.reshape((128*128,-1))
T1 = T1[:,:,np.newaxis]
T2 = T2[:,:,np.newaxis]
predictions = np.concatenate([T1, T2], axis=2)
predictions = predictions.reshape((128*128,-1))

coeff_T1 = np.corrcoef(Label[:,0],predictions[:,0])
coeff_T1 = coeff_T1[0,1]
coeff_T2 = np.corrcoef(Label[:,1],predictions[:,1])
coeff_T2 = coeff_T2[0,1]


print('{:0.2f} / {:0.2f}'.format( PSNR_T1 ,  PSNR_T2 ))
print('{:0.2f} / {:0.2f}'.format( SNR_T1 ,  SNR_T2 ))
print('{:0.2f} / {:0.2f}'.format( RMSE_T1 ,  RMSE_T2 ))
print('{:0.8f} / {:0.8f}'.format( coeff_T1, coeff_T2))
print('{:0.2f} / {:0.2f}'.format( MRE_T1 ,  MRE_T2 ))


# save results
FileName = 'HYDRA_Test_2D_Anatomical_SpiralSubSample.npz'
np.savez(FileName,PSNR_T1 = PSNR_T1,PSNR_T2 = PSNR_T2,SNR_T1 = SNR_T1,SNR_T2 = SNR_T2,
         RMSE_T1 = RMSE_T1, RMSE_T2 = RMSE_T2, coeff_T1 = coeff_T1, coeff_T2 = coeff_T2,
         T1 = T1, T2 = T2, T1_true = T1_true,T2_true = T2_true, Tcost = Tcost)


# In[38]:

Results=np.load(FileName)
print('{:0.2f} / {:0.2f}'.format(Results['PSNR_T1'], Results['PSNR_T2']))
print('{:0.2f} / {:0.2f}'.format(Results['SNR_T1'], Results['SNR_T2']))
print('{:0.2f} / {:0.2f}'.format(Results['RMSE_T1'], Results['RMSE_T2']))
print('{:0.8f} / {:0.8f}'.format(Results['coeff_T1'], Results['coeff_T2']))
#print(Results['val_loss'][-10:-1],Results['loss'][-10:-1])

T1 = Results['T1']
T2 = Results['T2']

T1 = np.squeeze(T1)
T2 = np.squeeze(T2)



FigNameT1 = "T1_CNN_SpiralSubSample.png"
FigNameT2 = "T2_CNN_SpiralSubSample.png"
FigNameT1res = "T1_res_CNN_SpiralSubSample.png"
FigNameT2res = "T2_res_CNN_SpiralSubSample.png"
FigNameT1corr = "T1_corr_CNN_SpiralSubSample.png"
FigNameT2corr = "T2_corr_CNN_SpiralSubSample.png"
FigNameT1error = "T1_error_CNN_SpiralSubSample.png"
FigNameT2error = "T2_error_CNN_SpiralSubSample.png"

# In[36]:

mycmap = 'jet' # 'gray'
mydpi = 200
plt.figure()
plt.imshow(T1, cmap = mycmap)
plt.colorbar()
plt.clim(0,T1max)
plt.axis('off') 
#plt.title('T1_Rec')
plt.grid(True)
plt.savefig(FigNameT1,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

plt.figure()
plt.imshow(T2, cmap = mycmap)
plt.colorbar()
plt.clim(0,T2max)
plt.axis('off') 
#plt.title('T2_Rec')
plt.grid(True)
plt.savefig(FigNameT2,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


plt.figure()
plt.imshow(np.abs(T1_true-T1), cmap = mycmap)
plt.colorbar()
plt.clim(0,200)
plt.axis('off') 
#plt.title('T1_residual')
plt.grid(True)
plt.savefig(FigNameT1res,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

plt.figure()
plt.imshow(np.abs(T2_true-T2), cmap = mycmap)
plt.colorbar()
plt.clim(0,100)
plt.axis('off') 
#plt.title('T2_residual')
plt.grid(True)
plt.savefig(FigNameT2res,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

#%% show correlation coefficients

ind_T1 = np.argsort(T1_true.flatten())
temp = T1_true.flatten();
Label_T1 = temp[ind_T1]
temp = T1.flatten();
predictions_T1 = temp[ind_T1]

ind_T2 = np.argsort(T2_true.flatten())
temp = T2_true.flatten();
Label_T2 = temp[ind_T2]
temp = T2.flatten();
predictions_T2 = temp[ind_T2]

#%% 
plt.figure(figsize = (3,3))
plt.plot(Label_T1, predictions_T1,'r.',label='Estimation')
plt.plot(Label_T1, Label_T1,'b-',label='Reference')
#plt.title('T1_Corr')
plt.grid(True)
plt.xlim((0, 5000))
plt.ylim((0, 5000))
plt.xlabel('Reference T1 (ms)')
plt.ylabel('Estimated T1 (ms)')
plt.legend(loc='best')
plt.savefig(FigNameT1corr,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

plt.figure(figsize = (3,3))
plt.plot(Label_T2, predictions_T2,'r.',label='Estimation')
plt.plot(Label_T2, Label_T2,'b-',label='Reference')
#plt.title('T2_Corr')
plt.grid(True)
plt.xlim((0, 2000))
plt.ylim((0, 2000))
plt.xlabel('Reference T2 (ms)')
plt.ylabel('Estimated T2 (ms)')
plt.legend(loc='best')
plt.savefig(FigNameT2corr,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


#%%
# show error maps

plt.figure(figsize = (3,3))
plt.plot(Label_T1, predictions_T1-Label_T1,'r.',label='Estimation')
plt.grid(True)
plt.xlim((0, 5000))
plt.ylim((-100, 100))
plt.xlabel('Reference T1 (ms)')
plt.ylabel('Error of estimated T1 (ms)')
#plt.legend(loc='best')
plt.savefig(FigNameT1error,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


plt.figure(figsize = (3,3))
plt.plot(Label_T2, predictions_T2-Label_T2,'r.',label='Estimation')
plt.grid(True)
plt.xlim((0, 2000))
plt.ylim((-40, 40))
plt.xlabel('Reference T2 (ms)')
plt.ylabel('Error of estimated T2 (ms)')
#plt.legend(loc='best')
plt.savefig(FigNameT2error,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()


# In[37]:

