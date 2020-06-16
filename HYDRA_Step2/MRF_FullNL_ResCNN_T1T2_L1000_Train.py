
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


# In[4]:

MRFData = scipy.io.loadmat('D_LUT_L1000_TE10_Train.mat') 
Label = MRFData['LUT']
print(Label.shape, Label.dtype)
X = MRFData['D']
X = np.real(X) #X = np.abs(X)
X = X[:,0::1] # fully-sampled from 1000 time points;
X = normalize(X, norm = 'l2', axis=1)# L2 normalization along time dimention
X = np.expand_dims(X, axis=2)

np.set_printoptions(precision=1)
np.set_printoptions(suppress=True) #suppress the use of scientific notation for small numbers:
print(X.shape,X.dtype)
print(Label[0:16000:1000].T)

#validation data
MRFData_Val = scipy.io.loadmat('D_LUT_L1000_TE10_Validation.mat') #
val_Label = MRFData_Val['LUT']
val_X = MRFData_Val['D']
#val_X = val_X[:,0::5] # sub-sampled from 1000 time points;
val_X = val_X[:,0::1] # fully-sampled from 1000 time points;
val_X = normalize(val_X, norm = 'l2', axis=1)# L2 normalization along time dimention
val_X = np.expand_dims(val_X, axis=2)

val_data = (val_X, val_Label)

#%% use more training dataset
X = np.concatenate([X, val_X], axis=0)
Label = np.concatenate([Label,val_Label], axis=0)
print(X.shape,X.dtype)
print(Label.shape, Label.dtype)

del MRFData, MRFData_Val, val_X, val_Label
# In[6]:


plt.figure()
Xpart = X[0:800:100,:,0]
print(Xpart.shape)
plt.plot(np.real(np.transpose(Xpart)))
plt.figure()
plt.plot(np.real(np.transpose(Label[0:80000:333,0])))
plt.figure()
plt.plot(np.real(np.transpose(Label[0:80000:333,1])))


# In[7]:


def lr_schedule0(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 10, 20, 30, 40 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-1
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced by half after every 20 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr_init = 1e-3
    if epoch == 0:
        K.set_value(model.optimizer.lr, lr_init)
    elif epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

# In[10]:

# define resnet
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 name=''):
    """1D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input 1D signal
        num_filters (int): Conv1D number of filters
        kernel_size (int): Conv1D kernel dimensions
        strides (int): Conv1D stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or 
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4),
                  name = name)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else: # full pre-activation
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_MRF():
    # create model CNN
    """ResNet Version 1 Model builder
    Stacks of BN-ReLU-Conv1D
    # Arguments
        input_shape (tensor): shape of input image tensor, e.g. 200x1
        depth (int): number of core convolutional layers, 6n+2, e.g. 20, 32, 44 
        num_classes (int): number of classes, e.g.2 types of tissue parameters, 
        T1 and T2)
    # Returns
        model (Model): Keras model instance
    """
    
    seq_length = 1000 
    inputs = Input(shape=(seq_length,1))
    
    x = Conv1D(16,21,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(1e-4),
               name='block1_conv1')(inputs)  
    x = Activation('relu')(x)
   
    x = Conv1D(16,21,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(1e-4),
               name='block1_conv2')(x)
#    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = non_local_block(x, compression=1, mode='embedded') # mode: `embedded`, `gaussian`, `dot` or `concatenate`.
        
    x = MaxPooling1D(2)(x)
    x = resnet_layer(inputs=x,num_filters=32,kernel_size=1,batch_normalization=False,name='block2_conv1')  
    y = resnet_layer(inputs=x,num_filters=32,kernel_size=21,batch_normalization=False,name='block2_conv2')
    x = add([x, y])
    x = non_local_block(x, compression=1, mode='embedded') # mode: `embedded`, `gaussian`, `dot` or `concatenate`.
  
    
    x = MaxPooling1D(2)(x)
    x = resnet_layer(inputs=x,num_filters=64,kernel_size=1,batch_normalization=False,name='block3_conv1')  
    y = resnet_layer(inputs=x,num_filters=64,kernel_size=21,batch_normalization=False,name='block3_conv2')
    x = add([x, y])
    x = non_local_block(x, compression=1, mode='embedded') # mode: `embedded`, `gaussian`, `dot` or `concatenate`.
     
    x = MaxPooling1D(2)(x)
    x = resnet_layer(inputs=x,num_filters=128,kernel_size=1,batch_normalization=False,name='block4_conv1')  
    y = resnet_layer(inputs=x,num_filters=128,kernel_size=21,batch_normalization=False,name='block4_conv2')
    x = add([x, y])
    x = non_local_block(x, compression=2, mode='embedded') # mode: `embedded`, `gaussian`, `dot` or `concatenate`.
    
    
    x = GlobalAveragePooling1D()(x)
	
    outputs = Dense(2, 
                    kernel_initializer='he_normal',
                    kernel_constraint=maxnorm(3))(x)
    
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
        
    return model

#%%
model = resnet_MRF()

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-3),              
#              optimizer=Adam(lr=lr_schedule),
              metrics=['mean_absolute_error'])
model.summary()


# In[11]:

if os.path.exists('weights.best.hdf5'):
    print("Loading saved weights ...")
    model.load_weights('weights.best.hdf5')
            
# Checkpoint the weights when validation accuracy improves
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

reduce_lr = LearningRateScheduler(lr_schedule)
callbacks_list = [checkpoint, reduce_lr]

# Fit the model
History = model.fit(X, Label, validation_split=0.2, epochs=50, batch_size=50, callbacks=callbacks_list, verbose=1, shuffle=True,)
#History = model.fit(X, Label, validation_data=val_data, epochs=50, batch_size=50, callbacks=callbacks_list, verbose=1, shuffle=True,)

plt.figure(3)
plt.plot(History.history['val_loss'])


# In[12]:

# evaluate the model
model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4), metrics=['mean_absolute_error'])
scores = model.evaluate(X, Label, verbose=0)
print("\n%s: %.2f" % (model.metrics_names[1], scores[1]))


# In[13]:

# calculate predictions
Tstart = time.clock()
predictions = model.predict(X)
Tend = time.clock()
Tcost = Tend - Tstart

print(predictions.shape, Label.shape)
print(predictions[0:8000:1000,:].T)
print(Label[0:8000:1000,:].T)

# In[15]:

mydpi = 200

plt.figure()
plt.plot(predictions[:80000:1000,0],'r-o')
plt.plot(Label[:80000:1000,0],'b-*')
plt.title('T1_acc')
plt.grid(True)
plt.savefig("T1_acc.png",bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

plt.figure()
plt.plot(np.sort(predictions[:80000:2000,1]),'r-o')
plt.plot(np.sort(Label[:80000:2000,1]),'b-*')
plt.title('T2_acc')
plt.grid(True)
plt.savefig("T2_acc.png",bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()

# In[16]:

plt.figure()
plt.plot(predictions[:1000:100,0],'r-o')
plt.plot(Label[:1000:100,0],'b-*')
plt.figure()
plt.plot(np.sort(predictions[:1000:100,1]),'r-o')
plt.plot(np.sort(Label[:1000:100,1]),'b-*')
plt.show()


# In[17]:

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
# export pickle
with open('History.pickle', 'wb') as f:
    pickle.dump(History.history, f)


# In[18]:

print(History.history['val_loss'][-1])
print(History.history['loss'][-1])
val_loss_his = History.history['val_loss']
loss_his = History.history['loss']
plt.figure()
plt.plot(val_loss_his[1:],'r-o', label='val_loss')
plt.plot(loss_his[1:],'b-*', label='train_loss')
plt.legend(loc='best')
plt.ylim((0,20))
plt.xlabel('epochs')
plt.ylabel('MSE')
#plt.title('Train_Val_Curve')
plt.grid(True)
plt.savefig("Train_Val_Curve.png",bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
plt.show()
print(val_loss_his[-1])
print(loss_his[-1])


# In[19]:

## later...
## load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
## load weights into new model
## model.load_weights("model.h5") # load saved weights in final epoch.
#model.load_weights("weights.best.hdf5") # load saved weights from the checkpoint.
#print("Loaded model from disk")
#
#with open('History.pickle', 'rb') as f:
##     History.history = {'val_loss': [0] }
#    history = pickle.load(f)
#    print('Load and show saved training history')
#    print(history.keys())
#
#val_loss_his = history['val_loss']
#loss_his = history['loss']
#plt.figure()
#plt.plot(val_loss_his[1:],'r-o', label='val_loss')
#plt.plot(loss_his[1:],'b-*', label='train_loss')
#plt.legend(loc='best')
#plt.ylim((0,20))
#plt.xlabel('epochs')
#plt.ylabel('MSE')
##plt.title('Train_Val_Curve')
#plt.grid(True)
#plt.savefig("Train_Val_Curve.png",bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
#plt.show()
#print(history['val_loss'][-1])
#print(history['loss'][-1])
#
## evaluate loaded model on test data
#model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4), metrics=['mean_absolute_error'])
#scores = model.evaluate(X, Label, verbose=0)
#print("\n%s: %.2f" % (model.metrics_names[1], scores[1]))





