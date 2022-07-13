import numpy as np
import pandas as pd
import time

import random

import matplotlib
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

import pickle

#import matplotlib.ticker as tick

import sys
sys.path.append('..')

from nsgp_vi import nsgpVI

# We'll use double precision throughout for better numerics.
dtype = np.float64

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels


n_list = [1,8,128]


def run_training(N,nrd):
    print('Starting opt dataset repeat: ' + str(nrd) + ', N: ' + str(N))
    df = pd.read_csv('../data/ns_synthetic_data_indv_' + str(N) + '_' + str(nrd) +'.csv')

    T = df['Time'].values[:,None]
    ID = df['ID'].values
    X = np.array(df['observations']).reshape(len(T),1)
    true_len = np.array(df['Lengthscale']).reshape(len(T),1)
    true_var = np.array(df['Variance']).reshape(len(T),1)
    
    num_training_points_ = T.shape[0]
    num_inducing_points_ = 50
    inducing_index_points = np.linspace(0., 60*24., num_inducing_points_, endpoint=False)[..., np.newaxis]
    np.random.shuffle(inducing_index_points)
    
    BATCH_SIZE=8
    SEG_LENGTH=1024
    allT = []
    allX = []
    
    for i in np.unique(df['ID'].values):
        allT.append(df['Time'][df['ID'].values == i].values[...,None])
        allX.append(df['observations'][df['ID'].values == i].values[...,None])

    class segment_generator:
        def __iter__(self):
            # loop over individuals
            self.i = 0
            self.max_i = len(allT)
            
            # loop over segments
            self.j = 0
            self.max_j = num_training_points_//(self.max_i*SEG_LENGTH)
        
            return self
        
        def __next__(self):
            
            if self.i == self.max_i:
                self.i = 0
                self.j +=1
                if self.j==self.max_j:
                    raise StopIteration
            
            T = allT[self.i]
            X = allX[self.i] 

            TT = T[self.j*SEG_LENGTH:(self.j+1)*SEG_LENGTH]
            XX = X[self.j*SEG_LENGTH:(self.j+1)*SEG_LENGTH]
    
            self.i += 1

            return TT,XX
        
    dataset = tf.data.Dataset.from_generator(segment_generator, (tf.float64)) 
    dataset = dataset.map(lambda dd: (dd[0],dd[1]))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    dataset = dataset.with_options(options)
    
    kernel_len_a = tfp.util.TransformedVariable(2.0, tfb.Softplus(),dtype=tf.float64, name='k_len_a',trainable=True)
    kernel_len_l = tfp.util.TransformedVariable(30.0,tfb.Chain([tfb.Scale(np.float64(30.)),tfb.Softplus()]),dtype=tf.float64, name='k_len_l',trainable=True)
    
    # amplitude kernel parameters, lower levels
    kernel_amp_a = tfp.util.TransformedVariable(2.0, tfb.Softplus(), dtype=tf.float64, name='k_amp_a',trainable=True)
    kernel_amp_l = tfp.util.TransformedVariable(30.0,tfb.Chain([tfb.Scale(np.float64(30.)),tfb.Softplus()]), dtype=tf.float64, name='k_amp_l',trainable=True)
    
    #kernels on the second layer
    kernel_len = tfk.ExponentiatedQuadratic(kernel_len_a,kernel_len_l)
    kernel_amp = tfk.ExponentiatedQuadratic(kernel_amp_a,kernel_amp_l)
    
    #print(str(nrd))

    vgp = nsgpVI(kernel_len,kernel_amp,n_inducing_points=num_inducing_points_,inducing_index_points=inducing_index_points,dataset=dataset,num_training_points=num_training_points_, num_sequential_samples=5,num_parallel_samples=10,init_observation_noise_variance=0.005**2,jitter=1e-4)  
    
    loss = vgp.optimize(BATCH_SIZE, SEG_LENGTH, NUM_EPOCHS=200)
    
    #ZZ = np.linspace(0,24*60,200)[:,None]
    
    #[len_mean,amp_mean], [len_var,amp_var] = vgp.get_conditional(ZZ[None,...],full_cov=False)

    #len_mean = len_mean[0,:,0].numpy()
    #len_std = len_var[:,0].numpy()**0.5
    
    #amp_mean = amp_mean[0,:,0].numpy()
    #amp_std = amp_var[:,0].numpy()**0.5
    
    #f, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6))
    #ax1.plot(ZZ,tf.math.softplus(vgp.mean_len + len_mean),color='C1')
    #ax1.fill_between(ZZ[:,0],tf.math.softplus(vgp.mean_len + len_mean - 1.96*len_std),tf.math.softplus(vgp.mean_len + len_mean + 1.96*len_std),color='C1',alpha=0.5)
    #ax1.plot(T[:8192],true_len[:8192],'--',markersize=1,color='k',alpha=0.5)
    
    #ax2.plot(T[:8192],true_var[:8192],'--',markersize=1,color='k',alpha=0.5)
    #ax2.plot(ZZ,tf.math.softplus(vgp.mean_amp + amp_mean),color='C1')
    #ax2.fill_between(ZZ[:,0],tf.math.softplus(vgp.mean_amp + amp_mean - 1.96*amp_std),tf.math.softplus(vgp.mean_amp + amp_mean + 1.96*amp_std),color='C1',alpha=0.5)
    
    #plt.savefig('../results/n' + str(N) + '_' + str(nrd) + '.png')
    
    outputvars = []
    
    for v in vgp.trainable_variables:
        outputvars.append(v.numpy())
    
    with open('../results/opt_n' + str(N) + '_' + str(nrd) + '.pkl', 'wb') as f:
        pickle.dump(outputvars, f)
    
    np.save('../results/T_ind_n' + str(N) + '_' + str(nrd) + '.npy',inducing_index_points)  
    
    
   
    
    



for N in n_list:
    for nrd in range(5):
        run_training(N,nrd)
        
      
