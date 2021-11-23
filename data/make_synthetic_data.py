
import numpy as np
import pandas as pd
import time

import random
import math

import multiprocessing
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from scipy import interpolate

from scipy import interpolate
# We'll use double precision throughout for better numerics.
dtype = np.float64

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


obs_error = 0.005
repeat = 4

tf.random.set_seed(repeat)
np.random.seed(repeat)


def make_functions():
    start = 0.
    end = 24.0*60
    N = 1000# 10,000 points per individual

    Tlen = np.sort(np.random.uniform(start-60,end+60,size=N))

    KK = tfk.ExponentiatedQuadratic(amplitude=0.5,length_scale=np.float64(60.0)).matrix(Tlen[...,None],Tlen[...,None])

    I = 1e-8 * tf.eye(N, dtype=tf.float64) #NN
    KK = KK +I 
    LL=tf.linalg.cholesky(KK)

    mvn = tfd.MultivariateNormalLinearOperator(
        loc=tf.zeros_like(Tlen[None,...]),
        scale=tf.linalg.LinearOperatorLowerTriangular(LL))

    meanl = 0
    fun_len_vals = np.squeeze(tf.math.softplus(meanl + mvn.sample()).numpy()[0])#/tf.math.softplus(0.0).numpy()
    func_len = interpolate.interp1d(Tlen, fun_len_vals)


    start = 0.
    end = 24.0*60
    N = 1000# 10,000 points per individual

    Tamp = np.sort(np.random.uniform(start-60,end+60,size=N))

    KK = tfk.ExponentiatedQuadratic(length_scale=np.float64(60.0)).matrix(Tamp[...,None],Tamp[...,None])

    I = 1e-8 * tf.eye(N, dtype=tf.float64) #NN
    KK = KK +I 
    LL=tf.linalg.cholesky(KK)

    mvn = tfd.MultivariateNormalLinearOperator(
        loc=tf.zeros_like(Tamp[None,...]),
        scale=tf.linalg.LinearOperatorLowerTriangular(LL))


    fun_amp_vals = np.squeeze(tf.math.softplus(mvn.sample()).numpy()[0])

    func_amp = interpolate.interp1d(Tamp, fun_amp_vals)
    
    return func_len, func_amp


def non_stat_matern12( X, lengthscales, stddev):
    ''' Non-stationary Matern 12 kernel'''

    Xs = tf.reduce_sum(input_tensor=tf.square(X), axis=-1, keepdims=True)#(1000,1)
    Ls = tf.square(lengthscales)#(1,1000,1)

    dist = -2 * tf.matmul(X, X, transpose_b=True)
    dist += Xs + tf.linalg.matrix_transpose(Xs)
    Lscale = Ls + tf.linalg.matrix_transpose(Ls)
    dist = tf.divide(2*dist,Lscale)
    dist = tf.sqrt(tf.maximum(dist, 1e-40))
    prefactL = 2 * tf.matmul(lengthscales, lengthscales, transpose_b=True)
    prefactV = tf.matmul(stddev, stddev, transpose_b=True)

    return tf.multiply(prefactV,tf.multiply( tf.sqrt(tf.maximum(tf.divide(prefactL,Lscale), 1e-40)),tf.exp(-dist)))
    

def make_ts(func_len, func_amp, start, end, N):

    T = np.sort(np.random.uniform(start,end,size=N))
    # generate random times between 0 and 24 and sort them in ascending order
    # we record observations of different individuals at different time points     
    L = func_len(T)
    sigma = func_amp(T)


    KK=non_stat_matern12(T[None,...,None],L[None,...,None],sigma[None,...,None])
    I = obs_error**2 * tf.eye(N, dtype=tf.float64) #NN
    KK = KK +I 
            
    jitter=1e-4

    while True:
        try:
            LL=tf.linalg.cholesky(KK)
            break
        except:
            print('adding jitter...')
            I = jitter * tf.eye(N, dtype=tf.float64) #NN
            KK = KK +I 
            jitter*=10
    
    mvn = tfd.MultivariateNormalLinearOperator(
        loc=tf.zeros_like(T[None,...]),
        scale=tf.linalg.LinearOperatorLowerTriangular(LL))
    sample = mvn.sample().numpy()[0]

    return sample, T, L, sigma
        
def make_repeat(repeat):

    n_list = [1,8,128]

    func_len, func_amp = make_functions()
    for n_indv in n_list:
        start = 0.
        end = 24.0*60
        N = 8192

        Y =[]
        T_index = []
        L_all =[]
        var_all=[]

        for i in tqdm(range(n_indv)):
            
            sample, T, L, sigma = make_ts(func_len, func_amp, start, end, N)
            # collect the data and all the parameters
            Y.append(sample)
            T_index.append(T)
            L_all.append(L)
            var_all.append(sigma)
            

        y= np.array(Y).reshape(N*n_indv,1)

        T_total = np.array(T_index).reshape(N*n_indv,1)
        len_total = np.array(L_all).reshape(N*n_indv,1)
        var_total = np.array(var_all).reshape(N*n_indv,1)
        dataset = pd.DataFrame({'Time':T_total.flatten(), 'observations':y.flatten(),'Lengthscale':len_total.flatten(),'Variance':var_total.flatten(),'ID':0})


        # set the IDs according to the batches
        nb = n_indv
        npb = N

        for j in tqdm(range(nb)):
            dataset['ID'].iloc[j*npb:(j+1)*npb]= j

        dataset.to_csv('ns_synthetic_data_indv_' + str(n_indv) + '_' + str(repeat) + '.csv')


# simulate TS data for multiple individuals
make_repeat(repeat)


