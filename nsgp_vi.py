"""
Copyright 2021 Colin Torney

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm

from tensorflow.python.eager import tape
from tensorflow_probability.python.distributions import kullback_leibler

from addons.gradient_accumulator import GradientAccumulator

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

dtype = np.float64

class nsgpVI(tf.Module):
                                        
    def __init__(self,kernel_len,kernel_amp,n_inducing_points,inducing_index_points,dataset,num_training_points, init_observation_noise_variance=1e-2,num_sequential_samples=10,num_parallel_samples=10,jitter=1e-6):
               
        self.jitter=jitter
        
        #mean for the latent parameters; miu_l,miu_sigma
        self.mean_len = tf.Variable([0.0], dtype=tf.float64, name='len_mean', trainable=False)
        self.mean_amp = tf.Variable([0.0], dtype=tf.float64, name='var_mean', trainable=False)
        
        self.kernel_len = kernel_len
        self.kernel_amp = kernel_amp
    
        self.amp_inducing_index_points = tf.Variable(inducing_index_points,dtype=dtype,name='amp_ind_points',trainable=False) #z's for amplitude
        self.len_inducing_index_points = tf.Variable(inducing_index_points,dtype=dtype,name='len_ind_points',trainable=False) #z's for len
        
        #parameters for variational distribution for len,phi(l_z)
        self.len_variational_inducing_observations_loc = tf.Variable(np.zeros((n_inducing_points),dtype=dtype),name='len_ind_loc_post')
        #self.len_variational_inducing_observations_scale = tf.Variable(np.eye(n_inducing_points, dtype=dtype),name='len_ind_scale_post')
        self.len_variational_inducing_observations_scale = tfp.util.TransformedVariable(0.1*np.eye(n_inducing_points, dtype=dtype),tfp.bijectors.FillScaleTriL(diag_shift=np.float64(1e-05)),dtype=tf.float64, name='len_ind_scale_post', trainable=True)

        #parameters for variational distribution for var,phi(sigma_z)
        self.amp_variational_inducing_observations_loc = tf.Variable(np.zeros((n_inducing_points), dtype=dtype),name='amp_ind_loc_post')
        #self.amp_variational_inducing_observations_scale = tf.Variable(np.eye(n_inducing_points, dtype=dtype),name='amp_ind_scale_post')
        self.amp_variational_inducing_observations_scale = tfp.util.TransformedVariable(0.1*np.eye(n_inducing_points, dtype=dtype),tfp.bijectors.FillScaleTriL(diag_shift=np.float64(1e-05)),dtype=tf.float64, name='amp_ind_scale_post', trainable=True)


        
        #approximation to the posterior: phi(l_z)
        self.len_variational_inducing_observations_posterior = tfd.MultivariateNormalLinearOperator(
                                                                      loc=self.len_variational_inducing_observations_loc,
                                                                      scale=tf.linalg.LinearOperatorLowerTriangular(self.len_variational_inducing_observations_scale))
        #approximation to the posterior:phi(sigma_z)
        self.amp_variational_inducing_observations_posterior = tfd.MultivariateNormalLinearOperator(
                                                                      loc=self.amp_variational_inducing_observations_loc,
                                                                      scale=tf.linalg.LinearOperatorLowerTriangular(self.amp_variational_inducing_observations_scale))

        #p(l_z)
        self.len_inducing_prior = tfd.MultivariateNormalDiag(loc=tf.zeros((n_inducing_points),dtype=tf.float64),name='len_ind_prior')
        
        #p(sigma_z)
        self.amp_inducing_prior = tfd.MultivariateNormalDiag(loc=tf.zeros((n_inducing_points),dtype=tf.float64),name='amp_ind_prior')


        self.vgp_observation_noise_variance = tf.Variable(np.log(np.exp(init_observation_noise_variance)-1),dtype=dtype,name='nv', trainable=False)

        self.num_sequential_samples=num_sequential_samples
        self.num_parallel_samples=num_parallel_samples
        
        self.dataset = dataset
        self.num_training_points=num_training_points
        
        #if velocity:
        #    self.log_likelihood_fn = self.log_likelihood_fn
        #else:
        #    self.log_likelihood_fn = self.log_likelihood_fn

    def optimize(self, BATCH_SIZE, SEG_LENGTH, NUM_EPOCHS=100, WARM_UP=10):


        initial_learning_rate = float(BATCH_SIZE*SEG_LENGTH)/float(self.num_training_points)

        steps_per_epoch = self.num_training_points//(BATCH_SIZE*SEG_LENGTH)
        learning_rate = tf.optimizers.schedules.ExponentialDecay(
	    initial_learning_rate=initial_learning_rate,
            decay_steps=steps_per_epoch,
            decay_rate=0.95,
            staircase=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        optimizer = GradientAccumulator(optimizer, accum_steps=self.num_sequential_samples, reduction='MEAN')


        self.len_variational_inducing_observations_scale.variables[0]._trainable=False
        self.amp_variational_inducing_observations_scale.variables[0]._trainable=False

        @tf.function
        def train_step(x_train_batch, y_train_batch):

            kl_weight = tf.reduce_sum(tf.ones_like(x_train_batch))/self.num_training_points

            with tf.GradientTape(watch_accessed_variables=True) as tape:
                # Create the loss function we want to optimize.
                loss = self.variational_loss(observations=y_train_batch,observation_index_points=x_train_batch,kl_weight=kl_weight) 
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return loss

        pbar = tqdm(range(NUM_EPOCHS))
        loss_history = np.zeros((NUM_EPOCHS))


        for i in pbar:
            batch_count=0    
            epoch_loss = 0.0
            if i>=WARM_UP:
                self.len_variational_inducing_observations_scale.variables[0]._trainable=True
                self.amp_variational_inducing_observations_scale.variables[0]._trainable=True
            for batch in self.dataset:
                batch_loss = 0.0
                for s in range(self.num_sequential_samples):
                    loss = train_step(*batch)
                    batch_loss += loss.numpy()
                batch_loss/=self.num_sequential_samples
                epoch_loss+=batch_loss
                batch_count+=1
                pbar.set_description("Loss %f" % (epoch_loss/batch_count))
            loss_history[i] = epoch_loss/batch_count

        #vgp.len_variational_inducing_observations_scale.variables[0]._trainable=True
        #vgp.amp_variational_inducing_observations_scale.variables[0]._trainable=True
        return loss_history



    def variational_loss(self,observations,observation_index_points,kl_weight=1.0):
        
        kl_penalty = self.surrogate_posterior_kl_divergence_prior()
        recon = self.surrogate_posterior_expected_log_likelihood(observations,observation_index_points)
        return (-recon + kl_weight*kl_penalty)

    
    def surrogate_posterior_kl_divergence_prior(self):
        return kullback_leibler.kl_divergence(self.len_variational_inducing_observations_posterior,self.len_inducing_prior) + kullback_leibler.kl_divergence(self.amp_variational_inducing_observations_posterior,self.amp_inducing_prior)

    
    def surrogate_posterior_expected_log_likelihood(self,observations,observation_index_points):

        amp_vals = self.get_amp_samples(observation_index_points,S=self.num_parallel_samples)   
        len_vals = self.get_len_samples(observation_index_points,S=self.num_parallel_samples)   
        K = self.non_stat_matern12(observation_index_points, len_vals, amp_vals) # BxNxN
        K = K + (tf.eye(tf.shape(K)[-1], dtype=tf.float64) * tf.nn.softplus(self.vgp_observation_noise_variance))

        logpdf = tf.reduce_sum(tf.reduce_mean(tfd.MultivariateNormalTriL(scale_tril = tf.linalg.cholesky(K)).log_prob((observations[...,0])),axis=0))

        return logpdf

    def get_amp_samples(self,observation_index_points,S=1, full_cov=True):
        mean, var = self.get_amp_cond(observation_index_points)
        return (tf.math.softplus(self.mean_amp + self.sample_conditional(mean, var, S)))# changed to + instead of *
    
    def get_len_samples(self,observation_index_points,S=1, full_cov=True):
        mean, var = self.get_len_cond(observation_index_points)
        return (tf.math.softplus(self.mean_len + self.sample_conditional(mean, var, S))) # changed to + instead of *

    #p(sigma|sigma_z)
    def get_amp_cond(self, observation_index_points, full_cov=True):

        Xnew = observation_index_points

        Z = self.amp_inducing_index_points 

        kernel = self.kernel_amp
        f = self.amp_variational_inducing_observations_loc
        q_sqrt = self.amp_variational_inducing_observations_scale

        
        M = tf.shape(f)[0]
        Kmm = kernel.matrix(Z,Z)
        Kmm += self.jitter * tf.eye(M, dtype=Kmm.dtype)
        Kmn = kernel.matrix(Z, Xnew)
        Knn = kernel.matrix(Xnew,Xnew)
        mean,var = self.conditional(Kmn,Kmm,Knn,f,q_sqrt,full_cov=full_cov)

        return mean, var
    
    #p(l|l_z)
    def get_len_cond(self, observation_index_points, full_cov=True):
        

        Xnew = observation_index_points
        

        Z = self.len_inducing_index_points 
       
    
        kernel = self.kernel_len
        f = self.len_variational_inducing_observations_loc
        q_sqrt = self.len_variational_inducing_observations_scale

        M = tf.shape(f)[0]
        Kmm = kernel.matrix(Z,Z)
        Kmm += self.jitter * tf.eye(M, dtype=Kmm.dtype)
        Kmn = kernel.matrix(Z, Xnew)
        Knn = kernel.matrix(Xnew,Xnew)
        mean,var = self.conditional(Kmn,Kmm,Knn,f,q_sqrt,full_cov=full_cov)

        return mean, var

    def sample_conditional(self, mean, var, S=1):
        # mean BxNx1
        # var BxNxN
        # returns SxBxNx1
        B = tf.shape(mean)[0]
        N = tf.shape(mean)[1]
        z = tf.random.normal((S,B,N,1),dtype=tf.float64)
        
        I = self.jitter * tf.eye(N, dtype=tf.float64) #NN
        chol = tf.linalg.cholesky(var + I)  # BNN
        samples = mean + tf.matmul(chol, z)#[:, :, :, 0]  # BSN1

        return samples

    def conditional(self, Kmn, Kmm, Knn, f, q_sqrt, full_cov=True):

        f = tf.expand_dims(f,-1)
        q_sqrt= tf.expand_dims(q_sqrt,0)

        if not full_cov:
            Knn = tf.linalg.diag_part(Knn)
        Lm = tf.linalg.cholesky(Kmm)

        N = tf.shape(Kmn)[-1]
        M = tf.shape(f)[0]

        # Compute the projection matrix A
        Lm = tf.broadcast_to(Lm, tf.shape(Lm))
        A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [..., M, N]

        # compute the covariance due to the conditioning
        if full_cov:
            fvar = Knn - tf.linalg.matmul(A, A, transpose_a=True)  # [..., N, N]
        else:
            fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # [..., N]

        # construct the conditional mean
        f_shape = [M, 1]
        f = tf.broadcast_to(f, f_shape)  # [..., M, R]
        fmean = tf.linalg.matmul(A, f, transpose_a=True)  # [..., N, R]

        L = tf.linalg.band_part(q_sqrt, -1, 0)  

        LTA = tf.linalg.matmul(L, A, transpose_a=True)  # [R, M, N]

        if full_cov:
            fvar = fvar + tf.linalg.matmul(LTA, LTA, transpose_a=True)  # [R, N, N]
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), -2)  # [R, N]
            fvar = tf.linalg.adjoint(fvar)  # [N, R]

        return fmean, fvar
    
        
    def non_stat_matern12(self, X, lengthscales, stddev):
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


    def non_stat_vel(self,T,lengthscales, stddev):
        
        """Non-stationary integrated Matern12 kernel"""

        sigma_ = 0.5*(stddev[...,:-1,0,None] + stddev[...,1:,0,None])
        len_ = 0.5*(lengthscales[...,:-1,0,None] + lengthscales[...,1:,0,None])

        Ls = tf.square(len_)

        L = tf.math.sqrt(0.5*(Ls + tf.linalg.matrix_transpose(Ls)))

        prefactL = tf.math.sqrt(tf.matmul(len_, len_, transpose_b=True))
        prefactV = tf.matmul(sigma_, sigma_,transpose_b=True)

        zeta = tf.math.multiply(prefactV,tf.math.divide(prefactL,L))
    

        tpq1 = tf.math.exp(tf.math.divide(-tf.math.abs(tf.linalg.matrix_transpose(T[:-1]) - T[1:]),L))
        tp1q1 = tf.math.exp(tf.math.divide(-tf.math.abs(tf.linalg.matrix_transpose(T[1:]) - T[1:]),L))
        tpq = tf.math.exp(tf.math.divide(-tf.math.abs(tf.linalg.matrix_transpose(T[:-1]) - T[:-1]),L))
        tp1q = tf.math.exp(tf.math.divide(-tf.math.abs(tf.linalg.matrix_transpose(T[1:]) - T[:-1]),L))


        Epq_grid = tpq1-tp1q1-tpq+tp1q
        Epq_grid = (L**2)*Epq_grid
                
        Epq_grid = tf.linalg.set_diag(Epq_grid,(tf.linalg.diag_part(Epq_grid)) + 2.0*tf.squeeze(len_)[:]*(tf.squeeze(T[1:])-tf.squeeze(T[:-1])))
        Epq_grid = zeta*Epq_grid
        
        
        K = tf.math.cumsum(tf.math.cumsum(Epq_grid,axis=-2,exclusive=False),axis=-1,exclusive=False)
        
        return K
    
