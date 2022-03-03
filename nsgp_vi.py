"""
Copyright 2022 Colin Torney

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

from tensorflow_probability.python.distributions import kullback_leibler

from utils.gradient_accumulator import GradientAccumulator

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

dtype = np.float64
NUM_LATENT = 2

class nsgpVI(tf.Module):
                                        
    def __init__(self,kernel_len,kernel_amp,n_inducing_points,inducing_index_points,dataset,num_training_points, init_observation_noise_variance=1e-2,num_sequential_samples=10,num_parallel_samples=10,kernel_len_priors=None,kernel_amp_priors=None,obs_noise_prior=None,jitter=1e-6):
               
        self.jitter=jitter
        
        self.mean_len = tf.Variable([0.0], dtype=tf.float64, name='len_mean')
        self.mean_amp = tf.Variable([0.0], dtype=tf.float64, name='var_mean')
        
        self.inducing_index_points = tf.Variable(inducing_index_points,dtype=dtype,name='ind_points',trainable=1) #z's for lower level functions

        self.kernel_len = kernel_len
        self.kernel_amp = kernel_amp
        
        #parameters for variational distribution for len,phi(l_z) and var,phi(sigma_z)
        self.q_mu = tf.Variable(np.zeros((NUM_LATENT*n_inducing_points),dtype=dtype),name='ind_loc_post')
        self.len_scale = tfp.util.TransformedVariable([np.eye(n_inducing_points, dtype=dtype)],tfp.bijectors.FillScaleTriL(diag_shift=np.float64(1e-05)),dtype=tf.float64, name='len_scale_post', trainable=1)
        self.amp_scale = tfp.util.TransformedVariable([np.eye(n_inducing_points, dtype=dtype)],tfp.bijectors.FillScaleTriL(diag_shift=np.float64(1e-05)),dtype=tf.float64, name='amp_scale_post', trainable=1)
        self.cc_scale = tf.Variable([np.zeros((n_inducing_points),dtype=dtype)],name='cc_scale',trainable=1)

        len_op = tf.linalg.LinearOperatorLowerTriangular(self.len_scale)
        amp_op = tf.linalg.LinearOperatorLowerTriangular(self.amp_scale)
        cc_op = tf.linalg.LinearOperatorDiag(self.cc_scale)
        self.q_sqrt = tf.linalg.LinearOperatorBlockLowerTriangular([[len_op],[cc_op,amp_op]])    
        
        #approximation to the posterior: phi(l_z)
        self.variational_inducing_observations_posterior = tfd.MultivariateNormalLinearOperator(
                                                                      loc=self.q_mu,
                                                                      scale=self.q_sqrt) 

        #p(l_z)
        self.inducing_prior = tfd.MultivariateNormalDiag(loc=tf.zeros((NUM_LATENT*n_inducing_points),dtype=tf.float64),name='ind_prior')
        self.M = n_inducing_points
        
        
        self.kernel_len_priors = kernel_len_priors
        self.kernel_amp_priors = kernel_amp_priors
        self.obs_noise_prior = obs_noise_prior
        
        
        
        self.vgp_observation_noise_variance = tf.Variable(np.log(np.exp(init_observation_noise_variance)-1),dtype=dtype,name='nv', trainable=1)

        self.num_sequential_samples=num_sequential_samples
        self.num_parallel_samples=num_parallel_samples
        
        self.dataset = dataset
        self.num_training_points=num_training_points
        

    def optimize(self, BATCH_SIZE, SEG_LENGTH, NUM_EPOCHS=100):


        strategy = tf.distribute.MirroredStrategy()
        dist_dataset = strategy.experimental_distribute_dataset(self.dataset)

        initial_learning_rate = 1e-1
        steps_per_epoch = self.num_training_points//(BATCH_SIZE*SEG_LENGTH)
        learning_rate = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,decay_steps=steps_per_epoch,decay_rate=0.99,staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) #,beta_2=0.99, amsgrad=False)
        accumulator = GradientAccumulator()

        def train_step(inputs):
            x_train_batch, y_train_batch = inputs
            kl_weight = tf.reduce_sum(tf.ones_like(x_train_batch))/self.num_training_points

            with tf.GradientTape(watch_accessed_variables=True) as tape:
                loss = self.variational_loss(observations=y_train_batch,observation_index_points=x_train_batch,kl_weight=kl_weight) 
            grads = tape.gradient(loss, self.trainable_variables)
            return loss, grads

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses, per_replica_grads = strategy.run(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_grads, axis=None)

        pbar = tqdm(range(NUM_EPOCHS))
        loss_history = np.zeros((NUM_EPOCHS))

        for i in pbar:
            batch_count=0    
            epoch_loss = 0.0
            for batch in dist_dataset:
                batch_loss = 0.0
                for s in range(self.num_sequential_samples):
                    loss, grads = distributed_train_step(batch)
                    # accumulate the loss and gradient
                    accumulator(grads)
                    batch_loss += loss.numpy()
               
                grads = accumulator.gradients
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                batch_loss/=self.num_sequential_samples
                accumulator.reset()
                    
                epoch_loss+=batch_loss
                batch_count+=1
                pbar.set_description("Loss %f, klen_l %f, kamp_l %f " % (epoch_loss/batch_count, self.kernel_len.length_scale.numpy(), self.kernel_amp.length_scale.numpy()))
            loss_history[i] = epoch_loss/batch_count
            #print(epoch_loss)

        return loss_history


    def variational_loss(self,observations,observation_index_points,kl_weight=1.0):
        
        kl_penalty = self.surrogate_posterior_kl_divergence_prior()
        recon = self.surrogate_posterior_expected_log_likelihood(observations,observation_index_points)
        return (-recon + kl_weight*kl_penalty)

    
    def surrogate_posterior_kl_divergence_prior(self):
        return kullback_leibler.kl_divergence(self.variational_inducing_observations_posterior,self.inducing_prior) 

    
    def surrogate_posterior_expected_log_likelihood(self,observations,observation_index_points):

        len_vals, amp_vals = self.get_samples(observation_index_points,S=self.num_parallel_samples)   
        K = self.non_stat_matern12(observation_index_points, len_vals, amp_vals) # BxNxN
        K = K + (tf.eye(tf.shape(K)[-1], dtype=tf.float64) * tf.nn.softplus(self.vgp_observation_noise_variance))

        logpdf = tf.reduce_sum(tf.reduce_mean(tfd.MultivariateNormalTriL(scale_tril = tf.linalg.cholesky(K)).log_prob((observations[...,0])),axis=0))

        return logpdf

    
    def get_samples(self,predictor_values,S=1):
        mean, var = self.get_conditional(predictor_values)
        samples = self.sample_conditional(mean, var, S)
    
        len_samples,amp_samples = tf.split(samples,NUM_LATENT,axis=2)
        
        return tf.math.softplus(self.mean_len + len_samples), tf.math.softplus(self.mean_amp + amp_samples)
    
    def get_conditional(self, X):
        
        Z = self.inducing_index_points 
        M = self.M

        Lm_len = tf.linalg.LinearOperatorFullMatrix(self.kernel_len.matrix(Z,Z) + self.jitter * tf.eye(M, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True).cholesky()
        Lm_amp = tf.linalg.LinearOperatorFullMatrix(self.kernel_amp.matrix(Z,Z) + self.jitter * tf.eye(M, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True).cholesky()

        Kmn_len = tf.linalg.LinearOperatorFullMatrix(self.kernel_len.matrix(Z, X),is_positive_definite=True,is_self_adjoint=True)
        Kmn_amp = tf.linalg.LinearOperatorFullMatrix(self.kernel_amp.matrix(Z, X),is_positive_definite=True,is_self_adjoint=True)

        Lm_len_inv_Kmn = Lm_len.solve(Kmn_len)
        Lm_amp_inv_Kmn = Lm_amp.solve(Kmn_amp)
        Lm_inv_Kmn = tf.linalg.LinearOperatorBlockDiag([Lm_len_inv_Kmn,Lm_amp_inv_Kmn])

        mean_f = tf.expand_dims(Lm_inv_Kmn.matvec(self.q_mu, adjoint=True),-1)

        Lm_inv_Kmn_q = Lm_inv_Kmn.matmul(self.q_sqrt, adjoint=True)
        Lm_inv_Kmn_q2 = Lm_inv_Kmn_q.matmul(Lm_inv_Kmn_q,adjoint_arg=True)

        Knn_len = tf.linalg.LinearOperatorFullMatrix(self.kernel_len.matrix(X, X),is_positive_definite=True,is_self_adjoint=True)
        Knn_amp = tf.linalg.LinearOperatorFullMatrix(self.kernel_amp.matrix(X, X),is_positive_definite=True,is_self_adjoint=True)

        Knn = tf.linalg.LinearOperatorBlockDiag([Knn_len,Knn_amp])

        Lm_len_inv_Kmn2 = Lm_len_inv_Kmn.matmul(Lm_len_inv_Kmn,adjoint=True)
        Lm_amp_inv_Kmn2 = Lm_amp_inv_Kmn.matmul(Lm_amp_inv_Kmn,adjoint=True)
        Lm_inv_Kmn2 = tf.linalg.LinearOperatorBlockDiag([Lm_len_inv_Kmn2,Lm_amp_inv_Kmn2])

        covar_f = Lm_inv_Kmn_q2.to_dense() + Knn.to_dense() - Lm_inv_Kmn2.to_dense()

        return mean_f, covar_f

    def get_marginal(self, X):

        tf.debugging.assert_rank(X,3,message="get_marginal expects a batch of locations. Add first dimension of size 1 if processing a single batch" )

        mean_f, covar_f = self.get_conditional(X)

        covar_f = tf.linalg.diag_part(covar_f)
        mean_list = tf.split(mean_f,NUM_LATENT,axis=1)
        var_list = tf.split(covar_f,NUM_LATENT,axis=1)

        return mean_list, var_list


       

    def sample_conditional(self, mean, var, S=1):
        # mean BxNx1
        # var BxNxN
        # returns SxBxNx1
        B = tf.shape(mean)[0]
        N = tf.shape(mean)[1]
        z = tf.random.normal((S,B,N,1),dtype=tf.float64)
        
        I = self.jitter**1 * tf.eye(N, dtype=tf.float64) #NN
        chol = tf.linalg.cholesky(var + I)  # BNN

        samples = tf.expand_dims(mean,0) + tf.matmul(chol, z)#[:, :, :, 0]  # BSN1
        return samples

    
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

