#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import tensorflow.keras as K
from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston


##=========== Inner Class for the MDN tensorflow with one output at a time
class MDN(tf.keras.Model):
    def __init__(self, neurons=100, components = 3):
        super(MDN, self).__init__(name="MDN")
        self.neurons = neurons
        self.components = components ## This is the number of mixture model you would like to consider
        
        self.h1 = Dense(neurons, activation="relu", name="h1")
        self.h2 = Dense(neurons, activation="relu", name="h2")
        
        self.alphas = Dense(components, activation="softmax", name="alphas")
        self.mus = Dense(components, name="mus")
        self.sigmas = Dense(components, activation="nnelu", name="sigmas")
        self.pvec = Concatenate(name="pvec")
        
    def call(self, inputs):
        x = self.h1(inputs)
        x = self.h2(x)
        
        alpha_v = self.alphas(x)
        mu_v = self.mus(x)
        sigma_v = self.sigmas(x)
        
        return self.pvec([alpha_v, mu_v, sigma_v])

class MDN_prediction():
    def __init__(self, **kwargs):
        self.no_parameters = kwargs['no_parameters']
        self.components = kwargs['components']
        self.neurons = kwargs['neurons']
        tf.keras.utils.get_custom_objects().update({'nnelu': Activation(self.nnelu)})
        self.mdn = MDN(neurons=self.neurons, components=self.components)
        self.compile_MDN()


    def compile_MDN(self):
        
        opt = tf.optimizers.Adam(1e-3)

        # self.mdn = self.MDN(neurons=self.neurons, components=self.components)
        self.mdn.compile(loss=self.gnll_loss, optimizer=opt)


    def fit(self, **kwargs):
        x_train = kwargs['x_train']
        y_train = kwargs['y_train']
        x_test = kwargs['x_test']
        y_test = kwargs['y_test']
        epochs = kwargs['epochs']
        batch_size = kwargs['batch_size']
        tensorboard = TensorBoard(histogram_freq=0, write_graph=True, write_images=False)
        mon = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        self.mdn.fit(x=x_train, y=y_train,epochs=epochs, validation_data=(x_test, y_test), 
                     callbacks=[mon, tensorboard], batch_size=batch_size, verbose=0)

    def predict_MDN(self, **kwargs):
        x_test = kwargs['x_test']
        pred_components = self.mdn.predict(x_test)
        y_pred_mean, y_pred_std = self.prediction_calc(pred_components)
        return y_pred_mean, y_pred_std
    


    ##============= Utils Methods =============
    def nnelu(self, input):
        '''
        Computes the Non-Negative Exponential Linear Unit
        '''
        return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))

    def slice_parameter_vectors(self, parameter_vector):
        '''
        Returns an unpacked list of paramter vectors.
        '''
        return [parameter_vector[:,i*self.components:(i+1)*self.components] for i in range(self.no_parameters)]

    def gnll_loss(self, y , parameter_vector):
        '''
        Computes the mean negative log-likelihood loss of y given the mixture parameters.
        '''
        alpha, mu, sigma = self.slice_parameter_vectors(parameter_vector) # Unpack parameter vectors
        
        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alpha),
            components_distribution=tfd.Normal(
                loc=mu,       
                scale=sigma))
        
        log_likelihood = gm.log_prob(tf.transpose(y)) # Evaluate log-probability of y
        
        return -tf.reduce_mean(log_likelihood, axis=-1)


    def prediction_calc(self, pred_components):
        '''
        : pi = this is an array N by n_components
        : mu = the mean array that is N by n_components
        : std= the std array that is N by components

        : result = this should be also N by number of outputs
        '''
        alpha_pred, mu_pred, sigma_pred = self.slice_parameter_vectors(pred_components)
        y_pred_mean = np.multiply(alpha_pred, mu_pred).sum(axis=-1)
        y_pred_std  = np.multiply(alpha_pred, ( sigma_pred + (mu_pred - np.dot(y_pred_mean.reshape(-1,1), np.ones((1,self.components))))**2 ) ).sum(axis=-1)
        return y_pred_mean, y_pred_std


    def MAPE_calc(self, percentage_error, error, threshold):
        MAPE = 0
        c = 0
        for i in range(len(error)):
            if error[i] < threshold[i]:
                MAPE += percentage_error[i]
                c    += 1
        if c ==0: 
            MAPE = -1
            return MAPE
        else:
            return MAPE/c


#%%
if __name__ == "__main__":
    samples = int(1e3)
    x_data = np.random.sample((samples,1)).astype(np.float32)
    y_data = 5*x_data[:,0] +np.multiply((x_data[:,0])**2, np.random.standard_normal(samples))
    plt.scatter(x_data[:,0], y_data)
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=42)

    s = np.linspace(0,1,500)
    # for i in range(2):
    #     s[:,i] = np.linspace(0.,1.,int(1e3))

    sample_MDN = MDN_prediction(no_parameters = 3, components = 5, neurons= 256)
    sample_MDN.fit(x_train = x_train,
                        y_train = y_train,
                        x_test = x_test,
                        y_test = y_test,
                        epochs = 200,
                        batch_size = 128)

    y_pred_mean, y_pred_std = sample_MDN.predict_MDN(x_test = s)

    plt.scatter(x_data[:,0], y_data)
    plt.scatter(s,y_pred_mean)
    plt.scatter(s,y_pred_mean - y_pred_std)
    plt.scatter(s,y_pred_mean + y_pred_std)

    plt.show()
