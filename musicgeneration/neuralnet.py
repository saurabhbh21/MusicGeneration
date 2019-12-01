import numpy as np
import tensorflow as tf

from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.python.keras.callbacks import ModelCheckpoint


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class NeuralNetwork(object):
    def __init__(self):
        pass

    def generator(self, n_vocab, random_noise_size, sequence_length):
        model = Sequential()
        model.add(LSTM(
            1,
            input_shape=(sequence_length, random_noise_size),
            return_sequences=True
        ))

        # model.add(Dropout(0.3))
        # model.add(LSTM(100, return_sequences=True))
        # model.add(Dropout(0.3))
        # model.add(LSTM(100, return_sequences=True))
        # model.add(Dense(256))
        # model.add(Dropout(0.3))
        # model.add(Dense(n_vocab))
        #model.add(Activation('softmax'))

        return model
        

    def discriminator(self, network_input):
        """ create the discriminator neural network """
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        
        return model



    @staticmethod
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)


    @staticmethod
    def optimizers():
        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        return generator_optimizer, discriminator_optimizer



    
    
