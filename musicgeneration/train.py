import os
import time
import numpy as np
import tensorflow as tf
from musicgeneration.preprocess import Preprocess
from musicgeneration.neuralnet import NeuralNetwork

BATCH_SIZE = 1
EPOCHS = 10
NOISE_DIM = 1
SEQUENCE_LENGTH = 100


class Train(object):
    def __init__(self):
        pass 

    def generate_noise(self, batch_size, random_noise_size, sequence_length, n_vocab):
        random_sequence = [ np.random.randint(0, n_vocab-1) for index in range(sequence_length)]
        return np.reshape(random_sequence, (batch_size, sequence_length, random_noise_size))

        

    def checkpoints(self):
        checkpoint_dir = './model/checkpoint'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)

        return checkpoint, checkpoint_prefix



    def train_step(self, normalized_music_seq, n_vocab):

        noise = self.generate_noise(BATCH_SIZE, NOISE_DIM, SEQUENCE_LENGTH, n_vocab)
        normalised_noise = noise / float(n_vocab)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generator_sequence = self.generator(normalised_noise, training=True)
            
            # output_index = np.argmax(generator_prediction)
            # normalised_output = output_index / float(n_vocab)
            # generator_sequence = normalised_noise.flatten()
            # generator_sequence = np.append(generator_sequence, normalised_output)
            # generator_sequence = np.delete(generator_sequence, 0)
            # generator_sequence = np.reshape(normalised_noise, (BATCH_SIZE, SEQUENCE_LENGTH, NOISE_DIM))

            
            real_output = self.discriminator(normalized_music_seq, training=True)
            fake_output = self.discriminator(generator_sequence, training=True)


            gen_loss = NeuralNetwork.generator_loss(fake_output)
            disc_loss = NeuralNetwork.discriminator_loss(real_output, fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        print('GEN=', gradients_of_generator)
        print('DISC=', gradients_of_discriminator)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))



    def train_model(self, epochs):
        '''train the model for music generation'''
        
        #preprocess dataset
        preprocess = Preprocess()
        notes = preprocess.generateNotes()
        pitchnames = sorted(set(item for item in notes))
        n_vocab = len(set(notes))
        network_input, normalized_input = preprocess.prepareSequence(notes, n_vocab)


        #Initialise neural network (GAN) for sequence generation
        neural_network = NeuralNetwork()        
        self.discriminator = neural_network.discriminator(normalized_input)
        self.generator = neural_network.generator(n_vocab, NOISE_DIM, SEQUENCE_LENGTH)
        self.generator_optimizer, self.discriminator_optimizer = neural_network.optimizers()


        #train the neural network
        for epoch in range(epochs):
            start = time.time()

            batch_index = 1
            for music_seq in normalized_input:
                print('Batch index=', batch_index)
                batch_index += 1
                self.train_step(np.reshape(music_seq, (BATCH_SIZE, music_seq.shape[0], music_seq.shape[1])), n_vocab)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            #save model every 2 epoch
            checkpoint, checkpoint_prefix = self.checkpoints()
            checkpoint.save(file_prefix=checkpoint_prefix)


            




if __name__ == "__main__":
    train = Train()
    train.train_model(EPOCHS)


