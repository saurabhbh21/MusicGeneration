from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.python.keras.callbacks import ModelCheckpoint


class NeuralNet(object):
    def __init__(self):
        pass

    def create_network(self, network_input, n_vocab):
        """ create the structure of the neural network """
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
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        
        return model
    
    def train(self, model, network_input, network_output):
        """ train the neural network """
        filepath = "model/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]
        model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)
