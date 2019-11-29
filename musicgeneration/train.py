from musicgeneration.preprocess import Preprocess
from musicgeneration.neuralnet import NeuralNet


class Train(object):
    def __init__(self):
        pass

    
    def train_model(self):
        '''train the model for music generation'''
        
        #create network input & network output pair
        preprocess = Preprocess()
        notes = preprocess.generateNotes()
        n_vocab = len(set(notes))
        network_input, network_output = preprocess.prepareSequence(notes, n_vocab)

        #define neural network
        neural_net = NeuralNet()
        model = neural_net.create_network(network_input, n_vocab)
        print('Model Summary::')
        print(model.summary())


        #train neural network
        neural_net.train(model, network_input, network_output)
        






if __name__ == "__main__":
    train = Train()
    train.train_model()


