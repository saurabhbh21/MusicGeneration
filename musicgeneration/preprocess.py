import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord

import tensorflow as tf


class Preprocess(object):
    def __init__(self):
        pass

    def generateNotes(self, file_path=None):
        notes = list()

        for file in  glob.glob("./transposed_music/transposed_music/untransposed_songs/*.mid"):
            midi = converter.parse(file)
            notes_to_parse = None

            try: # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse() 
            
            except: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                
                elif isinstance(element, chord.Chord):
                    notes.append( '.'.join(str(n) for n in element.normalOrder) )
            
            #store all the notes as a pickle object in disk
            with open('data/notes', 'wb') as filepath:
                pickle.dump(notes, filepath)

        return notes 

    
    def prepareSequence(self, notes, n_vocab):
        '''Prepare sequence data which serves as neural network input'''

        sequence_length=100
       
        #get all pitchnames
        pitch_names = sorted(set(item for item in notes))
       
        #dictionary to map pitches to number
        note_to_int =  dict( (note, number) for number, note in enumerate(pitch_names))
        
        network_input = list()
        network_output = list()

        #create input and output sequence
        for i in range(0, len(notes)-sequence_length, 1):
            seq_input = notes[i:i+sequence_length]
            seq_output = notes[i+sequence_length]

            network_input.append([note_to_int[char] for char in seq_input])
            network_output.append(note_to_int[seq_output])

        num_patterns = len(network_input)

        #reshape network input and output data for sequential-NN (viz LSTM/GRU)
        network_input = np.reshape(network_input, (num_patterns, sequence_length, 1))
        #normalize input
        normalized_input = network_input/float(n_vocab)
        
        #one-hot encode output vector
        network_output = tf.keras.utils.to_categorical(network_output)

        return (network_input, normalized_input)



if __name__ == "__main__":
    preprocess = Preprocess()
    result = preprocess.vectorizeNoteSequence()
    
    # print('Network Input:')
    # print(result[0])

    # print('Network Output::')
    # print(result[1])
        


            



