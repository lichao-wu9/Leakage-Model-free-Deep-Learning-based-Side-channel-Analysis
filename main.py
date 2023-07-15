import sys
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *

import numpy as np
from sklearn.preprocessing import StandardScaler

import SCA_dataset as datasets
import Attack

def byte_to_bit_array(byte_val):
    return np.array([int(b) for b in format(byte_val, '08b')])

def convert_2D_decimal_array_to_bits(arr):

    def decimal_to_binary(number, bits):
        return format(number, f'0{bits}b')

    arr = np.array(arr)
    max_values = np.max(arr, axis=0)
    result = []

    for row in arr:
        binary_row = []
        for idx, val in enumerate(row):
            if max_values[idx] < 16:
                binary_val = decimal_to_binary(val, 4)
            elif max_values[idx] < 256:
                binary_val = decimal_to_binary(val, 8)
            else:
                raise ValueError(f"Column {idx} has a value greater than or equal to 256 which is not supported.")
            binary_row.extend([int(bit) for bit in binary_val])
        result.append(binary_row)
    
    return np.array(result)

def creat_multi_binary_model(input_length, bin_num, desync_level=5):
    # Define the shared 1D CNN base model
    inputs = Input(shape=(input_length, 1))
    x = RandomTranslation(width_factor=desync_level/input_length, height_factor=0, fill_mode='wrap')(inputs)

    x = Conv1D(kernel_size=40, strides=20, filters=4, activation="selu", padding="same")(x)
    x = AveragePooling1D(pool_size=2, strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv1D(kernel_size=8, strides=4, filters=32, activation="selu", padding="same")(x)
    x = AveragePooling1D(pool_size=2, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)

    # Define separate MLPs for each task
    def create_mlp(input_layer):
        mlp = Dense(32, activation="selu", kernel_initializer="random_uniform")(input_layer)
        mlp = Dense(32, activation="selu", kernel_initializer="random_uniform")(mlp)
        return mlp

    # Define separate output layers for each task
    bit_outputs = [Dense(1, activation='sigmoid', name=f'bit{bin_num-1-i}')(create_mlp(x)) for i in range(bin_num)]

    # Combine the shared base model and output layers into a single Model
    model = Model(inputs=inputs, outputs=bit_outputs)

    # Compile the model with separate losses for each task
    model.compile(optimizer=Adam(), loss=['binary_crossentropy'] * bin_num, metrics=['accuracy'])
    model.summary()
    return model

def get_multi_binary_model_label(data_bin):
    bin_num = data_bin.shape[1]
    return {f"bit{bin_num-i-1}": data_bin[:, i] for i in range(bin_num)}

if __name__ == "__main__":
    data_root = ''
    model_root = ''
    result_root = ''

    datasetss = sys.argv[1].split(',') #ASCAD
    num_bytes = int(sys.argv[2]) #16
    aug_level = int(sys.argv[3]) #5
    epochs = int(sys.argv[4]) #200
    batch_size = int(sys.argv[5]) #512
    train_model = bool(int(sys.argv[6])) #1
    index = sys.argv[7] #1
    
    for dataset in datasetss:
        num_of_attack_traces = 5000
        if dataset == 'ASCAD':
            all_key = [77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105]
            all_key = all_key[2:]
            (X_profiling, X_attack), (Y_profiling,  Y_attack), (plt_profiling,  plt_attack) = datasets.load_ascad(data_root+dataset+'/', num_bytes)
        elif dataset == 'ASCAD_rand':
            all_key = [0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255]
            all_key = all_key[2:]
            (X_profiling, X_attack), (Y_profiling,  Y_attack), (plt_profiling,  plt_attack) = datasets.load_ascad_rand(data_root+dataset+'/', num_bytes)
        elif dataset == 'CHES_CTF':
            all_key = [23, 92, 242, 153, 122, 133, 131, 65, 60, 119, 223, 172, 126, 108, 89, 216]
            (X_profiling, X_attack), (Y_profiling,  Y_attack), (plt_profiling,  plt_attack) = datasets.load_chesctf(data_root+dataset+'/', num_bytes)
        elif dataset == 'Eshard':
            all_key = [7, 123, 218, 15, 171, 30, 85, 1, 234, 208, 21, 10, 177, 224, 32, 254]
            (X_profiling, X_attack), (Y_profiling,  Y_attack), (plt_profiling,  plt_attack) = datasets.load_eshard(data_root+dataset+'/', num_bytes)
        
        Y_profiling_bin = convert_2D_decimal_array_to_bits(Y_profiling)
        Y_attack_bin = convert_2D_decimal_array_to_bits(Y_attack)

        bin_num = Y_profiling_bin.shape[1]
        # Normalize the data
        scaler = StandardScaler()
        X_profiling = scaler.fit_transform(X_profiling)
        X_attack = scaler.transform(X_attack) 
        
        # Test info
        test_info = '{}_{}_epoch{}_bs{}_numBytes{}_{}'.format(dataset, aug_level, epochs, batch_size, num_bytes, index)
        print('====={}====='.format(test_info))
        
        # if model is CNN, we have to make the data dim equals to 3 
        X_profiling = np.expand_dims(X_profiling, axis=-1)
        X_attack = np.expand_dims(X_attack, axis=-1)
        
        print("Create model")
        model = creat_multi_binary_model(X_attack.shape[1], bin_num, desync_level=aug_level)

        # Train the model
        if train_model: 
            print('Train model...')
            history = model.fit(X_profiling, 
                                get_multi_binary_model_label(Y_profiling_bin),
                                validation_split=0.15,
                                epochs=epochs, 
                                batch_size=batch_size, 
                                verbose=2)
        

            model.save(model_root+"Model_{}.h5".format(test_info))

        else:
            print('Load model...')
            model = load_model(model_root+"Model_{}.h5".format(test_info))

        model_test = creat_multi_binary_model(X_attack.shape[1], bin_num, desync_level=0)
        model_test.set_weights(model.get_weights()) 

        predictions = np.array(model_test.predict(X_attack))

        # Create an array of all possible bytes and their bit representations
        all_bytes = np.arange(256)
        all_byte_bits = np.array([byte_to_bit_array(byte) for byte in all_bytes])

        # Calculate probabilities for each byte using broadcasting
        save_container = np.zeros((16, num_of_attack_traces))
        for idx, target_byte in enumerate(range(Y_profiling.shape[1])):
            print("===============")
            print("Attack key byte {}".format(target_byte))
            byte_prob = np.ones((X_attack.shape[0], 256))
            for i in range(8):
                byte_prob *= all_byte_bits[:, i] * predictions[target_byte*8+i] + (1 - all_byte_bits[:, i]) * (1 - predictions[target_byte*8+i])

            avg_rank, all_rank = Attack.perform_attacks(num_of_attack_traces, byte_prob, all_key[target_byte], plt_attack, target_byte, log=True, dataset=dataset, nb_attacks=5)
            print('GE: ', avg_rank[-1])
            print('GE smaller than 1: ', np.argmax(avg_rank < 1))
            print('GE smaller than 5: ', np.argmax(avg_rank < 5))
            save_container[idx] = avg_rank

        np.save(result_root+f'GE_{test_info}.npy', save_container)

        K.clear_session()

