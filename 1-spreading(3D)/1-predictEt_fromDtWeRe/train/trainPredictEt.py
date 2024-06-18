import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

def slicing(entrada, saida, tam_sequencia, pulo):
    x, y = [], []
        
    for i in range(len(entrada)):
        for j in range(0, len(entrada[0]) - tam_sequencia + 1, pulo):
            _x = entrada[i, j:(j + tam_sequencia), :]
            _y = saida[i, j:(j + tam_sequencia), :]
            x.append(_x)
            y.append(_y)

    return [np.array(x), np.array(y)]

def main():
    #------------------------------------------------------------------------------------------------------#
    #----------------------------------- Open the file with the data --------------------------------------#
    #------------------------------------------------------------------------------------------------------#
    arq = np.load("Time_series_spreading.npz", allow_pickle=True)
    dados = arq["TS"]
    passos_total = 1000
    dt = 0.05
    tempos = np.linspace(0, 50, passos_total)

    
    #------------------------------------------------------------------------------------------------------#
    #------------------------ Separate data for training, validation, and testing -------------------------#
    #------------------------------------------------------------------------------------------------------#
    # Train
    shapes_train = np.where(np.isin(arq["shape"], [2, 3, 5, 7, 9, 15, 17]))[0]
    repetir = np.array([arq["Re"][shapes_train], arq["We"][shapes_train]]).T
    repetir = np.array([[[repetir[k][i] for i in range(len(repetir[0]))] for j in range(arq["TS"].shape[1])] for k in range(len(repetir))])
    entrada_train = np.concatenate((arq["TS"][shapes_train, :, 0].reshape(len(repetir), len(repetir[0]), 1), repetir), -1)
    entrada_train = np.c_[entrada_train, np.array([tempos for i in range(len(shapes_train))]).reshape(len(shapes_train), len(tempos), 1)]
    saida_train = arq["TS"][shapes_train, :, 1:]
    
    # Validation
    shapes_valid = np.where(np.isin(arq["shape"], [4, 20]))[0]
    repetir = np.array([arq["Re"][shapes_valid], arq["We"][shapes_valid]]).T
    repetir = np.array([[[repetir[k][i] for i in range(len(repetir[0]))] for j in range(arq["TS"].shape[1])] for k in range(len(repetir))])
    entrada_valid = np.concatenate((arq["TS"][shapes_valid, :, 0].reshape(len(repetir), len(repetir[0]), 1), repetir), -1)
    entrada_valid = np.c_[entrada_valid, np.array([tempos for i in range(len(shapes_valid))]).reshape(len(shapes_valid), len(tempos), 1)]
    saida_valid = arq["TS"][shapes_valid, :, 1:]
    
    # Test
    shapes_test = np.where(np.isin(arq["shape"], [6]))[0]
    repetir = np.array([arq["Re"][shapes_test], arq["We"][shapes_test]]).T
    repetir = np.array([[[repetir[k][i] for i in range(len(repetir[0]))] for j in range(arq["TS"].shape[1])] for k in range(len(repetir))])
    entrada_test = np.concatenate((arq["TS"][shapes_test, :, 0].reshape(len(repetir), len(repetir[0]), 1), repetir), -1)
    entrada_test = np.c_[entrada_test, np.array([tempos for i in range(len(shapes_test))]).reshape(len(shapes_test), len(tempos), 1)]
    saida_test = arq["TS"][shapes_test, :, 1:]
    

    #------------------------------------------------------------------------------------------------------#
    #----------------------------------------- Shuffle the data -------------------------------------------#
    #------------------------------------------------------------------------------------------------------#
    entrada_train, saida_train = shuffle(entrada_train, saida_train, random_state=42)
    entrada_train = np.array(entrada_train)
    saida_train = np.array(saida_train)

    entrada_valid, saida_valid = shuffle(entrada_valid, saida_valid, random_state=42)
    entrada_valid = np.array(entrada_valid)
    saida_valid = np.array(saida_valid)

    entrada_test, saida_test = shuffle(entrada_test, saida_test, random_state=42)
    entrada_test = np.array(entrada_test)
    saida_test = np.array(saida_test)


    #------------------------------------------------------------------------------------------------------#
    #-------------------------------------- Apply Z-standardization ---------------------------------------#
    #------------------------------------------------------------------------------------------------------#
    media_entrada, desvio_entrada = [], []
    media_saida, desvio_saida = [], []
    
    # Calculate the mean and standard deviation of each column to normalize the input data
    for i in range(len(entrada_train[0][0])):
        media_entrada.append(np.mean(np.concatenate((entrada_train[:, :, i], entrada_valid[:, :, i], entrada_test[:, :, i]))))
        desvio_entrada.append(np.std(np.concatenate((entrada_train[:, :, i], entrada_valid[:, :, i], entrada_test[:, :, i]))))
        entrada_train[:, :, i] = (entrada_train[:, :, i] - media_entrada[i]) / desvio_entrada[i]
        entrada_valid[:, :, i] = (entrada_valid[:, :, i] - media_entrada[i]) / desvio_entrada[i]
        entrada_test[:, :, i] = (entrada_test[:, :, i] - media_entrada[i]) / desvio_entrada[i]
    media_entrada = np.array(media_entrada)
    desvio_entrada = np.array(desvio_entrada)

    # Calculate the mean and standard deviation of each column to normalize the output data
    for i in range(len(saida_train[0][0])):
        media_saida.append(np.mean(np.concatenate((saida_train[:, :, i], saida_valid[:, :, i], saida_test[:, :, i]))))
        desvio_saida.append(np.std(np.concatenate((saida_train[:, :, i], saida_valid[:, :, i], saida_test[:, :, i]))))
        saida_train[:, :, i] = (saida_train[:, :, i] - media_saida[i]) / desvio_saida[i]
        saida_valid[:, :, i] = (saida_valid[:, :, i] - media_saida[i]) / desvio_saida[i]
        saida_test[:, :, i] = (saida_test[:, :, i] - media_saida[i]) / desvio_saida[i]
    media_saida = np.array(media_saida)
    desvio_saida = np.array(desvio_saida)


    #------------------------------------------------------------------------------------------------------#
    #------------------------------------------ Slice the data --------------------------------------------#
    #------------------------------------------------------------------------------------------------------#
    passos = 100
    pulos = 100

    entrada_train, saida_train = slicing(entrada_train, saida_train, passos, pulos)
    entrada_valid, saida_valid = slicing(entrada_valid, saida_valid, passos, pulos)
    entrada_test, saida_test = slicing(entrada_test, saida_test, passos, pulos)


    #------------------------------------------------------------------------------------------------------#
    #------------------ Save the test data and the parameters to undo the standarization ------------------#
    #------------------------------------------------------------------------------------------------------#    
    save_dict = {
        'entrada_train': entrada_train,
        'saida_train': saida_train,
        'entrada_test': entrada_test,
        'saida_test': saida_test,
        'entrada_valid': entrada_valid,
        'saida_valid': saida_valid,
        'media_entrada': media_entrada,
        'desvio_entrada': desvio_entrada,
        'media_saida': media_saida,
        'desvio_saida': desvio_saida
    }
    np.savez('dados_teste', **save_dict)


    #------------------------------------------------------------------------------------------------------#
    #----------------------------------------- LSTM architecture ------------------------------------------#
    #------------------------------------------------------------------------------------------------------#
    taxa_aprendizado = 0.0001 # Learning rate
    epocas = 5000             # Epochs
    batch = 8                 # Batch size

    tf.keras.utils.set_random_seed(42)

    model = Sequential()

    input_shape = (passos, 4)
    output_shape = (passos, 3)
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape, name="Oculta1"))
    model.add(LSTM(32, return_sequences=True, name="Oculta2"))
    model.add(LSTM(32, name="Oculta3"))
    model.add(Dense(output_shape[0]*output_shape[1], activation="linear", name="Output"))
    model.add(Reshape(output_shape))

    optimizer = Adam(lr = taxa_aprendizado)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])

    model.summary()

    mcp_save = ModelCheckpoint('predictEt.keras', save_best_only=True, monitor='val_loss', mode='min')
    erl_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    historico1 = model.fit(entrada_train, saida_train, validation_data = (entrada_valid, saida_valid), epochs=epocas,
                           batch_size=batch, callbacks=[mcp_save, erl_stop], verbose=1)


    #------------------------------------------------------------------------------------------------------#
    #---------------------------------------- Data about training -----------------------------------------#
    #------------------------------------------------------------------------------------------------------#
    # Training and validation accuracy
    plt.plot(historico1.history['mean_absolute_error'])
    plt.plot(historico1.history['val_mean_absolute_error'])
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Validação'], loc='upper left')
    plt.savefig("predictEt_acuracia.png")
    plt.clf()

    # Training and validation loss
    plt.plot(historico1.history['loss'])
    plt.plot(historico1.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Validação'], loc='upper left')
    plt.savefig("predictEt_loss.png")
    plt.clf()


if __name__ == "__main__":
    main()