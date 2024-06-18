import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Reshape
from keras.optimizers import Adamax
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

def slicing(entrada, saida, tam_sequencia, pulo):
    x, y = [], []
    
    for i in range(len(entrada)):
        for j in range(0, len(entrada[0]) - tam_sequencia + 1, pulo):
            _x = entrada[i, j:(j + tam_sequencia), :]
            _y = saida[i, :]
            x.append(_x)
            y.append(_y)

    return [np.array(x), np.array(y)]

def main():
    #------------------------------------------------------------------------------------------------------#
    #------------------------------------ Reads and standardizes the data ---------------------------------#
    #------------------------------------------------------------------------------------------------------#
    arq = np.load("Time_series_collision.npz", allow_pickle=True)
    entrada = arq["TS"][:, :, :2]
    saida = np.array([arq["Re"], arq["We"]]).T

    media_entrada, desvio_entrada = [], []
    for i in range(len(entrada[0][0])):
        media_entrada.append(np.mean(entrada[:, :, i]))
        desvio_entrada.append(np.std(entrada[:, :, i]))
        entrada[:, :, i] = (entrada[:, :, i] - media_entrada[i]) / desvio_entrada[i]
    media_entrada = np.array(media_entrada)
    desvio_entrada = np.array(desvio_entrada)

    media_saida, desvio_saida = [], []
    for i in range(len(saida[0])):
        media_saida.append(np.mean(saida[:, i]))
        desvio_saida.append(np.std(saida[:, i]))
        saida[:, i] = (saida[:, i] - media_saida[i]) / desvio_saida[i]
    media_saida = np.array(media_saida)
    desvio_saida = np.array(desvio_saida)


    #------------------------------------------------------------------------------------------------------#
    #--------------------------------- Splits data for training and testing -------------------------------#
    #------------------------------------------------------------------------------------------------------#
    entrada_train = entrada[:int(len(entrada) * 0.7)]
    saida_train = saida[:int(len(saida) * 0.7)]

    entrada_test = entrada[int(len(entrada) * 0.7):]
    saida_test = saida[int(len(saida) * 0.7):]


    #------------------------------------------------------------------------------------------------------#
    #-------------------------------------------- Slice the data ------------------------------------------#
    #------------------------------------------------------------------------------------------------------#
    passos = 100
    pulo = 100
    [entrada_train, saida_train] = slicing(entrada_train, saida_train, passos, pulo)
    [entrada_test, saida_test] = slicing(entrada_test, saida_test, passos, pulo)


    #------------------------------------------------------------------------------------------------------#
    #------------------ Save the test data and the parameters to undo the standarization ------------------#
    #------------------------------------------------------------------------------------------------------#    
    save_dict = {
        'entrada_train': entrada_train,
        'saida_train': saida_train,
        'entrada_test': entrada_test,
        'saida_test': saida_test,
        'media_entrada': media_entrada,
        'desvio_entrada': desvio_entrada,
        'media_saida': media_saida,
        'desvio_saida': desvio_saida
    }
    np.savez('dados_teste', **save_dict)


    #------------------------------------------------------------------------------------------------------#
    #----------------------------------------- LSTM architecture ------------------------------------------#
    #------------------------------------------------------------------------------------------------------#
    taxa_aprendizado = 0.0001
    epocas = 5000
    batch = 32

    tf.keras.utils.set_random_seed(42)

    model = Sequential()

    model.add(LSTM(32, return_sequences=True, input_shape=(passos, 2), name="Oculta1"))
    model.add(LSTM(32, return_sequences=True, name="Oculta2"))
    model.add(LSTM(32, name="Oculta3"))
    model.add(Dense(2, activation="linear", name="Output"))

    optimizer = Adamax(learning_rate = taxa_aprendizado)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])

    model.summary()

    mcp_save = ModelCheckpoint('predictReWe.keras', save_best_only=True, monitor='loss', mode='min')
    erl_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5000)

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    historico1 = model.fit(entrada_train, saida_train, epochs=epocas,
                           batch_size=batch, callbacks=[mcp_save, erl_stop], verbose=1)


if __name__ == "__main__":
    main()