import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics as skm

def main():
    #------------------------------------------------------------------------------------------------------#
    #------------------------------------------- Reading the data -----------------------------------------#
    #------------------------------------------------------------------------------------------------------#
    # Reads test data for the model that predicts energies
    dados1 = np.load('dados_teste1.npz', allow_pickle='True')
    entrada_test1 = dados1["entrada_test"]
    saida_test1 = dados1["saida_test"]
    media_entrada1 = dados1["media_entrada"]
    desvio_entrada1 = dados1["desvio_entrada"]
    media_saida1 = dados1["media_saida"]
    desvio_saida1 = dados1["desvio_saida"]

    # Reads test data for the model that predicts Re and We
    dados2 = np.load('dados_teste2.npz', allow_pickle='True')
    entrada_test2 = dados2["entrada_test"]
    saida_test2 = dados2["saida_test"]
    media_entrada2 = dados2["media_entrada"]
    desvio_entrada2 = dados2["desvio_entrada"]
    media_saida2 = dados2["media_saida"]
    desvio_saida2 = dados2["desvio_saida"]


    #------------------------------------------------------------------------------------------------------#
    #--------------------------------- Testing the energy prediction model --------------------------------#
    #------------------------------------------------------------------------------------------------------#
    # Load the model
    model1 = load_model('predictEt.keras')
    pulo = len(saida_test1[0])

    # Apply prediction on the test data
    saida_rede1 = model1.predict(entrada_test1)

    # Undo the standardization for input data
    for i in range(len(entrada_test1[0][0])):
        entrada_test1[:, :, i] = entrada_test1[:, :, i] * desvio_entrada1[i] + media_entrada1[i]

    # Undo the standardization for output data
    for i in range(len(saida_test1[0][0])):
        saida_test1[:, :, i] = saida_test1[:, :, i] * desvio_saida1[i] + media_saida1[i]
        saida_rede1[:, :, i] = saida_rede1[:, :, i] * desvio_saida1[i] + media_saida1[i]

    # R2-score
    r2s = skm.r2_score(saida_test1.flatten(), saida_rede1.flatten())
    print("R2-score:", r2s)

    # Root Mean Square Error
    rmse = skm.mean_squared_error(saida_test1.flatten(), saida_rede1.flatten(), squared=False)
    print("RMSE:", rmse)

    # Normalized Root Mean Square Error
    nrmse = rmse / (np.max(saida_test1) - np.min(saida_test1))
    print("NRMSE:", nrmse)

    # Changes plot appearence
    mpl.rcParams.update({'text.color': "#32292F",
                         'axes.labelcolor': "#32292F",
                         'xtick.color': "#32292F",
                         'ytick.color': "#32292F",
                         'text.usetex': True})

    
    #------------------------------------------------------------------------------------------------------#
    #-------------------------- Testing the dimensionless numbers prediction model ------------------------#
    #------------------------------------------------------------------------------------------------------#    
    entrada_test2 = np.array([saida_rede1[i] * (entrada_test2[i, 0, 0] + entrada_test2[i, 0, 1] + entrada_test2[i, 0, 2]) for i in range(len(saida_rede1))])

    # Load the model
    model2 = load_model('predictReWe.keras')

    # Apply prediction on the test data
    saida_rede2 = model2.predict(entrada_test2)

    # Undo the standardization for output data
    a, b = -1, 1
    for i in range(len(saida_test2[0])):
        saida_test2[:, i] = saida_test2[:, i] * desvio_saida2[i] + media_saida2[i]
        saida_rede2[:, i] = saida_rede2[:, i] * desvio_saida2[i] + media_saida2[i]

    # Selects only the median from predicted values
    saida_test2 = saida_test2[::10]
    saida_rede2 = np.array([np.median(saida_rede2[i*10:(i+1)*10], axis=0) for i in range(len(saida_rede2) // 10)])

    # R2-score
    r2s = skm.r2_score(saida_test2.flatten(), saida_rede2.flatten())
    print("R2-score2:", r2s)

    # Root Mean Square Error
    rmse = skm.mean_squared_error(saida_test2.flatten(), saida_rede2.flatten(), squared=False)
    print("RMSE2:", rmse)

    # Normalized Root Mean Square Error
    nrmse = rmse / (np.max(saida_test2) - np.min(saida_test2))
    print("NRMSE2:", nrmse)

    # Stores data in a dataframe
    d = {"Re": saida_test2[:, 0], "Re_aprox": saida_rede2[:, 0],
         "We": saida_test2[:, 1], "We_aprox": saida_rede2[:, 1]}
    df = pd.DataFrame(d)

    fig = plt.figure(figsize=(9, 4)) 

    # Compares Re X Re_approx
    fig.add_subplot(1, 2, 1) 
    plt.plot([np.min(df["Re"]), np.max(df["Re"])], [np.min(df["Re"]), np.max(df["Re"])], color="#575366", zorder=1, label="Expected")
    plt.scatter(df["Re"], df["Re_aprox"], facecolors='none', edgecolors='#06A77D', zorder=2, label="Obtained")
    plt.xlabel('$Re$', fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylabel('$Re_{\\mathrm{approx}}$', fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=14)


    # Compares We X We_approx
    fig.add_subplot(1, 2, 2) 
    plt.plot([np.min(df["We"]), np.max(df["We"])], [np.min(df["We"]), np.max(df["We"])], color="black", zorder=1, label="Expected")
    plt.scatter(df["We"], df["We_aprox"], facecolors='none', edgecolors='#5762D5', zorder=2, label="Obtained")
    plt.xlabel('$We$', fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylabel('$We_{\\mathrm{approx}}$', fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=14)

    fig.tight_layout(pad=1.5)
    plt.savefig("Images/predictReWe_aprox.png")
    plt.clf()


if __name__ == "__main__":
    main()