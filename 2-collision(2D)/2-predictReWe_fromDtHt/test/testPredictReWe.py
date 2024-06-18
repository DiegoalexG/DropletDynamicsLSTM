import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics as skm

def main():
    #------------------------------------------------------------------------------------------------------#
    #------------------------------------------ Reading the data ------------------------------------------#
    #------------------------------------------------------------------------------------------------------#
    dados = np.load('dados_teste.npz', allow_pickle='True')
    entrada_test = dados["entrada_test"]
    saida_test = dados["saida_test"]
    media_entrada = dados["media_entrada"]
    desvio_entrada = dados["desvio_entrada"]
    media_saida = dados["media_saida"]
    desvio_saida = dados["desvio_saida"]


    #------------------------------------------------------------------------------------------------------#
    #------------------------------------------ Testing the model -----------------------------------------#
    #------------------------------------------------------------------------------------------------------#    
    # Load the model
    model = load_model('predictReWe.keras')

    # Apply prediction on the test data
    saida_rede = model.predict(entrada_test)

    # Undo the standardization for output data
    for i in range(len(saida_test[0])):
        saida_test[:, i] = saida_test[:, i] * desvio_saida[i] + media_saida[i]
        saida_rede[:, i] = saida_rede[:, i] * desvio_saida[i] + media_saida[i]

    # Selects only the median from predicted values
    saida_test = saida_test[::10]
    saida_rede = np.array([np.median(saida_rede[i*10:(i+1)*10], axis=0) for i in range(len(saida_rede) // 10)])

    # R2-score
    r2s = skm.r2_score(saida_test.flatten(), saida_rede.flatten())
    print("R2-score2:", r2s)

    # Root Mean Square Error
    rmse = skm.mean_squared_error(saida_test.flatten(), saida_rede.flatten(), squared=False)
    print("RMSE2:", rmse)

    # Normalized Root Mean Square Error
    nrmse = rmse / (np.max(saida_test) - np.min(saida_test))
    print("NRMSE2:", nrmse)

    # Changes plot appearence
    mpl.rcParams.update({'text.color': "#32292F",
                         'axes.labelcolor': "#32292F",
                         'xtick.color': "#32292F",
                         'ytick.color': "#32292F",
                         'text.usetex': True})

    # Stores data in a dataframe
    d = {"Re": saida_test[:, 0], "Re_aprox": saida_rede[:, 0],
         "We": saida_test[:, 1], "We_aprox": saida_rede[:, 1]}
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
    plt.savefig("Images/predictReWe_direto_aprox.png")
    plt.clf()


if __name__ == "__main__":
    main()