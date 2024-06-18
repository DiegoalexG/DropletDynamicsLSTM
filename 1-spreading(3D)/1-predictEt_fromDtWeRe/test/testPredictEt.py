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
    model = load_model('predictEt.keras')
    pulo = 100
    
    # Calculate model score
    score = model.evaluate(entrada_test, saida_test, verbose=2)
    print('\n\nScore =', score, '\n\n')

    # Apply prediction on the test data
    saida_rede = model.predict(entrada_test)

    # Undo the standardization for input data
    for i in range(len(entrada_test[0][0])):
        entrada_test[:, :, i] = entrada_test[:, :, i] * desvio_entrada[i] + media_entrada[i]

    # Undo the standardization for output data
    for i in range(len(saida_test[0][0])):
        saida_test[:, :, i] = saida_test[:, :, i] * desvio_saida[i] + media_saida[i]
        saida_rede[:, :, i] = saida_rede[:, :, i] * desvio_saida[i] + media_saida[i]

    # Divide by the total energy at time 0
    for i in range(len(saida_test)):
        saida_test[i] /= (saida_test[i, 0, 0] + saida_test[i, 0, 1] + saida_test[i, 0, 2])
        saida_rede[i] /= (saida_rede[i, 0, 0] + saida_rede[i, 0, 1] + saida_rede[i, 0, 2])

    # R2-score
    r2 = skm.r2_score(saida_test.flatten(), saida_rede.flatten())
    print("R2-score:", r2)

    # Root Mean Square Error
    rmse = skm.mean_squared_error(saida_test.flatten(), saida_rede.flatten(), squared=False)
    print("RMSE:", rmse)

    # Normalized Root Mean Square Error
    nrmse = rmse / (np.max(saida_test) - np.min(saida_test))
    print("NRMSE:", nrmse)

    # Changes plot appearence
    mpl.rcParams.update({'text.color': "#32292F",
                         'axes.labelcolor': "#32292F",
                         'xtick.color': "#32292F",
                         'ytick.color': "#32292F",
                         'text.usetex': True})

    # Used to group subsamples into a big sample (1000 time steps)
    num = 1000 // pulo
    n = len(saida_test) // num

    instancias = np.arange(1, n + 1, 1)

    r2s = [skm.r2_score(saida_test[i*num:i*num + num].flatten(), saida_rede[i*num:i*num + num].flatten()) for i in range(n)]
    rmse = [skm.mean_squared_error(saida_test[i*num:i*num + num].flatten(), saida_rede[i*num:i*num + num].flatten(), squared=False) for i in range(n)]
    nrmse = rmse / (np.max(saida_test) - np.min(saida_test))

    print(np.min(r2s), np.max(r2s))
    print(np.min(rmse), np.max(rmse))

    # Plot R2-score for each sample
    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(instancias, r2s, color="#ff3333")
    plt.xlabel('Instance', fontsize=18)
    plt.xticks(instancias, fontsize=18)
    plt.ylabel('$R^2$', fontsize=18)
    plt.yticks(fontsize=18)
    fig.tight_layout(pad=1.5)
    plt.savefig("Images/r2s.png")
    plt.clf()
    plt.close()

    # Plot RMSE for each sample
    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(instancias, rmse, color="#ff3333")
    plt.xlabel('Instance', fontsize=18)
    plt.xticks(instancias, fontsize=18)
    plt.ylabel('RMSE', fontsize=18)
    plt.yticks(fontsize=18)
    fig.tight_layout(pad=1.5)
    plt.savefig("Images/rmse.png")
    plt.clf()
    plt.close()

    # Plot NRMSE for each sample
    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(instancias, nrmse, color="#ff3333")
    plt.xlabel('Instance', fontsize=18)
    plt.xticks(instancias, fontsize=18)
    plt.ylabel('NRMSE', fontsize=18)
    plt.yticks(fontsize=18)
    fig.tight_layout(pad=1.5)
    plt.savefig("Images/nrmse.png")
    plt.clf()
    plt.close()

    # Plot predictions for each sample
    for i in range(n):
        tempos = np.linspace(0, 50, 1000) / entrada_test[i*int(1000 / pulo), 0, 0]
        
        fig = plt.figure(figsize=(12, 4)) 

        ec0 = [saida_test[i*int(1000 / pulo), :pulo, 0]]
        ec0_aprox = [saida_rede[i*int(1000 / pulo), :pulo, 0]]
        for j in range(1, int(1000 / pulo)):
            ec0.append(saida_test[i*int(1000 / pulo) + j, :pulo, 0])
            ec0_aprox.append(saida_rede[i*int(1000 / pulo) + j, :pulo, 0])
        ec0 = np.array(ec0)
        ec0_aprox = np.array(ec0_aprox)

        # Guarda os dados em um dataframe
        d = {"ec0": ec0.flatten(), "ec0_aprox": ec0_aprox.flatten()}
        df = pd.DataFrame(d)

        fig.add_subplot(1, 3, 1) 
        plt.plot(tempos, df["ec0"], color="#575366", zorder=1, label="Expected")
        plt.plot(tempos, df["ec0_aprox"], color="#06A77D", zorder=2)
        plt.plot(tempos[:2], df["ec0_aprox"][:2], marker='o', mfc='none', color="#06A77D", zorder=2, label="Predicted")
        plt.scatter(tempos[::25], df["ec0_aprox"][::25], facecolors='none', edgecolors='#06A77D')
        plt.xlabel('$\\bar{t}$', fontsize=18)
        plt.xticks(fontsize=18)
        plt.ylabel('$\\bar{E}_k$', fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=14)

        es0 = [saida_test[i*int(1000 / pulo), :pulo, 1]]
        es0_aprox = [saida_rede[i*int(1000 / pulo), :pulo, 1]]
        for j in range(1, int(1000 / pulo)):
            es0.append(saida_test[i*int(1000 / pulo) + j, :pulo, 1])
            es0_aprox.append(saida_rede[i*int(1000 / pulo) + j, :pulo, 1])
        es0 = np.array(es0)
        es0_aprox = np.array(es0_aprox)

        # Guarda os dados em um dataframe
        d = {"es0": es0.flatten(), "es0_aprox": es0_aprox.flatten()}
        df = pd.DataFrame(d)

        fig.add_subplot(1, 3, 2) 
        plt.plot(tempos, df["es0"], color="#575366", zorder=1, label="Expected")
        plt.plot(tempos, df["es0_aprox"], color="#5762D5", zorder=2)
        plt.plot(tempos[:2], df["es0_aprox"][:2], marker='o', mfc='none', color="#5762D5", zorder=2, label="Predicted")
        plt.scatter(tempos[::25], df["es0_aprox"][::25], facecolors='none', edgecolors='#5762D5')
        plt.xlabel('$\\bar{t}$', fontsize=18)
        plt.xticks(fontsize=18)
        plt.ylabel('$\\bar{E}_s$', fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=14)

        ed0 = [saida_test[i*int(1000 / pulo), :pulo, 2]]
        ed0_aprox = [saida_rede[i*int(1000 / pulo), :pulo, 2]]
        for j in range(1, int(1000 / pulo)):
            ed0.append(saida_test[i*int(1000 / pulo) + j, :pulo, 2])
            ed0_aprox.append(saida_rede[i*int(1000 / pulo) + j, :pulo, 2])
        ed0 = np.array(ed0)
        ed0_aprox = np.array(ed0_aprox)

        # Guarda os dados em um dataframe
        d = {"ed0": ed0.flatten(), "ed0_aprox": ed0_aprox.flatten()}
        df = pd.DataFrame(d)

        fig.add_subplot(1, 3, 3) 
        plt.plot(tempos, df["ed0"], color="#575366", zorder=1, label="Expected")
        plt.plot(tempos, df["ed0_aprox"], color="#C1292E", zorder=2)
        plt.plot(tempos[:2], df["ed0_aprox"][:2], marker='o', mfc='none', color="#C1292E", zorder=2, label="Predicted")
        plt.scatter(tempos[::25], df["ed0_aprox"][::25], facecolors='none', edgecolors='#C1292E')
        plt.xlabel('$\\bar{t}$', fontsize=18)
        plt.xticks(fontsize=18)
        plt.ylabel('$\\bar{E}_d$', fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=14)

        plt.suptitle("Re = " + str(np.round(entrada_test[i*int(1000 / pulo), 0, 1], 2)) + 
                    ", We = " + str(np.round(entrada_test[i*int(1000 / pulo), 0, 2], 2)), fontsize=18)
        fig.tight_layout(pad=1.5)
        plt.savefig("Images/predictEt_aprox_shape_"+str(i+1)+".png")
        plt.clf()
        plt.close()


if __name__ == "__main__":
    main()