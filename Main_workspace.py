import AuxiliaryFunctions
import pandas as pd
from pandas import DataFrame
import re
import os

import LTSpice_RawRead as LTSpice
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn import svm

if __name__ == "__main__":

    #circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw', 'Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw',
    #             'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']

    # circuitos = ['Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw']
    circuitos = ['CTSV mc + 4bitPRBS [FALHA].raw']
    #circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw']
    #circuitos = ['REDUX.raw']

    conjunto = []
    conjunto1 = []
    verificacao = np.zeros((10, 6000))
    dadosReduzidos = []
    dictData = {}
    df1 = pd.DataFrame()
    df = pd.DataFrame()
    dfTime = pd.DataFrame()
    listaFinal, dados = [], []
    n_ts, sz, d = 1, 100, 1
    matriz = None

    for circuito in circuitos:
        print("Circuito: {}".format(circuito))
        csv_name = re.sub('\.', '', circuito)
        csv_name = "{}.csv".format(csv_name)
        plotTargets = {}
        pltName = ''
        # =-=-=-=-=-=-=-=-
        # início da leitura do arquivo
        # =-=-=-=-=-=-=-=-

        if not os.path.isfile(csv_name):
            print("Obtendo dados do arquivo '{}'...".format(circuito))
            saida, dados, time = LTSpice.principal(circuito)
            conjunto.append(saida)
            voutIndex = AuxiliaryFunctions.findVout(saida._traces)

            #print("dados: \n{}".format(dados))
            MaiorIndice = 0
            for dado in dados:
                if len(dado) > MaiorIndice:
                    MaiorIndice = len(dado)

            matriz = np.zeros((MaiorIndice, len(dados)))

            i = 0
            j = 0
            for k in range(0, len(saida._traces[voutIndex].data)):
                matriz[i][j] = saida._traces[voutIndex].data[k]
                if ((saida._traces[voutIndex].axis.data[k]) == 0.0) and (k != 0):
                    print("k: ", k)
                    print("onde o k tá quebrando: ",saida._traces[voutIndex].axis.data[k])
                    if ((saida._traces[voutIndex].axis.data[k - 1]) != 0.0):
                        j += 1
                        i = 0
                    else:
                        i += 1
                else:
                    i += 1
            dadosOriginais = pd.DataFrame(matriz)
            dadosOriginais.to_csv(csv_name, index=False, header=False, sep=';')
        else:
            print("Obtendo dados do arquivo '{}' .".format(csv_name))
            try: dadosOriginais = pd.read_csv(csv_name, header=None, low_memory=False, sep=',')
            except: dadosOriginais = pd.read_csv(csv_name, header=None, low_memory=False, sep=';')

        print("Leitura do arquivo terminada.\nSalvando características do circuito...")

        circuito = re.sub('\.', '', circuito)
        circuito = re.sub(' ', '_', circuito)

        #print("dados originais: \n{}".format(dadosOriginais))
        pltName = ("Dados Originais [{}]".format(circuito))
        plotTargets[pltName] = dadosOriginais
        plt.plot(dadosOriginais.T)
        #plt.show()

        classificacao = []
        for i in range(0, int(dadosOriginais.T.shape[0] / 300)):  # gambiarra para confirmação binária de acerto
            classificacao += [i + 1] * 300
        for i,clas in enumerate(classificacao):
            print("classificacao {}: {}".format(i,clas))
