#Codigo principal
import pandas as pd
import LTSpice_RawRead as LTSpice
import tslearn
import matplotlib.pyplot as plt
import numpy as np
import visuals as vs
import random


if __name__ == "__main__":
    circuitos = ['Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw','Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']
    for circuito in circuitos:
        saida,  dados, time = LTSpice.principal(circuito)
        print("leu")
        MaiorIndice = 0
        for dado in dados:
            if len(dado) > MaiorIndice:
                MaiorIndice = len(dado)


        matriz = np.zeros((MaiorIndice, len(dados)))

        i = 0
        j = 0
        for k in range(0, len(saida._traces[10].data)):
            matriz[i][j] = saida._traces[10].data[k]
            if ((saida._traces[10].axis.data[k]) == 0.0) and (k != 0):
                if ((saida._traces[10].axis.data[k - 1]) != 0.0):
                    j += 1
                    i = 0
                else:
                    i += 1
            else:
                i += 1
        file_name = circuito.replace('.raw','.csv')
        dadosOriginais = pd.DataFrame(matriz)
        dadosOriginais.to_csv(file_name, index=False, header=None, sep=';')
        print("escreveu")



