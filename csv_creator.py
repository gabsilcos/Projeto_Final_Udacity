import sys
import re
import os

import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation

import LTSpice_RawRead as LTSpice
import AuxiliaryFunctions

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)
####

conjunto = []
matriz = None

####


#circuito = 'REDUX.raw'
#circuito = 'CTSV mc + 4bitPRBS [FALHA].raw'
circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw', 'Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw',
               'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']
#circuitos = ['REDUX.raw']
circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw']

for circuito in circuitos:
    csv_name = re.sub('\.', '', circuito)
    csv_name = "{}.csv".format(csv_name)

    print("Obtendo dados do arquivo '{}' .".format(circuito))
    saida, dados, time = LTSpice.principal(circuito)
    conjunto.append(saida)

    voutIndex = AuxiliaryFunctions.findVout(saida._traces)

    MaiorIndice = 0
    for dado in dados:
        if len(dado) > MaiorIndice:
            MaiorIndice = len(dado)

    print("\n\n")
    matriz = np.zeros((MaiorIndice, len(dados)))

    i = 0
    j = 0
    for k in range(0, len(saida._traces[voutIndex].data)):
        matriz[i][j] = saida._traces[voutIndex].data[k]
        if ((saida._traces[voutIndex].axis.data[k]) == 0.0) and (k != 0):
            if ((saida._traces[voutIndex].axis.data[k - 1]) != 0.0):
                j += 1
                i = 0
            else:
                i += 1
        else:
            i += 1

    print("matriz: \n {}".format(matriz))

    dadosOriginais = pd.DataFrame(matriz)
    print("dados originais antes de salvar:\n {}".format(dadosOriginais))
    dadosOriginais.to_csv(csv_name, index=False, header=False, sep=';')
    #dadosOriginais.to_csv(csv_name)

    #dadosOriginais = pd.read_csv(csv_name, low_memory=False)
    dadosOriginais = pd.read_csv(csv_name, header=None, low_memory=False)
    print("dados originais depois de salvar:\n {}".format(dadosOriginais))
