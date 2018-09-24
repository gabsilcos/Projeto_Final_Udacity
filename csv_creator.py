import sys
import re
import os

import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt

import LTSpice_RawRead as LTSpice

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)
####

matriz = None

####


circuito = 'CTSV mc + 4bitPRBS [FALHA].raw'
csv = re.sub('\.', '', circuito)
csv = "{}.csv".format(csv)

print("Obtendo dados do arquivo '{}' .".format(circuito))
saida, dados, time = LTSpice.principal(circuito)
conjunto.append(saida)
MaiorIndice = 0
for dado in dados:
    if len(dado) > MaiorIndice:
        MaiorIndice = len(dado)

print("\n\n")
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

print("matriz: \n {}".format(matriz))


arquivo = 'CTSV mc + 4bitPRBS [FALHA]raw.csv'
#df = pd.read_csv(arquivo, header = None, names=['Country', 'Capital', 'Population'])

#print("dataframe lido:\n {}".format(df))

plt.show()