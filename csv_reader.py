import sys
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)

arquivo = 'Sallen Key mc + 4bitPRBS [FALHA]raw.csv'
df = pd.read_csv(arquivo, header = None, sep = ';', low_memory = False)
df = df.replace(',','.')

print("dataframe lido:\n {}".format(df))

plt.show()