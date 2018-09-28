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

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
		
if __name__ == "__main__":

    #circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw', 'Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw',
    #           'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']

    #circuitos = ['Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw']
    #circuitos = ['CTSV mc + 4bitPRBS [FALHA].raw']
    #circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw']
    circuitos = ['Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw']
    
    classifiers = [DecisionTreeClassifier(random_state=20),AdaBoostClassifier(random_state=20),
                       svm.SVC(kernel='linear', C=1, random_state=20),RandomForestClassifier(random_state=20),
                       GaussianNB(),KNeighborsClassifier(),SGDClassifier(random_state=20),
                       LogisticRegression(random_state=20)]
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
        # =-=-=-=-=-=-=-=-
        # início da leitura do arquivo
        # =-=-=-=-=-=-=-=-

        verifica = pd.read_csv("verifica_Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 02sraw.csv.csv",header=None, low_memory = False)
        #print(verifica,"\n",verifica[3],"\n",verifica.iloc[3])
        
        limite = 18
        dataSize = 3300
        k = 8
        zeros = np.zeros((20,), dtype=int)
        modas = pd.DataFrame({})
        hits = pd.DataFrame({})
        lines = pd.DataFrame({})

        verifica = pd.DataFrame(verifica)

        for i,clf in enumerate(classifiers):
            modaName = ("moda{}".format(clf.__class__.__name__))
            hitName = ("accuracy{}".format(clf.__class__.__name__))
            lineName = ("line{}".format(clf.__class__.__name__))
            modas[modaName] = zeros
            hits[hitName] = zeros
            lines[lineName] = verifica.iloc[i]
			
            for m in range(0, len(classifiers)):
                modas[modaName][m] = lines[lineName][m * 300:300 + m * 300].mode()[0]
                print("modas: \n{} :: {} :: {}\n".format(m,modas[modaName].name,modas[modaName][m]))
                for n in range((m*300),(300+m*300)):
                    if lines[lineName][n] == modas[modaName][m]:
                        hits[hitName][m] += 1
        f = lambda x: round(x*100./300.,2)
        hits = hits.apply(f)
        print("acurácia:\n{}".format(hits))

        k = 0
        conjunto = []
        conjunto1 = []
        verificacao = np.zeros((10, dataSize))
        dadosReduzidos = []
        dictData = {}
        df1 = pd.DataFrame()
        df = pd.DataFrame()
        dfTime = pd.DataFrame()
        listaFinal, dados = [], []
		