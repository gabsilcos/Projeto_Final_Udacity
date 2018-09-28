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
    #           'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']

    #circuitos = ['Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw']
    #circuitos = ['CTSV mc + 4bitPRBS [FALHA].raw']
    circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw']
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
        plotTargets = pd.DataFrame({})
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
                    if ((saida._traces[voutIndex].axis.data[k - 1]) != 0.0):
                        j += 1
                        i = 0
                    else:
                        i += 1
                else:
                    i += 1
			
            #print("matriz: \n {}".format(matriz))
            dadosOriginais = pd.DataFrame(matriz)
            dadosOriginais.to_csv(csv_name, index = False, header=False)
			
        else:
            print("Obtendo dados do arquivo '{}' .".format(csv_name))
            dadosOriginais = pd.read_csv(csv_name,header=None, low_memory = False)
			
        print("Leitura do arquivo terminada.\nSalvando características do circuito...")

        circuito = re.sub('\.', '', circuito)
        circuito = re.sub(' ', '_', circuito)
		
        pltName = ("Dados Originais {}".format(circuito))
        plotTargets[pltName] = [dadosOriginais]

        # =-=-=-=-=-=-=-=-
        # Aplicação do Paa
        # =-=-=-=-=-=-=-=-
        print("\nIniciando a aplicação do PAA")
        n_paa_segments = 100
        dadosPaa = AuxiliaryFunctions.ApplyPaa(n_paa_segments,dadosOriginais,circuito)
        dataSize = dadosPaa.shape[0]
		
        pltName = "PAA_{}".format(circuito)
        #plotTargets[pltName] = dadosPaa
        plotTargets[pltName] = [dadosPaa]
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # Aplicação do PCA
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        print("\nIniciando a aplicação do PCA")
        ran = np.random.randint(dadosPaa.shape[0], size=(int(0.1 * dadosPaa.shape[0])))
        samples = pd.DataFrame(dadosPaa.loc[ran], columns=dadosPaa.keys()).reset_index(
            drop=True)  # amostras para treino

        reduced_data, pca_samples = AuxiliaryFunctions.ApplyPca(dadosPaa, samples,circuito)
        pltName = "PCA_{}".format(circuito)
        #plotTargets[pltName] = reduced_data.T
        plotTargets[pltName] = [reduced_data.T]
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição supervisionado
        # modelo: 8 modelos diferentes; em destaque: NaiveBayes
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        print("\nIniciando a aplicação dos métodos de aprendizagem supervisionados")
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.linear_model import LogisticRegression

        classifiers = [DecisionTreeClassifier(random_state=20),AdaBoostClassifier(random_state=20),
                       svm.SVC(kernel='linear', C=1, random_state=20),RandomForestClassifier(random_state=20),
                       GaussianNB(),KNeighborsClassifier(),SGDClassifier(random_state=20),
                       LogisticRegression(random_state=20)]
        k=0
        #classifiers = [GaussianNB()]
        for clf in classifiers:
            acc_train_results, acc_test_results, \
            fscore_train_results, fscore_test_results, \
            clfs = AuxiliaryFunctions.SupervisedPreds(dadosPaa, clf)

            print("Acurácia teste: {}\t Acurácia treino: {}\nFscore teste: {}\t Fscore treino: {}\n".format(
                acc_test_results, acc_train_results, fscore_test_results, fscore_train_results))
            for ct in range(10):
                rd = np.random.randint(0, dataSize)
                print("Predição do ponto {}: {}".format(rd, clfs.predict(dadosPaa.iloc[rd, :].values.reshape(1, -1))))

            for j in range(dataSize):
                verificacao[k][j]= clfs.predict(dadosPaa.iloc[j, :].values.reshape(1, -1))
            
            clfName = clf.__class__.__name__
            #plotTargets[clf.__class__.__name__] = verificacao[k]
            plotTargets[clfName] = [verificacao[k]]
            k+=1

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição não supervisionado
        # modelo: Gaussian Mixed Models
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        '''
        print("\nIniciando a aplicação dos métodos de aprendizagem não-supervisionados")
        from sklearn.mixture import GMM  # importar outro método no lugar do GMM, talvez o dbscan

        range_n_components = list(range(2, 12))
        clusterers = [GMM()]
        for clt in clusterers:
            print("Classificador: {}".format(clt.__class__.__name__))
            clts, preds = AuxiliaryFunctions.UnsupervisedPreds(reduced_data, pca_samples, clt, range_n_components)

            for ct in range(10):
                rd = np.random.randint(0, dataSize)
                print("Predição do ponto {}: {}".format(rd, clts.predict(reduced_data.iloc[rd, :].values.reshape(1, -1))))

            for j in range(dataSize):
                verificacao[k][j]= clts.predict(reduced_data.iloc[j, :].values.reshape(1, -1))
            k+=1
        '''
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição não supervisionado
        # modelo: Kmeans
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        ''''
        print("Classificador: KMeans")
        preds, clts = AuxiliaryFunctions.UnsupervisedKmeans(reduced_data,pca_samples)

        #print("k do kmeans: {}".format(k))
        for ct in range(10):
            rd = np.random.randint(0, dataSize)
            print("Predição de {}: {}".format(rd, clts.predict(reduced_data.iloc[rd, :].values.reshape(1, -1))))

        for j in range(dataSize):
            verificacao[k][j] = clts.predict(reduced_data.iloc[j, :].values.reshape(1, -1))
        '''
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação dos teste de validação de resultado
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        limite = int(dataSize/300)
        zeros = np.zeros((20,), dtype=int)
        modas = pd.DataFrame({})
        hits = pd.DataFrame({})
        lines = pd.DataFrame({})

        verifica = pd.DataFrame(verificacao)
        verificaName = ("verifica_{}".format(csv_name))
        verifica.to_csv(verificaName, index = False, header=False)

        for i,clf in enumerate(classifiers):
            modaName = ("moda{}".format(clf.__class__.__name__))
            hitName = ("accuracy{}".format(clf.__class__.__name__))
            lineName = ("line{}".format(clf.__class__.__name__))
            modas[modaName] = zeros
            hits[hitName] = zeros
            lines[lineName] = verifica.iloc[i]

            for m in range(0, limite):
                modas[modaName][m] = lines[lineName][m * 300:300 + m * 300].mode()[0]
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

        print("plot targets:\n{}".format(plotTargets))
        for i,plot in enumerate(plotTargets):
            fig = plt.figure()
            print("Plot ",i," :",plot,"\n",plotTargets[plot],"\n")
            plt.plot([plotTargets[plot]], '*')
            try:plt.savefig(plot, bbox_inches='tight')
            except: plt.savefig(plot)
			
			