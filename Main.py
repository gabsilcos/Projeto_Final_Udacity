import AuxiliaryFunctions
import pandas as pd
from pandas import DataFrame
import re
import os

import LTSpice_RawRead as LTSpice
import matplotlib.pyplot as plt
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":

    #circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw', 'Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw',
    #           'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']

    # circuitos = ['CTSV mc + 4bitPRBS [FALHA].raw','Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw']

    #circuitos = ['Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw']
    circuitos = ['CTSV mc + 4bitPRBS [FALHA].raw']
    #circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw']
    #circuitos = ['Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw']

    saidaCompleta = []
    matriz = None

    for circuito in circuitos:
        print("\n\nCircuito: {}".format(circuito))
        csv_name = re.sub('\.', '', circuito)
        csv_name = "{}.csv".format(csv_name)
        plotTargets = {}
        pltName = ''
        # =-=-=-=-=-=-=-=-
        # início da leitura do arquivo
        # =-=-=-=-=-=-=-=-

        if not os.path.isfile(csv_name):
            print("Obtendo dados do arquivo '{}'...".format(circuito))
            saida, dados, time = AuxiliaryFunctions.LTSpiceReader(circuito)
            saidaCompleta.append(saida)
            voutIndex = AuxiliaryFunctions.findVout(saida._traces)

            # print("dados: \n{}".format(dados))
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

            dadosOriginais = pd.DataFrame(matriz)
            dadosOriginais.to_csv(csv_name, index=False, header=False)

        else:
            print("Obtendo dados do arquivo '{}' .".format(csv_name))
            dadosOriginais = pd.read_csv(csv_name, header=None, low_memory=False)

        print("Leitura do arquivo terminada.\nSalvando características do circuito...")

        circuito = re.sub('\.', '', circuito)
        circuito = re.sub(' ', '_', circuito)

        pltName = ("Dados_Originais")
        plotTargets[pltName] = dadosOriginais

        # =-=-=-=-=-=-=-=-
        # Aplicação do Paa
        # =-=-=-=-=-=-=-=-
        print("\nIniciando a aplicação do PAA")
        n_paa_segments = 100
        dadosPaa = AuxiliaryFunctions.ApplyPaa(n_paa_segments, dadosOriginais)
        dataSize = dadosPaa.shape[0]

        dadosPaa.to_csv("a.csv",index=False, header=False,sep=';')

        pltName = "PAA"
        plotTargets[pltName] = dadosPaa

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição supervisionado
        # modelo: 8 modelos diferentes; em destaque: NaiveBayes
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        print("\nIniciando a aplicação dos métodos de aprendizagem supervisionados")
        from sklearn.svm import SVC
        from sklearn import svm
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.linear_model import LogisticRegression
        '''
        classifiers = [DecisionTreeClassifier(random_state=20), AdaBoostClassifier(random_state=20),
                       svm.SVC(kernel='linear', C=1, random_state=20), RandomForestClassifier(random_state=20),
                       GaussianNB(), KNeighborsClassifier(),
                       SGDClassifier(max_iter=5, random_state=20),
                       AdaBoostClassifier(base_estimator=RandomForestClassifier(random_state=20), random_state=20),
                       LogisticRegression(random_state=20)]
        '''

        #classifiers = [GaussianNB()]#,AdaBoostClassifier(base_estimator=RandomForestClassifier(random_state=20), random_state=20),
        #               RandomForestClassifier(random_state=20)]
        classifiers = [AdaBoostClassifier(random_state=20)]
        k = 0
        preds = np.zeros((len(classifiers), dataSize))

        for i,clf in enumerate(classifiers):
            clfName = ("[{}]_{}".format(i,clf.__class__.__name__))
            test_score, cnf_matrix,\
            clfs = AuxiliaryFunctions.SupervisedPreds(dadosPaa, clf,[],0)

            cm = AuxiliaryFunctions.confusionMatrixPlot(cnf_matrix,circuito,clfName,dataSize)
            print("Score de teste: {}\nConfusion Matrix:\n{}\n".format(test_score, cm))

            for j in range(dataSize):
                preds[k][j] = clfs.predict(dadosPaa.iloc[j, :].values.reshape(1, -1))

            plotTargets[clfName] = preds[k]
            k += 1

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # OTIMIZAÇÃO DE RESULTADOS
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        clf = AdaBoostClassifier(random_state=20)
        print("Aplicando tuning do modelo...")
        parameters = {'learning_rate': [0.1, 0.5, 1.],
                      'random_state':[20,40,120,360]}

        results = AuxiliaryFunctions.SupervisedPreds(dadosPaa, clf, parameters, 1)

        print("Best Score: ", results['best_score'])
        print("Best Parameters: ", results['best_params'])
        print("Unoptimized model\n------")
        print("F-score on testing data: {:.4f}".format(results['test_score']))
        print("\nOptimized Model\n------")
        print("Final F-score on the testing data: {:.4f}".format(results['final_test_score']))
        adjusted_predictions = results['best_clf'].predict(dadosOriginais.T)
        plotTargets["PREDIÇÕES FINAIS"] = adjusted_predictions.T

        print("Confusion Matrix:\n{}\n".format(results['cnf_matrix']))

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação dos teste de validação de resultado
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        predTable = pd.DataFrame(preds)

        fileName = ("verifica_{}".format(csv_name))
        predTable.to_csv(fileName, index=False, header=False, sep=';')

        for i, key in enumerate(plotTargets.keys()):
            fig = plt.figure(figsize=(15, 15))
            plt.plot(plotTargets[key], 'o')
            print("Plotando gráficos de ",key, "...")
            try:
                plt.savefig("{}_{}".format(circuito, key), bbox_inches='tight')
            except:
                plt.savefig(key)
