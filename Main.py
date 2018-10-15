import AuxiliaryFunctions
import pandas as pd
import re
import os
import time as timer

import LTSpice_RawRead as LTSpice
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (30,15)

if __name__ == "__main__":
    inicio = timer.time()

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # raiz do projeto
    CSV_DIR = "{}\CSV\\".format(ROOT_DIR)  # dump de CSVs
    IMG_DIR = "{}\IMG\\".format(ROOT_DIR)  # dump de imagens

    '''
    Descomentar o circuito ou grupo de circuitos que será observado
    '''

    circuitos = ['Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw','Sallen Key mc + 4bitPRBS [FALHA].raw',
                 'Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw','CTSV mc + 4bitPRBS [FALHA].raw']

    # circuitos = ['Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw']
    # circuitos = ['CTSV mc + 4bitPRBS [FALHA].raw']
    # circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw']
    # circuitos = ['Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw']

    saidaCompleta = []
    matriz = None

    for circuito in circuitos:
        print("\n\nCircuito: {}".format(circuito))
        csv_name = re.sub('\.', '', circuito)
        csv_name = "{}.csv".format(csv_name)
        plotTargets = {}
        pltName = ''
        scores = {}
        # =-=-=-=-=-=-=-=-
        # início da leitura do arquivo
        # =-=-=-=-=-=-=-=-

        if not os.path.isfile("{}{}".format(CSV_DIR, csv_name)):
            iniRaw = timer.time()
            print("Obtendo dados do arquivo '{}'...".format(circuito))
            saida, dados, time = AuxiliaryFunctions.LTSpiceReader(circuito)
            saidaCompleta.append(saida)
            voutIndex = AuxiliaryFunctions.findVout(saida._traces)

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
            dadosOriginais.to_csv("{}{}".format(CSV_DIR, csv_name), index=False, header=False)
            print("Leitura do arquivo raw executada em: ",timer.time()-iniRaw," segundos")

        else:
            iniCsv = timer.time()
            print("Obtendo dados do arquivo '{}' .".format(csv_name))
            dadosOriginais = pd.read_csv("{}{}".format(CSV_DIR, csv_name), header=None, low_memory=False)
            print("Leitura do arquivo csv executada em: ", timer.time() - iniCsv," segundos")

        print("Salvando características do circuito...")

        circuito = re.sub('\.', '', circuito)
        circuito = re.sub(' ', '_', circuito)

        pltName = ("Dados_Originais")
        plotTargets[pltName] = dadosOriginais

        # =-=-=-=-=-=-=-=-
        # Aplicação do Paa
        # =-=-=-=-=-=-=-=-
        iniPaa = timer.time()
        print("\nIniciando a aplicação do PAA")
        n_paa_segments = 100
        dadosPaa = AuxiliaryFunctions.ApplyPaa(n_paa_segments, dadosOriginais)
        dataSize = dadosPaa.shape[0]
        print("Aplicação do Paa executada em: ", timer.time() - iniPaa)

        pltName = "PAA"
        plotTargets[pltName] = dadosPaa

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição supervisionado
        # modelo: diversos
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        print("\nIniciando a aplicação dos métodos de aprendizagem supervisionados")
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import BaggingClassifier

        classifiers = [DecisionTreeClassifier(random_state=20), AdaBoostClassifier(random_state=20),
                       SVC(random_state=20), RandomForestClassifier(random_state=20),
                       GaussianNB(), KNeighborsClassifier(),
                       SGDClassifier(max_iter=5, random_state=20),
                       AdaBoostClassifier(base_estimator=RandomForestClassifier(random_state=20), random_state=20),
                       LogisticRegression(random_state=20), BaggingClassifier()]

        k = 0
        preds = np.zeros((len(classifiers), dataSize))

        for i, clf in enumerate(classifiers):
            iniPreds = timer.time()
            clfName = ("[{}]_{}".format(i, clf.__class__.__name__))
            test_score, cnf_matrix, \
            clfs = AuxiliaryFunctions.SupervisedPreds(dadosPaa, clf, [], 0)

            cm = AuxiliaryFunctions.confusionMatrixPlot(cnf_matrix, circuito, clfName, dataSize)
            print("Score de teste: {}\nConfusion Matrix:\n{}\n".format(test_score, cm))
            scores[clfName] = test_score

            for j in range(dataSize):
                preds[k][j] = clfs.predict(dadosPaa.iloc[j, :].values.reshape(1, -1))

            print("Treino, teste e predições do {} executados em: {} segundos\n\n".format(clfName,timer.time() - iniPreds))
            plotTargets[clfName] = preds[k]
            k += 1

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # OTIMIZAÇÃO DE RESULTADOS
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        clf = AdaBoostClassifier(random_state=20)
        print("Aplicando tuning do modelo...")
        parameters = {'learning_rate': [0.1, 0.5, 1.],
                      'random_state': [20, 40, 120, 360]}

        iniGscv = timer.time()
        results = AuxiliaryFunctions.SupervisedPreds(dadosPaa, clf, parameters, 1)
        print("Otimização de parâmetros executada em: ", timer.time() - iniGscv," segundos")

        print("Melhor Score: ", results['best_score'])
        print("Melhores Parâmetros: ", results['best_params'])
        print("\n------\nModelo não-otimizado\n------")
        print("F-score dos dados de teste: {:.4f}".format(results['test_score']))
        print("\n------\nModelo otimizado\n------")
        print("F-score dos dados de teste: {:.4f}\n\n".format(results['final_test_score']))
        adjusted_predictions = results['best_clf'].predict(dadosOriginais.T)
        plotTargets["PREDICOES_FINAIS"] = adjusted_predictions.T

        cm = AuxiliaryFunctions.confusionMatrixPlot(results['cnf_matrix'], circuito, '{}_CM_GSCV'.format(circuito), dataSize)
        print("Confusion Matrix:\n{}\n".format(cm))

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # Exportação dos resultados
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        predTable = pd.DataFrame(preds)

        fileName = ("{}verifica_{}".format(CSV_DIR, csv_name))
        predTable.to_csv(fileName, index=False, header=False, sep=';')

        for i, key in enumerate(plotTargets.keys()):
            fig = plt.figure(figsize=(15, 15))
            plt.plot(plotTargets[key],",")
            plt.xlabel("Passo de simulação", fontsize=25)
            plt.ylabel("Grupo de Falha", fontsize=25)
            print("Plotando gráficos de ", key, "...")
            try:
                plt.savefig("{}{}_{}".format(IMG_DIR, circuito, key), bbox_inches='tight')
            except:
                plt.savefig(key)

        for i, key in enumerate(scores.keys()):
            print("Score do {}: {}".format(key,scores[key]))

    print("Executado em: ", timer.time()-inicio," segundos")