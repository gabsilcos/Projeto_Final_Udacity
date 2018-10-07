import LTSpice_RawRead
import re
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import collections

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.metrics import silhouette_score,r2_score,confusion_matrix



def LTSpiceReader(Circuito):
    # if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    if Circuito == 'CTSV mc + 4bitPRBS [FALHA].raw' or Circuito == 'REDUX.raw':
        Variavel = 'V(bpo)'
    else:
        Variavel = 'V(vout)'
    if len(sys.argv) > 1:
        raw_filename = sys.argv[1]
    else:
        raw_filename = Circuito

    LTR = LTSpice_RawRead.LTSpiceRawRead(raw_filename, traces_to_read=Variavel, loadmem=True)
    # print(LTR.get_trace_names())
    fig0 = plt.figure()
    plt.title("Dados Brutos")
    for trace in LTR.get_trace_names():
        print("\nLendo grandeza: {}".format(LTR.get_trace(trace).name))
        Vo = LTR.get_trace(Variavel)
        x = LTR.get_trace(0)  # Zero is always the X axis
        steps = LTR.get_steps()
        Dados = []
        time = []
        for step in range(len(steps)):
            ValueVar = Vo.get_wave(step)
            Dados.append(ValueVar)
            valueTime = x.get_wave(step)
            time.append(valueTime)
            plt.plot(valueTime, ValueVar)
        print("\"{}\" lido.".format(LTR.get_trace(trace).name))

    name = "Brutos_{}".format(Circuito)
    name = re.sub('\.', '', name)
    plt.savefig(name, bbox_inches='tight')
    print("Grandezas lidas.")
    return (LTR, Dados, time)


def ApplyPaa(n_paa_segments,df,ckt):
    circuito = ckt
    print("Quantidade de segmentos de PAA: {}".format(n_paa_segments))
    paa = PiecewiseAggregateApproximation(n_paa_segments)
    scaler = TimeSeriesScalerMeanVariance()
    dadosPaa = df
    for i in range(0, len(df)):
        dataset = scaler.fit_transform(df[i])
        dadosPaa[i] = paa.inverse_transform(paa.fit_transform(dataset))[0]
    dadosPaa = dadosPaa.T

    return dadosPaa


def SupervisedPreds(df,clf):
    '''
    Aplica um único método de cada vez.
    :param df: dataframe com dados em que se deseja aplicar o aprendizado
    :param clf: classifacador que se deseja usar no aprendizado
    :return: acurácias de treino e teste, f-beta scores de treino e teste e o objeto classificador
    '''

    import warnings
    warnings.filterwarnings("ignore")   #pra não cagar o meu log :D

    classificacao = []
    for i in range(0, int(df.shape[0] / 300)):  # gambiarra para confirmação binária de acerto
        classificacao += [i + 1] * 300

    classi = pd.DataFrame(classificacao)
    X_train, X_test, y_train, y_test = train_test_split(df, classi, test_size=0.25, random_state=0)
    print("Total training subjects: {}\nTotal testing subjects: {}".format(len(X_train),len(X_test)))
    #for i in range(len(X_test)):
    #    outlier = log_data[~((log_data[feature] >= minVal) & (log_data[feature] <= maxVal))]
    '''
    classifiers = [DecisionTreeClassifier(random_state=20),AdaBoostClassifier(random_state=20),
                   svm.SVC(kernel='linear', C=1, random_state=20),RandomForestClassifier(random_state=20),
                   GaussianNB(),KNeighborsClassifier(),SGDClassifier(random_state=20),
                   LogisticRegression(random_state=20)]
    '''
    print("\nClassificador: {}".format(clf.__class__.__name__))
    clf = clf.fit(X_train, y_train)
    clf_test_predictions = clf.predict(X_test)
    #clf_train_predictions = clf.predict(X_train)
    cnf_matrix = confusion_matrix(y_test, clf_test_predictions)

    try:
        #fscore_train_results = fbeta_score(y_train, clf_train_predictions, beta=0.5, average='macro')
        fscore_test_results = fbeta_score(y_test, clf_test_predictions, beta=0.5, average='macro')
        #return(fscore_train_results,fscore_test_results,cnf_matrix,clf)
        return(fscore_test_results,cnf_matrix,clf)
    except:
        #acc_train_results = r2_score(y_train, clf_train_predictions)
        acc_test_results = r2_score(y_test, clf_test_predictions)
        return (acc_test_results,cnf_matrix,clf)


def findVout(traces):
    voutIndex = None
    for i, trace in enumerate(range(0, len(traces))):
        if traces[trace].name == 'V(bpo)' or traces[trace].name == 'V(vout)':
            voutIndex = i

    return voutIndex
