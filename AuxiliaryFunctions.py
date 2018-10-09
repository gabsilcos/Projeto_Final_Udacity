import LTSpice_RawRead
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from sklearn.metrics import r2_score,confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) #raiz do projeto
IMG_DIR = "{}\IMG\\".format(ROOT_DIR) #dump de imagens

def LTSpiceReader(Circuito):
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
    fig0 = plt.figure()
    plt.title("Dados Brutos")
    for trace in LTR.get_trace_names():
        print("\nLendo grandeza: {}".format(LTR.get_trace(trace).name))
        Vo = LTR.get_trace(Variavel)
        x = LTR.get_trace(0)  # Zero é sempre o eixo X
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
    plt.savefig("{}{}".format(IMG_DIR,name), bbox_inches='tight')
    print("Grandezas lidas.")
    return (LTR, Dados, time)


def ApplyPaa(n_paa_segments,df):
    df = df.values.T.tolist()
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    dadosPaa = scaler.fit_transform(df)
    print("Quantidade de segmentos de PAA: {}".format(n_paa_segments))
    paa = PiecewiseAggregateApproximation(n_paa_segments)
    dadosPaa = paa.inverse_transform(paa.fit_transform(dadosPaa))

    df = pd.DataFrame()

    for i in range(len(dadosPaa.T)):
        for j in range(len(dadosPaa.T[0])):
            df[j] = dadosPaa.T[i][j]

    return df


def SupervisedPreds(df,clf,parameters,optimization):
    '''
    Aplica um único método de cada vez.
    :param df: dataframe com dados em que se deseja aplicar o aprendizado
    :param clf: classifacador que se deseja usar no aprendizado
    :param parameters: parâmetros para a aplicação do GridSearchCV
    :param optimization: flag sinalizando se é uma operação de otimização de algoritmos
    :return: acurácias de treino e teste, f-beta scores de treino e teste e o objeto classificador
    '''

    import warnings
    warnings.filterwarnings("ignore")

    classificacao = []
    for i in range(0, int(df.shape[0] / 300)):  # gabarito da simulação
        classificacao += [i + 1] * 300

    classi = pd.DataFrame(classificacao)
    X_train, X_test, y_train, y_test = train_test_split(df, classi, test_size=0.25, random_state=0)
    print("Total training subjects: {}\nTotal testing subjects: {}".format(len(X_train),len(X_test)))
    '''
    classifiers = [DecisionTreeClassifier(random_state=20),AdaBoostClassifier(random_state=20),
                   svm.SVC(kernel='linear', C=1, random_state=20),RandomForestClassifier(random_state=20),
                   GaussianNB(),KNeighborsClassifier(),SGDClassifier(random_state=20),
                   LogisticRegression(random_state=20)]
    '''
    clfName = clf.__class__.__name__
    if optimization == 0:
        print("\nClassificador: {}".format(clfName))
        try:
            clf = clf.fit(X_train, y_train)
        except:
            y_train = np.ravel(y_train)
            clf = clf.fit(X_train, y_train)

        test_predictions = clf.predict(X_test)
        cnf_matrix = confusion_matrix(y_test, test_predictions)

        fscore = fbeta_score(y_test, test_predictions, beta=0.5, average='macro')
        return(fscore,cnf_matrix,clf)

    else:
        results = {}
        scorer = make_scorer(fbeta_score, beta=0.5, average='macro')
        grid_obj = GridSearchCV(clf, parameters, scorer)
        y_train = np.ravel(y_train)
        grid_fit = grid_obj.fit(X_train, y_train)
        results['best_clf'] = grid_fit.best_estimator_
        results['predictions'] = clf.fit(X_train, y_train).predict(X_test)
        results['best_predictions'] = results['best_clf'].predict(X_test)
        results['best_score'] = grid_fit.best_score_
        results['best_params'] = grid_fit.best_params_
        results['cnf_matrix'] = confusion_matrix(y_test, results['best_predictions'])
        results['test_score'] = fbeta_score(y_test, results['predictions'], beta=0.5, average='macro')
        results['final_test_score'] = fbeta_score(y_test, results['best_predictions'], beta=0.5, average='macro')
        return results


def findVout(traces):
    voutIndex = None
    for i, trace in enumerate(range(0, len(traces))):
        if traces[trace].name == 'V(bpo)' or traces[trace].name == 'V(vout)':
            voutIndex = i

    return voutIndex


def confusionMatrixPlot(cnf_matrix,circuito,clfName,dataSize):
    cm = 100 * cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    title = '{}_CM_{}'.format(circuito, clfName)
    cmap = plt.cm.Blues

    fig = plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    classNames = list(range(1, dataSize // 300 + 1))
    plt.title(title)
    plt.colorbar()
    plt.ylabel('Real')
    plt.xlabel('Predito')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.savefig("{}{}.png".format(IMG_DIR,title), bbox_inches='tight')

    f = lambda x: round(x, 2)
    cm = pd.DataFrame(cm).apply(f)

    return cm