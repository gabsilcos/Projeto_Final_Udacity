# Machine Learning Engineer Nanodegree
# Projeto Final
## Projeto: Detecção de Falhas em Circuitos Elétricos Analógicos

### Sobre o projeto:

  Este é um projeto que tem como finalidade prever falhas em circuitos elétricos analógicos tomando por base dados de simulação originados no software LTSpiceIV.<br>
  O LTSpiceIV gera um arquivo _.raw_ contendo diversos passos de simulação, e os valores de grandezas do circuito para cada passo de simulação. Estes passos seguem um comportamento predefinido, onde os componentes do circuito elétrico são forçados a assumir valores diferentes daqueles que caracterizariam o funcionamento normal do ciruito, de forma que para cada passo de simulação do LTSpiceIV pode-se determinar se o valor aferido da grandeza foi gerado em condição de falha, qual o componente defeituoso e qual o valor deste componente que iduziu à falha.<br>
  Assim, sabendo previamente como e quando a falha é gerada, os dados da simulação são extraídos do arquivo _.raw_ e submetidos aos métodos de aprendizagem supervisionada.<br>
  Por fim, este projeto observará a grandeza de saída dos circuitos, _V(out)_, aplicará vários algoritmos de aprendizagem supervisionada afim de determinar o desempenho destes algoritmos diante da finalidade proposta (prever as falhas no circuito), comparar o desempenho entre os diversos algoritmos, e, claro, prover uma forma de avaliação (principalmente auto avaliação) do meu domínio como aluno do curso sobre as ferramentas de aprendizaagem de máquina.<br><br>
  Não será necessário realizar uma nova simulação dos mesmos circuitos apresentados no projeto, visto que para estes foi gerado um arquivo ".csv" com os respectivos dataframes que serão necessários ao projeto.<br>
  O software LTSpiceIV pode ser baixado em: http://www.linear.com/designtools/software/demo_circuits.php<br><br>

### Dica:

  Afim de facilitar a navegação nesta página de readme os conteúdos dos tópicos foram dobrados sob os mesmos, sendo necessário, apenas, que você clique sobre a _seta_ antes do título do tópico para que o mesmo seja visualizado:
<details>
<summary> :point_left:  Clique na seta para expandir</summary>
<p>
      :ok_hand: :grin:
</p>
</details>

- - - -
### Pré-requisitos:
<details>
<summary>De instalação</summary>
<p>
  
Este projeto demanda uma instalção do **Python 3.6** e das seguintes bibliotecas:

- [matplotlib](http://matplotlib.org/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [tslearn](https://tslearn.readthedocs.io/en/latest/)


1. A instalção da biblioteca **matplotlib** é necessária pois será usada para plotar e salvar os resultados em gráficos;
2. A instalção da biblioteca **Numpy** é necessária para a manipulação de dados em arrays e algebrismos;
3. A instalção da biblioteca **Pandas** é necessária para a manipulção de dataframes e outras estruturas de dados;
4. A instalção da biblioteca **scikit-learn** é um toolkit para mineração e análise de dados, cujas ferramentas compõem a essência deste projeto onde foram usados:
    1. [Algoritmos de aprendizagem supervisionada:](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
       1. [svm](https://scikit-learn.org/stable/modules/svm.html)
       2. [ensemble](https://scikit-learn.org/stable/modules/ensemble.html)
       3. [tree](https://scikit-learn.org/stable/modules/tree.html)
       4. [naive bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
       5. [nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
       6. [generalized linear model](https://scikit-learn.org/stable/modules/linear_model.html)
       7. [Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/sgd.html)

    2. [Métricas de desempenho:](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)<br>
       1. [Métricas para classificadores:](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
             1. [fbeta_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score)
             2. [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)
       2. [Avaliação de modelos:](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer)<br>
             1. [make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer)
 
    3. Otimização de desempenho:<br>
       1. [Seleção de Modelo:](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)
             1. [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)
             2. [GridSearchCV (Otimização de hiperparâmetros)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)
 
5. A instalção da biblioteca **tslearn** é necessária pois o tslearn é um toolkit dedicado a dados de séries temporais, de onde foram utilizados o método de pré-processamento (TimeSeriesScalerMeanVariance, tslearn.preprocessing) e o método PAA (Piecewise Aggregate Approximation, tslearn.piecewise).
    1. [TimeSeriesScalerMeanVariance](https://tslearn.readthedocs.io/en/latest/gen_modules/preprocessing/tslearn.preprocessing.TimeSeriesScalerMeanVariance.html?highlight=TimeSeriesScalerMeanVariance)
    2. [Piecewise Aggregate Approximation](https://tslearn.readthedocs.io/en/latest/gen_modules/piecewise/tslearn.piecewise.PiecewiseAggregateApproximation.html?highlight=PAA)



Caso você ainda não tenha o Python instalado, recomendo que instale a distribuição [Anaconda](http://continuum.io/downloads), que já possui alguns destes pacotes instalados e facilita em muito a instalação de outros.
</p>
</details>

<details>
<summary>Pré instalados</summary><p>

Os pacotes a seguir são utilizados neste projeto, porém já se encontram presentes nas distribuições do Python:
1. [re](https://docs.python.org/3.6/library/re.html)
2. [os](https://docs.python.org/3.6/library/os.html?highlight=os#module-os)
3. [itertools](https://docs.python.org/3/library/itertools.html)
4. [time](https://docs.python.org/3.6/library/time.html?highlight=time#module-time)

</p>
</details>

- - - -

### Dados de entrada
<details>
<summary>Descrição e Links</summary><p>
  
Os arquivos com os dados das simulações podem vir a ser muito grandes por trazerem informações de cada grandeza a cada passo de simulação do circuito, o que torna inconcebível a hospedagem dos mesmos no github. Assim, deixo aqui um link (temporário) onde se encontram os dados de entrada deste projeto.
Os circuitos são:
* Sallen Key mc + 4bitPRBS [FALHA]
* Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s
* CTSV mc + 4bitPRBS [FALHA]
* Biquad Highpass Filter mc + 4bitPRBS [FALHA]

* Dados: https://drive.google.com/open?id=1KSEIhl6s4vkloGJGuobKPDel8952DXE4
</p>
</details>

- - - -


### Agradecimentos e Disclaimer

<details>
<summary></summary>
<p>
A realização deste projeto só foi possível devido ao trabalho do Engenheiro Sênior de Hardware Nuno Brum no desenvolvimento de ferramentas de tratamento de dados do LTSpice, especialmente do "LTSpiceRaw_Reader.py" que foi usado para a coleta dos dados de entrada.
Não possuo qualquer autoria sobre o LTSpiceRaw_Reader, e o PySpicer toolchain do qual é parte integrante se encontra no repositório original onde são fornecidas maiores informações:
* https://github.com/nunobrum/PyLTSpice
  
</p>
</details>

- - - -
