# Projeto final do curso de engenharia de Machine Learning pela Udacity/Kaggle

## Antes de mais nada
* O presente projeto implementa métodos de machine learning para a predição de falhas em circuitos elétricos analógicos, fazendo uso de dados de simulações. Dito isto faz-se necessário o uso de arquivos complementares, e até o download de software adicional, caso se deseje reproduzir o problema por completo.
Ressalto que não é necessário realizar uma nova simulação dos mesmos circuitos apresentados no projeto, visto que para estes foi gerado um arquivo ".csv" com os respectivos dataframes que serão necessários ao projeto.
* As simulações foram desenvolvidas com o uso do software LTSpiceIV, que pode ser baixado em: http://www.linear.com/designtools/software/demo_circuits.php

## Agradecimentos e Disclaimer
A realização deste projeto só foi possível devido ao trabalho do Engenheiro Sênior de Hardware Nuno Brum no desenvolvimento de ferramentas de tratamento de dados do LTSpice, especialmente do "LTSpiceRaw_Reader.py" que foi usado para a coleta dos dados de entrada.
Não possuo qualquer autoria sobre o LTSpiceRaw_Reader, e o PySpicer toolchain do qual é parte integrante se encontra no repositório original onde são fornecidas maiores informações:
* https://github.com/nunobrum/PyLTSpice

## Dados de entrada
Os arquivos com os dados das simulações pode ser muito grandes por trazerem informações de cada grandeza a cada passo de simulação do circuito, o que torna inconcebível a hospedagem dos mesmos no github. Assim, deixo aqui um link (temporário) onde se encontram os dados de entrada deste projeto.
Os circuitos são:
* Sallen Key mc + 4bitPRBS [FALHA]
* Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s
* CTSV mc + 4bitPRBS [FALHA]
* Biquad Highpass Filter mc + 4bitPRBS [FALHA]

* Dados: https://drive.google.com/open?id=1KSEIhl6s4vkloGJGuobKPDel8952DXE4

## Bibliotecas Necessárias

### re<br>
* Módulo de expressões regulares, usado para manipulação com nomes de arquivos e adaptações de títulos e legendas.<br>

### os<br>
* Utilizado para a amanipulação de caminhos de sistema, para indicação do local de origem dos arquivos e de destino das plotagens.<br>

### itertools<br>
* Ferramenta para auxiliar na iteração durante a plotagem da matriz de confusão.<br>

### time<br>
* Necessário para o controle de tempo de processos.<br>

### pandas<br>
* Necessário para a manipulção de dataframes e outras estruturas de dados.<br>

### pyplot do matplotlib<br>
* Necessário para a plotar e salvar os resultados em gráficos.<br>

### numpy<br>
* Necessário para a manipulação de dados em arrays e algebrismos.<br>

### svm, ensemble, tree, naive_bayes, neighbors, linear_model do sklearn<br>
* Necessários para a implementação dos métodos de classificação de dados.<br>

### train_test_split do sklearn.model_selection<br>
* Necessário para a criação dos grupos de treino e teste dos classificadores<br>

### fbeta_score, confusion_matrix, make_scorer do sklearn.metrics
* Métricas de desempenho<br>

### GridSearchCV do sklearn.grid_search<br>
* Necessário para a otimização dos classificadores.<br>

### TimeSeriesScalerMeanVariance do tslearn.preprocessing<br>PiecewiseAggregateApproximation do tslearn.piecewise<br>
* O tslearn é um toolkit dedicado a dados de séries temporais, de onde foi utilizado o método PAA (Piecewise Aggregate Approximation). Mais informações sobre o tslearn são encontradas em: https://tslearn.readthedocs.io/en/latest/
