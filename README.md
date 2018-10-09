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
