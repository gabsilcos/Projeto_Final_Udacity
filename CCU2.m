function [opt_param,opt_DR,true_labels,classes,classes_equivocadas,P,C,resultados] = CCU2(Treino,Teste,numlabels,gauss_param,knn_param,kmeans_param,parzen_param,circ_classe,type_DR,num_DR)

% Fun��o para construir a arquitetura do CCU das falhas no CUT
% CCUs usados --> kmeans_dd, knndd, gauss_dd, parzen_dd
%
% ENTRADA:          Treino: prdataset (para treino) com a resposta ao 
%                   impulso dos diversos circuitos do Sallen Key
%                   Teste: prdataset (para teste) com a resposta ao
%                   impulso dos diversos circuitos do Sallen Key
%                   numlabels: Numero de classes do circuito
%                   gauss_param: par�metros de ajuste do gauss_dd
%                   knn_param: par�metros de ajuste do knndd
%                   kmeans_param: par�metros de ajuste do kmeans_dd
%                   parzen_param: par�metro de ajuste do parzen_dd
%                   fracrej: fra��o de rejei��o
%
% SA�DA:            opt_param1: par�metros otimizados dos classificadores
%                   opt_param2: par�metros otimizados fracrej
%                   true_labels: rotulos verdadeiros pertinentes a cada
%                   classe
%                   classes: rotulos estimados pelos classificadores
%                   P: resultados de performance
%                   C: matiz confus�o
%                   resultados: Avalia��o de erros e acertos dos
%                   classificadores
%                   resultados: matriz com resultados da classifica��o
%                   (erros e acertos)
%


%% Gerando os valores de entrada
if nargin < 4
    error('N�o h� parametros de entrada suficiente')
end
    
param_adj = {gauss_param;knn_param;kmeans_param;parzen_param};
opt_DR = zeros(4,1);
opt_param = zeros(4,1);
resultados = cell(5,6);
resultados{1,2} = 'classif �nica';resultados{1,3} = 'classif grupo';
resultados{1,4} = 'erro'; resultados{1,5} = 'perdidos'; resultados{1,6} = 'desconhecidos';
%% Montando o classificador
for ii = 1:4
    
    [~,opt_DR(ii),opt_param(ii),opt_Treino,opt_Teste] = opt_class2(Treino,Teste,10,ii,param_adj{ii},circ_classe,numlabels,type_DR,num_DR);

        switch ii
            case 1 %gauss_dd
                disp '   '
                disp 'Matriz confus�o - Gauss_dd'
                disp '   '
                [t,clas,clas_eq,P_uni,C_uni] = moc(opt_Treino,gauss_dd([],0.01,opt_param(ii)),opt_Teste,numlabels);
                resultados{2,1} = 'Gauss_dd';

            case 2 %knndd
                disp '   '
                disp 'Matriz confus�o - Knndd'
                disp '   '
                [t,clas,clas_eq,P_uni,C_uni] = moc(opt_Treino,knndd([],0.01,opt_param(ii)),opt_Teste,numlabels);
                resultados{3,1} = 'Knndd';
                
            case 3 %kmeans_dd
                disp '   '
                disp 'Matriz confus�o - Kmeans_dd'
                disp '   '
                [t,clas,clas_eq,P_uni,C_uni] = moc(opt_Treino,kmeans_dd([],0.01,opt_param(ii)),opt_Teste,numlabels);   
                resultados{4,1} = 'Kmeans_dd';
                
            case 4 %parzen_dd
                disp '   '
                disp 'Matriz confus�o - Parzen_dd'
                disp '   '
                [t,clas,clas_eq,P_uni,C_uni] = moc(opt_Treino,parzen_dd([],0.01,opt_param(ii)),opt_Teste,numlabels);   
                resultados{5,1} = 'Parzen_dd';
                
        end
        C{ii} = C_uni;
        P{ii} = P_uni;
        classes{ii} = clas;
        classes_equivocadas{ii} = clas_eq;
        true_labels{ii} = t;
        [resultados{ii+1,2},resultados{ii+1,3},resultados{ii+1,4},resultados{ii+1,5}, resultados{ii+1,6}] = perf_class(clas,t,numlabels,clas_eq);
end
