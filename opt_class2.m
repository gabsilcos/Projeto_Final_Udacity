function [P_total,opt_DR, opt_param,opt_Treino,opt_Teste] = opt_class2(Treino,Teste,nfolds,type,param_adj,circ_classe,num_classes,type_DR,num_DR)
% 
% Função para encontrar os parametros que geram um modelo mais otimizado
% de acordo com a função oc_cross levando em consideração a redução de
% dimensão como parâmetro de ajuste
%
% ENTRADA:   Treino: conjunto de Treinamento
%            type: classificador de classe única não treinado. 1 para
%            gauss_dd, 2 para knndd, 3 para kmeans_dd, 4 para parzen_dd
%            nfolds: número de divisões para o oc_cross (tipicamente 10)
%            circ_classe = circuitos por classe
%            num_classes = número de classes de falhas do circuito
%            type_DR = tipo de DR a ser executada
%            num_DR: número de dimensões a se fazer redução de dimensão
%
% SAÍDA:     opt: vetor com os melhores valores avaliados pela função
%            oc_perf
%            opt_fracrej: vetor linha com valores correspondentes do primeiro
%            parâmetro para  opt
%            opt_param: vetor linha com valores correspondentes do segundo
%            parâmetro para opt
%
% EXEMPLO = [P_total,opt, opt_DR, opt_param] = opt_class(Treino,10,1,[0.01 0.02:0.02:0.28],'LPP',[2 3 4 5 6 7 8])
% 

if nargin < 7 || isempty(num_DR)
    num_DR = 1;
end

P_total = zeros(length(param_adj),length(num_DR));
% opt = zeros(1,7);
% opt_DR = zeros(1,7);
% opt_param = zeros(1,7);

for jj = 1:length(num_DR)
    
if ~(isempty(type_DR))
    [Treino2,~,~] = DR(Treino,Teste,type_DR,circ_classe,num_classes,[],num_DR(jj));
else
    Treino2 = Treino;
end
    for ii = 1:length(param_adj)
        switch type
            case 1 %gauss_dd
                P = oc_cross(Treino2,gauss_dd([],0.01,param_adj(ii)),nfolds,1); 
            case 2 %knndd
                P = oc_cross(Treino2,knndd([],0.01,param_adj(ii)),nfolds,1);                
            case 3 %kmeans_dd
                P = oc_cross(Treino2,kmeans_dd([],0.01,param_adj(ii)),nfolds,1);
            case 4 %parzen_dd
                P = oc_cross(Treino2,parzen_dd([],0.01,param_adj(ii)),nfolds,1);   
        end
        P_total(ii,jj) = mean(P(:,6)); 
    end
end

% for ii = 1:3
% [feat,ind] = min(P_total(:,ii,:));
% [feat,ind2] = min(feat);
% opt(ii) = feat;
% try opt_param(ii) = param_adj(ind(:,:,ind2));
% catch me 
%     opt_param(ii) = param_adj(end) ; 
% end
% opt_DR(ii) = num_DR(ind2);
% end
% for ii = 4:7
% [feat,ind] = max(P_total(:,ii,:));
% [feat,ind2] = max(feat);
% opt(ii) = feat;
% try opt_param(ii) = param_adj(ind(:,:,ind2));
% catch me 
%     opt_param(ii) = param_adj(end) ; 
% end
% opt_DR(ii) = num_DR(ind2);
% end

[~,ind] = max(P_total(:));
[row,column] = ind2sub([length(param_adj) length(num_DR)],ind);
opt_DR = num_DR(column);
opt_param = param_adj(row);

% Aplicação da DR otimizada preparando para a função CCU
if ~(isempty(type_DR))
[opt_Treino,opt_Teste,~] = DR(Treino,Teste,type_DR,circ_classe,num_classes,[],opt_DR);
else
    opt_Treino = Treino; opt_Teste = Teste;
end