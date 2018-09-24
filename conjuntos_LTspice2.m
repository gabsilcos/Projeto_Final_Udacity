% Extra��o de dados do CUT simulado no LTspice
% 
% ENTRADA:           raw_data = nome do arquivo struct vindo da fun��o LTspice2Matlab
%                    runs = n�mero de circuitos
%                    circ_classe = circuitos por classe
%                    var = numero de vari�veis a serem extra�das
%                    num_classes = n�mero de classes de falhas do circuito
%                    n = n�mero de segmentos do pre-processamento
%                    preproc = tipo de preprocessamento ('PAA' ou 'APCA')
%                    input = tipo de excita��o do circuito ('step' ou
%                    'prbs')
%                    norm = normaliza ou n�o o conjunto
% 
% SA�DA:             Teste_pr = conjunto de dados prdataset para Teste
%                    Treino_pr = conjunto de dados prdataset para Treino
%                    Teste = Sa�da ap�s PAA + normaliza��o
%
% EXEMPLO: [Teste_pr_300,Treino_pr_300,Teste,Treino,Teste_sinal,Treino_sinal]...
% = Teste_Treino_PAA('Sallen Key mc + step [FALHA] - 100 - 350us',3,[1 4 3],6600,300,11,32,'PAA')

tic;

if nargin < 7
    
    error('Vari�veis insuficientes para executar a fun��o')
    
elseif not(strcmp(preproc,'PAA') || strcmp(preproc,'APCA'))
    
    error('M�todo de preprocessamento n�o especificado corretamente')
    
elseif not(strcmp(input,'step') || strcmp(input,'prbs') || strcmp(input,'PRBS'))
    
    error('Tipo de entrada n�o especificado corretamente')
    
end

if var == 3 %extra��o de Vout (um ou tr�s pontos do circuito)
    Vout =  raw_data.variable_mat;
elseif var == 2
    Vout = raw_data.variable_mat(1:2,:);
elseif var == 1
    Vout = raw_data.variable_mat(3,:); %extrai apenas a saida do circuito
    
else
    error('Especificar n�mero correto de vari�veis a serem extra�das')
end

time=raw_data.time_vect; %extra��o do tempo
if ~(nargin<9)
    if strcmp(preproc,'APCA'),temp = 2*n; else temp = n;end,
    VOUT = zeros(runs,temp*var); 
end
serie_original = 512;
conjunto_original = zeros(runs,serie_original*var);
jj = 1;kk = 1; xi = linspace(0,time(end),serie_original);
if nargin<9, temp = 0; end
%% ------- Extra��o do Spice, diferencia��o e pre-processamento PAA -------

endtime = find(time == 0);endtime(1) = [];
endtime = endtime(1:2:end);
endtime = endtime - 1; endtime = [endtime length(time)];

for ii = 1:var
    j=1;m=1;
    for i = 1:length(endtime)
        if strcmp(input,'step')
            [~,~,tempv,~,~] = dif_pp(time(j:endtime(i)),Vout(ii,j:endtime(i)),serie_original);
        elseif or(strcmp(input,'prbs'),strcmp(input,'PRBS'))
            pp = spline(time(j:endtime(i)),Vout(ii,j:endtime(i)));
            tempv = ppval(pp,xi);  
        end
        if ~(nargin<9)
        if strcmp(preproc,'PAA') 
            VOUT(m,jj:ii*n) = paa(tempv,serie_original,n);
        elseif strcmp(preproc,'APCA') 
            I=apcav3(tempv,n); %I(:,2) = xi(I(:,2));
            VOUT(m,jj:ii*temp) = [I(:,1)' I(:,2)']; % reshape(I',1,n*2); % [I(:,1)' I(:,2)']; 
        end
        end
        conjunto_original(m,kk:ii*serie_original) = tempv;
        j=endtime(i)+2; m=m+1;
    end
    kk = ii*serie_original+1;
    jj = ii*temp+1; 
end

% Normaliza��o dos dados
if strcmp(norm,'norm')
    if ~(nargin<9),conjunto_preproc = zscore(VOUT')';end
    conjunto_original = zscore(conjunto_original')';
else
    if ~(nargin<9),conjunto_preproc = VOUT;end
end

% if isempty(norm)
%     [conjunto_preproc,norm{1,1},norm{1,2}]=zscore(VOUT);
%     conjunto_preproc = conjunto_preproc';
%     [conjunto_original,norm{2,1},norm{2,2}]=zscore(conjunto_original);    
%     conjunto_original = conjunto_original';
% else
%     sigma0 = norm{1,2}; sigma0(sigma0==0) = 1;
%     z = bsxfun(@minus,VOUT, norm{1,1});
%     conjunto_preproc = bsxfun(@rdivide, z, sigma0);
%     conjunto_preproc = conjunto_preproc';
%     
%    sigma0 = norm{2,2}; sigma0(sigma0==0) = 1;
%     z = bsxfun(@minus,conjunto_original, norm{2,1});
%    conjunto_original = bsxfun(@rdivide, z, sigma0);
%     conjunto_original = conjunto_original';
% end

%% ------------- IMPLEMENTAR CRIA��O DO PRDATASET ------------------------

num_labels = zeros(1,num_classes);
num_labels(num_labels==0) = circ_classe ;
label=1:num_classes;
if ~(nargin < 9)
    conjunto_dataset = prdataset(conjunto_preproc,genlab(num_labels,label'));
else
    conjunto_dataset = prdataset(conjunto_original,genlab(num_labels,label'));
    conjunto_preproc = [];
end
toc;
