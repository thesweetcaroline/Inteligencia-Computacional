clc; clear all;
%Conjunto de treino
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
figure;
colormap(gray) % set to grayscale
for i = 1:25 % preview first 25 samples
 subplot(5,5,i) % plot them in 5 x 5 grid
 digit = reshape(images(:,i),[28,28]); %row=28x28 image
 imagesc(digit) % show the image
end
%disp(labels(1:25))

conjuntoDeTreino_x = images;
conjuntoDeTreino_y_temp = labels;
for sample = 1 : length(conjuntoDeTreino_y_temp)
   conjuntoDeTreino_y(:, sample) = (conjuntoDeTreino_y_temp(sample)==0:9); 
end
size(conjuntoDeTreino_x)
size(conjuntoDeTreino_y)


%% conjunto de teste
imagesTeste = loadMNISTImages('t10k-images-idx3-ubyte');
labelsTeste = loadMNISTLabels('t10k-labels-idx1-ubyte');
figure;
colormap(gray) % set to grayscale
for i = 1:25 % preview first 25 samples
 subplot(5,5,i) % plot them in 5 x 5 grid
 digit = reshape(images(:,i),[28,28]); %row=28x28 image
 imagesc(digit) % show the image
end
%disp(labels(1:25))

conjuntoDeTeste_x = imagesTeste;
conjuntoDeTeste_y_temp = labelsTeste;
for sample = 1 : length(conjuntoDeTeste_y_temp)
   conjuntoDeTeste_y(:, sample) = (conjuntoDeTeste_y_temp(sample)==0:9); 
end
size(conjuntoDeTeste_x)
size(conjuntoDeTeste_y)

%Treinar a rede a avaliar desempenho
%numero_de_neuronios = 250;
coeficiente_aprendizagem = 0.01;
drawOn = 1;

% Métricas e desempenho
%Inicializar a rede
net = feedforwardnet(numero_de_neuronios);
%net = patternnet([300 300 500]);

%Personalizar algoritmo de treino
net.trainFcn = 'traingdx'; %Gradiant Descent
%net.trainFcn = 'traingdx'; %Gradiant Descent
net.trainParam.lr = coeficiente_aprendizagem;  %Learning rate


%Treinar a rede
net = train(net,conjuntoDeTreino_x,conjuntoDeTreino_y);

%Calcular as saidas dadas pela rede no conjunto de teste
saidaDaRedeParaConjuntoDeTeste = net(conjuntoDeTeste_x);

%Matriz de confusão
figure;
plotconfusion(conjuntoDeTeste_y,saidaDaRedeParaConjuntoDeTeste);
%net = Exercicio3(numero_de_neuronios,coeficiente_aprendizagem,conjuntoDeTreino_x,conjuntoDeTreino_y,conjuntoDeTeste_x,conjuntoDeTeste_y,drawOn);

[c,cm,ind,per] = confusion(conjuntoDeTeste_y,saidaDaRedeParaConjuntoDeTeste);
TotalP=0;
TotalA=0;
TotalR=0;
TotalE=0;
for j=1 : 10
    FN=0;
    FP=0;
    for i=1 : 10
        if(j~=i)
            FP= FP+(cm(j,i));
            FN=FN+(cm(i,j));
        end
    end
    TP=cm(j,j);
    TN=sum(sum(cm(:,:)))-(TP+FP+FN);
    C = sprintf('Resultados para a classe %d',j);
      disp(C)
      Precision = TP/(TP+FP);
      X = sprintf('%f Precision\n',Precision);
      disp(X);
    TotalP = TotalP + Precision;
      Recall = TP/ (TP+FN);
       X = sprintf('%f Recall\n',Recall);
      disp(X);
    TotalR = TotalR + Recall;
      Accuracy = (TP+TN)/(TP+TN+FN+FP);
        X = sprintf('%f Accuracy\n',Accuracy);
        disp(X);
    TotalA = TotalA + Accuracy;
    Espec = TN/(FP+TN);
          X = sprintf('%f Especificidade\n',Espec);
        disp(X);
    TotalE = TotalE + Espec;

    TPR= TP/(TP+FN);
    FPR = FP/(FP+TN);
    A = [0;TPR;1];
    B = [0;FPR;1];
    AUC = trapz(B,A);
    X = sprintf('%f Valor AUC',AUC);
    disp(X)
end

disp('-----Media -----');
fprintf('%f Precision Media\n',TotalP);
fprintf('%f Recall Media\n',TotalR);
fprintf('%f Accuracy Media\n',TotalA);
fprintf('%f Especificidade Media\n',TotalE);


 disp('-----Fmesure-----\n');
Fmesure = ((TotalP*TotalR/(TotalP + TotalR))*2)/10;
fprintf('%f\n',Fmesure);

