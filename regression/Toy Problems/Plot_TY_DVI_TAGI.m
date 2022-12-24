
%% Load Train Data
load('xtrain_TY1_TAGI_BNI.mat')
load('ytrain_TY1_TAGI_BNI.mat')

%% Load Pred
load('Pred_TAGI_BNI.mat');
load('Pred_TAGI.mat');
load('Pred_DNN.mat');

%clear;clc;
load('Pred_DVI.mat')

% load('Data_mean_DVI.mat')
% load('Data_std_DVI.mat')
% %% Data
% rng(1223) % Seed
% ntrain     = 500;
% ntest      = 100;
f          = @(x) -(x+0.5).*sin(3*pi*x);
noise      = @(x) 0.45*(x+0.5).^2;
% 
% xtrain     = [rand(ntrain, 1)*1 - 0.5]; % Generate covariate between [-1, 1];
% noiseTrain = noise(xtrain);
% ytrainTrue = f(xtrain);
% ytrain     = f(xtrain) + normrnd(zeros(length(noiseTrain), 1), noiseTrain);
xtest  = linspace(-1,1,200)';
x = sort(xtrain);
mean = f(x);
std  = noise(x);
% figure('Renderer', 'painters', 'Position', [10 10 900 600])

plot(x,f(x),'r');hold on
patch([x' fliplr(x')],[f(x') + sqrt(std') fliplr(mean' - sqrt(std'))],'red','EdgeColor','none','FaceColor','red','FaceAlpha',0.2);hold on
scatter(xtrain,ytrain,'dm');
plot(xtest,Pred_TAGI(:,1),'k');hold on
patch([xtest' fliplr(xtest')],[Pred_TAGI(:,1)' + sqrt(Pred_TAGI(:,2)') fliplr(Pred_TAGI(:,1)' - sqrt(Pred_TAGI(:,2)'))],'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.3);hold on
h=legend('$y_{true}$','$y_{true} \pm \sigma_{true}$','train','$E[\hat{y}]$','$E[\hat{y}] \pm 1\sigma$');
set(h,'Interpreter','latex')
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
title('TAGI')
xlim([-1,1]);
ylim([-2,2]);
%%
figure;
plot(x,f(x),'r');hold on
patch([x' fliplr(x')],[f(x') + sqrt(std') fliplr(mean' - sqrt(std'))],'red','EdgeColor','none','FaceColor','red','FaceAlpha',0.2);hold on
scatter(xtrain,ytrain,'dm');
plot(xtest,Pred_DNN(:,1),'k');hold on
patch([xtest' fliplr(xtest')],[Pred_DNN(:,1)' + sqrt(Pred_DNN(:,2)') fliplr(Pred_DNN(:,1)' - sqrt(Pred_DNN(:,2)'))],'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.3);hold on
h=legend('$y_{true}$','$y_{true} \pm \sigma_{true}$','train','$E[\hat{y}]$','$E[\hat{y}] \pm 1\sigma$');
set(h,'Interpreter','latex')
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
title('Neural Network')
xlim([-1,1]);
ylim([-2,2]);
%%
figure;
plot(x,f(x),'r');hold on
patch([x' fliplr(x')],[f(x') + sqrt(std') fliplr(mean' - sqrt(std'))],'red','EdgeColor','none','FaceColor','red','FaceAlpha',0.2);hold on
scatter(xtrain,ytrain,'dm');
plot(xtest,Pred_TAGI_BNI(:,1),'k');hold on
patch([xtest' fliplr(xtest')],[Pred_TAGI_BNI(:,1)' + sqrt(Pred_TAGI_BNI(:,2)') fliplr(Pred_TAGI_BNI(:,1)' - sqrt(Pred_TAGI_BNI(:,2)'))],'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.3);hold on
h=legend('$y_{true}$','$y_{true} \pm \sigma_{true}$','train','$E[\hat{y}]$','$E[\hat{y}] \pm 1\sigma$');
set(h,'Interpreter','latex')
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
title('TAGI-BNI')
xlim([-1,1]);
ylim([-2,2]);
%%
figure;
plot(x,f(x),'r');hold on
patch([x' fliplr(x')],[f(x') + sqrt(std') fliplr(mean' - sqrt(std'))],'red','EdgeColor','none','FaceColor','red','FaceAlpha',0.2);hold on
scatter(xtrain,ytrain,'dm');
plot(xtest,Pred_DVI(:,1),'k');hold on
patch([xtest' fliplr(xtest')],[Pred_DVI(:,1)' + sqrt(Pred_DVI(:,2)') fliplr(Pred_DVI(:,1)' - sqrt(Pred_DVI(:,2)'))],'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.3);hold on
h=legend('$y_{true}$','$y_{true} \pm \sigma_{true}$','train','$E[\hat{y}]$','$E[\hat{y}] \pm 1\sigma$');
set(h,'Interpreter','latex')
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
title('DVI')
xlim([-1,1]);
ylim([-2,2]);

set(gcf,'Color',[1 1 1])
opts=['scaled y ticks = false,',...
    'scaled x ticks = false,',...
    'x label style={font=\large},',...
    'y label style={font=\large},',...
    'z label style={font=\large},',...
    'legend style={font=\large},',...
    'title style={font=\Large},',...
    'mark size=2',...
    ];