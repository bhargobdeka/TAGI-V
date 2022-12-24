%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         toyExample with one network
% Description:  Apply TAGI to heteroscedasticity
% Authors:      Luong-Ha Nguyen & James-A. Goulet & Bhargob Deka
% Created:      Dec 9, 2020
% Updated:      Dec 9, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca &
% bhargob.deka@mail.polymtl.ca
% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet & Bhargob Deka
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
% clc
% close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
rand_seed=4;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
%% Data
rng(1223) % Seed
ntrain     = 40;
ntest      = 100;
f          = @(x) (5*x).^3/50;
noise      = @(x) (3*(x).^4)+0.02;      %(3*(x).^4)+0.02, (3*(x).^4)+0.01, (0.1*(x+0.5).^3)+0.1,  (0.4*(x).^4)+0.01, 0.45.*(x+0.5).^2+0.02
% xtrain     = linspace(-1,1,ntrain)';
xtrain     = unifrnd(-1,1,[ntrain,1]);
% xtrain     = [rand(ntrain, 1)*1 - 0.5]; % Generate covariate between [-1, 1];
% wtrain     = normrnd(zeros(length(xtrain), 1), 0.1);
% xtest      = linspace(-1,1,ntest)';
% xtest      = unifrnd(-1,1,[ntest,1]);
% xtest      = [rand(ntest, 1)*1 - 0.5];
noiseTrain = noise(xtrain);
ytrainTrue = f(xtrain);
ytrain     = f(xtrain) + normrnd(zeros(length(noiseTrain), 1), sqrt(noiseTrain));
% wtest      = normrnd(zeros(length(xtest), 1), 0.1);
% noiseTest  = noise(xtest);
% ytestTrue  = f(xtest);
% ytest      = ytestTrue + normrnd(zeros(length(noiseTest), 1), sqrt(noiseTest));
nx         = size(xtrain, 2);
ny         = size(ytrain, 2);
figure;
scatter(xtrain,ytrain,'ob');
xlabel('x');
ylabel('y');
figure;
scatter(xtrain,noiseTrain,'ok');
xlabel('x');
ylabel('variance');
% % Normalize Data
% % [xtrain, ytrain, ~, ~, ~, ~, mytrain, sytrain] = dp.normalize(xtrain, ytrain, xtest, ytest);
% scatter(xtrain,ytrain,'dk');hold off
% figure;
% subplot(1,3,1)
% scatter(xtest,noiseTest,'ok');
% xlabel('x');
% ylabel('y');
% subplot(1,3,2)
% scatter(xtrain,noiseTrain,'ok');
% xlabel('x');
% ylabel('y');
% subplot(1,3,3)
% scatter(xtrain,ytrain,'dr');
% xlabel('x');
% ylabel('y');
% Data Loading
% load('Data_Hetero_BNI.mat')
% ntrain     = 400;
% ntest      = 100;
% nx         = size(xtrain, 2);
% ny         = size(ytrain, 2);

%% Net
% GPU 1: yes; 0: no
net.task           = 'regression';
net.saveModel      = 1;
% GPU
net.gpu            = 0;
net.numDevices     = 0;
% Data type object half or double precision
net.dtype          = 'single';
% Number of input covariates
net.nx             = nx; 
% Number of output responses
net.nl             = ny;
net.nv2            = ny;
net.ny             = 2*ny;             % 2*ny for 'hete'
% Batch size 
net.batchSize      = 1; 
net.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = [0 0 0];
net.layer          = [1         1       1       ];
net.nodes          = [net.nx    50   net.ny     ]; 
net.actFunIdx      = [0         4       0       ];
% Observations standard deviation
net.learnSv        = 1;% Online noise learning
net.sv             = 0;%0.32*ones(1,1, net.dtype);                         %BD
net.noiseType      = 'hete';      %'hom' or 'hete'                         %BD
net.NoiseActFunIdx = 2;                                                    %BD
net.NoiseActFun_LL_Idx = 4;
net.epoch          = 1;
net.v2hat_LL       = [0.2 0.2];
net.idx_LL_M       = [];
net.var_mode       = 0;
% % Hierarchical Prior for variance
% mv2hat_W_fact       = 1e-03*ones(1,length(net.nodes)-1);
% sv2hat_W_fact       = 1e-03*ones(1,length(net.nodes)-1);
% v2hat_W_fact        = [mv2hat_W_fact; sv2hat_W_fact];
% mv2hat_B_fact       = 1e-04*ones(1,length(net.nodes)-1);
% sv2hat_B_fact       = 1e-04*ones(1,length(net.nodes)-1);
% v2hat_B_fact        = [mv2hat_B_fact; sv2hat_B_fact];
% [net.v2hat_W, net.v2hat_B]  = createHPWeights(net, v2hat_W_fact, v2hat_B_fact);
net.HP = 1;
% Parameter initialization
net.initParamType  = 'He';
net.gainS          = 0.5*ones(1, length(net.layer)-1);
net.gainS_v2hat    = 0.5;     %0.3 best for xavier and 0.15 best for He
net.gainSb_v2hat   = 1;
net.gainM_v2hat    = 1;
% Maximal number of epochs and splits
net.maxEpoch       = 50;   
net.numSplits      = 1;
% Cross-validation for v2hat_LL
net.cv_v2hatLL  = 0;
net.numFolds    = 1;
net.permuteData = 1;    % 1 for split, else for fold
net.ratio       = 0.8;
% creating mesh
load('EEW2g.mat');
load('SEW2g.mat');
load('aOg.mat');
net.EEW2g = EEW2g;
net.SEW2g = SEW2g;
net.aOg   = aOg;

%% Run
% Initialization          
saveModel  = net.saveModel;
maxEpoch   = net.maxEpoch;
svinit     = net.sv;

% Train net
net.trainMode = 1;
[net, states, maxIdx, netInfo] = network.initialization(net);

% Test net
netT              = net;
netT.trainMode    = 0;
netT.batchSize    = 1;
netT.repBatchSize = 1;
normStatT         = tagi.createInitNormStat(netT); 
[netT, statesT, maxIdxT] = network.initialization(netT); 


% Initalize weights and bias
if net.HP == 1
    theta = tagi.initializeWeights_HP_MeanOnly(net);
else
    theta = tagi.initializeWeightBias(net);
end
normStat = tagi.createInitNormStat(net);
net.sv   = svinit;

% metrics
RMSE        = zeros(net.maxEpoch,1);
LLlist      = zeros(net.maxEpoch,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
rumtime_s = tic;
stop  = 0;
epoch = 0;
%Open the video writer object.
% v = VideoWriter('TY1_T4_He.avi');
% v.FrameRate = 1;   % showing one frame per sec
% open(v);
while ~stop
    epoch = epoch + 1;
    net.epoch   = epoch;      %BD
    net.epochCV = epoch;
%     [theta, normStat,zl,Szl,sv, Prior_act_v2hat, Pos_act_v2hat, K_Gain] = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
    [theta, normStat] = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
% Testing   
    [~, ~, zl, Szl,~, Prior_act_v2hat] = network.regression(netT, theta, normStatT, statesT, maxIdxT, xtest, []);
% Denormalizing
mv2a = Prior_act_v2hat(:,1);
ml   = zl(:,1);
Sl   = Szl(:,1);

% [ml, Sl] = dp.denormalize(ml, Sl, mytrain, sytrain);
%h = figure;
subplot(1,2,1)
Pr_msv = Prior_act_v2hat(:,1);
Pr_psv = sqrt(Prior_act_v2hat(:,2));
plot(xtest,Prior_act_v2hat(:,1),'k');hold on;  %plot the line first
patch([xtest' fliplr(xtest')],[Pr_msv'+Pr_psv',fliplr(Pr_msv'-Pr_psv')],'g','EdgeColor','none','FaceColor','g','FaceAlpha',0.2);hold on;
scatter(xtest,noiseTest,'dr');hold on;
xlabel('x');
ylabel('variance')
title(['Epoch',num2str(epoch)''])
hold off
subplot(1,2,2)
Pzl = sqrt(Sl);
scatter(xtest,ml,'k');hold on;
patch([xtest' fliplr(xtest')],[ml'+Pzl',fliplr(ml'-Pzl')],'g','EdgeColor','none','FaceColor','g','FaceAlpha',0.2);hold on;
scatter(xtest,ytest,'db');hold on;
scatter(xtest,ytestTrue,'r');
xlabel('x');
ylabel('mean')
title(['Epoch',num2str(epoch)''])
drawnow
% F = getframe(h);
% writeVideo(v,F.cdata)
% hold off
% pause(0.01)
% Evaluation
RMSE(epoch)          = mt.computeError(ytest, ml);
LLlist(epoch)        = mt.loglik(ytest, ml, Sl);

disp(' ')
disp(['      RMSE : ' num2str(RMSE(epoch)) ])
disp(['      LL   : ' num2str(LLlist(epoch))])


    if epoch >= maxEpoch; break;end
end
% close(v)
close all
runtime_e = toc(rumtime_s);
figure;
scatter(1:net.maxEpoch,LLlist,'ob');hold on;scatter(1:net.maxEpoch,RMSE,'dk')
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing
xtest      = linspace(-1,1,ntest)';
noiseTest  = noise(xtest);
ytestTrue  = f(xtest);
ytest      = ytestTrue + normrnd(zeros(length(noiseTest), 1), sqrt(noiseTest));
nObsTest   = size(xtest, 1);

[~, ~, yp, Syp] = network.regression(netT, theta, normStatT, statesT, maxIdxT, xtest, []);
if net.learnSv==1&&strcmp(net.noiseType, 'hete') % Online noise inference
    ypM  = reshape(yp', [net.nl, 1, 2, nObsTest]);
    SypM = reshape(Syp', [net.nl, 1, 2, nObsTest]);
    
    mv2  = reshape(ypM(:,:,2,:), [net.nl*nObsTest, 1]);
    Sv2  = reshape(SypM(:,:,2,:), [net.nl*nObsTest, 1]);
    mv2a = act.NoiseActFun(mv2, Sv2, net.NoiseActFunIdx, net.gpu);
    
    ml   = reshape(ypM(:,:,1,:), [net.nl*nObsTest, 1]);
    Sl   = reshape(SypM(:,:,1,:), [net.nl*nObsTest, 1]) + mv2a;
else
    ml = yp;
    if strcmp(net.noiseType, 'homo')
        Sl = Syp+net.sv(1);
    else
        Sl = Syp+net.sv(1).^2;
    end
end

%%  Plotting Data
figure;
plot(xtest,ytestTrue,'r');hold on
patch([xtest' fliplr(xtest')],[ytestTrue' + sqrt(noiseTest') fliplr(ytestTrue' - sqrt(noiseTest'))],'red','EdgeColor','none','FaceColor','red','FaceAlpha',0.2);hold on
scatter(xtrain,ytrain,'+b');hold on
scatter(xtest,ytest,'dm');hold on
plot(xtest,ml,'k');hold on
patch([xtest' fliplr(xtest')],[ml' + sqrt(Sl') fliplr(ml' - sqrt(Sl'))],'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.2)
h=legend('$y_{true}$','$y_{true} \pm \sigma_{true}$','train','test','$E[\hat{y}]$','$E[\hat{y}] \pm 1\sigma$');
set(h,'Interpreter','latex')
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')



set(gcf,'Color',[1 1 1])
opts=['scaled y ticks = false,',...
    'scaled x ticks = false,',...
    'x label style={font=\large},',...
    'y label style={font=\large},',...
    'z label style={font=\large},',...
    'legend style={font=\large},',...
    'title style={font=\large},',...
    'mark size=5',...
    ];
%matlab2tikz('figurehandle',gcf,'filename',[ 'regression_1D_early_stop.tex'] ,'standalone', true,'showInfo', false,'floatFormat','%.5g','extraTikzpictureOptions','font=\large','extraaxisoptions',opts);


