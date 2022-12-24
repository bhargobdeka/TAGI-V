%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         toyExample with two networks
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
% rand_seed=4;
% RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
%% Data
rng(1223) % Seed
ntrain     = 400;
ntest      = 100;
f          =@(x) 0;
noise      =@(x) (3*(x).^4)+0.02;          %0.45*(x+0.5).^2
xtrain     = [rand(ntrain, 1)*1 - 0.5]; % Generate covariate between [-1, 1];
wtrain     = normrnd(zeros(length(xtrain), 1), 0.1);
xtest      = linspace(-1, 1, ntest)';
noiseTrain = noise(xtrain);
ytrainTrue = f(xtrain);
ytrain     = f(xtrain) + normrnd(zeros(length(noiseTrain), 1), sqrt(noiseTrain));
wtest      = normrnd(zeros(length(xtest), 1), 0.1);
noiseTest  = noise(xtest);
ytestTrue  = f(xtest);
ytest      = ytestTrue + normrnd(zeros(length(noiseTest), 1), sqrt(noiseTest));
nx         = size(xtrain, 2);
ny         = size(ytrain, 2);

figure;
subplot(1,2,1)
scatter(xtrain, noiseTrain)
subplot(1,2,2)
scatter(xtrain,ytrain)
%% Noise Net
% GPU 1: yes; 0: no
net.task            = 'regression';
net.saveModel       = 1;
% GPU
net.gpu             = 0;
net.numDevices      = 0;
% Data type object half or double precision
net.dtype           = 'single';
% Number of input covariates
net.nx              = 1; 
% Number of output responses
net.nl              = ny;
net.nv2             = ny;
net.ny              = ny; 
% Batch size 
net.batchSize      = 1; 
net.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = [0 0 0];
net.layer          = [1              1        1        ];
net.nodes          = [net.nx        200      net.ny    ]; 
net.actFunIdx      = [0              4        0        ];
net.NoiseActFunIdx = 3;
% Observations standard deviation
net.learnSv         = 0;% Online noise learning
net.sv              = 0;%0.32*ones(1,1, netN.dtype);  
net.lastLayerUpdate = 0;
net.noiseType       = 'hete';      %'hete' 
% Parameter initialization
net.initParamType  = 'He';      %'Xavier', 'He'
net.gainS          = 2*ones(1, length(net.layer)-1);     % Y = X:  G = 0.05, Epoch = 75 , other: 1.75 for original func  
% Maximal number of epochs and splits
net.maxEpoch       = 100;   
net.numSplits      = 1;

%% Run
% Initialization          
saveModel  = net.saveModel;
maxEpoch   = net.maxEpoch;
svinit     = net.sv;

% Train net
net.trainMode = 1;
[net, states, maxIdx, netinfo] = network.initialization(net);

% Test net
net_test              = net;
net_test.trainMode    = 0;
net_test.batchSize    = 1;
net_test.repBatchSize = 1;
[net_test, statesTest, maxIdx_statesTest] = network.initialization(net_test);
normStat_test = tagi.createInitNormStat(net_test);

% Initalize weights and bias
theta     = tagi.initializeWeightBias(net);
normStat  = tagi.createInitNormStat(net);
net.sv    = svinit;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
rumtime_s = tic;
stop  = 0;
epoch = 0;
while ~stop
    epoch = epoch + 1;
    [theta, normStat,~,~,Prior_act_v2hat, Pos_act_v2hat] = network.regressionOnlyNI(net, theta, normStat, states, maxIdx, xtrain, ytrain);
    % Figure
    subplot(1,2,1)
    xpr=(1:length(xtrain))';
    Pos_msv = Pos_act_v2hat(:,1);
    Pos_psv = sqrt(Pos_act_v2hat(:,1));
    %patch([xtrain' fliplr(xtrain')],[Pos_msv'+Pos_psv',fliplr(Pos_msv'-Pos_psv')],'g','EdgeColor','none','FaceColor','g','FaceAlpha',0.2);hold on
    scatter(xtrain,Pos_act_v2hat(:,1),'ok');hold on
    scatter(xtrain,noiseTrain,'dr');
    xlabel('x');
    ylabel('var (Pos)')
    title(['Epoch',num2str(epoch)''])
    hold off
    subplot(1,2,2)
    Pr_msv = Prior_act_v2hat(:,1);
    Pr_psv = sqrt(Prior_act_v2hat(:,1));
    %patch([xpr' fliplr(xpr')],[Pr_msv'+Pr_psv',fliplr(Pr_msv'-Pr_psv')],'g','EdgeColor','none','FaceColor','g','FaceAlpha',0.2);hold on
    scatter(xtrain,Prior_act_v2hat(:,1),'ok');hold on
    scatter(xtrain,noiseTrain,'dr');
    xlabel('x');
    ylabel('var (Prior)')
    title(['Epoch',num2str(epoch)''])
    drawnow
    hold off
    if epoch >= maxEpoch; break;end
end
runtime_e = toc(rumtime_s);