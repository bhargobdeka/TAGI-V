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
% rng(1223) % Seed
ntrain     = 400;
ntest      = 100;
f          = @(x) (5*x).^3/50;
% xtrain     = (rand(ntrain, 1)*8 - 4)/5; % Generate covariate between [-1, 1];
xtrain     = [rand(ntrain, 1)*1 - 0.5];
sv         = 0;
xtest      = linspace(-1, 1, ntest)';
ytrainTrue = f(xtrain);
ytrain     = ytrainTrue + normrnd(0, sv,[ntrain, 1]);

%test set
ytestTrue  = f(xtest);
ytest      = ytestTrue;
nx         = size(xtrain, 2);
ny         = size(ytrain, 2);

% scatter(xtrain,noiseTrain,'ok');
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
netD.task           = 'regression';
netD.saveModel      = 1;
% GPU
netD.gpu            = 0;
netD.numDevices     = 0;
% Data type object half or double precision
netD.dtype          = 'single';
% Number of input covariates
netD.nx             = nx; 
% Number of output responses
netD.nl             = ny;
netD.nv2            = ny;
netD.ny             = ny;             % 2 for 'hete'
% Batch size 
netD.batchSize      = 1; 
netD.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
netD.imgSize        = [0 0 0];
netD.layer          = [1          1       1       1      ];
netD.nodes          = [netD.nx   125    125     netD.ny  ]; 
netD.actFunIdx      = [0          4       4       0      ];
% Observations standard deviation
netD.learnSv        = 1;% Online noise learning
netD.sv             = [0.1   0.01];%0.32*ones(1,1, net.dtype);                   %BD
netD.noiseType      = 'homo';      %'homo' or 'hete'                             %BD
netD.NoiseActFunIdx = 2;
% Parameter initialization
netD.initParamType  = 'He';  %'Xavier', 'He'
netD.gainS          = 0.25*ones(1, length(netD.layer)-1);
% Maximal number of epochs and splits
netD.maxEpoch       = 25;   
netD.numSplits      = 1;


%% Run
% Initialization          
saveModel  = netD.saveModel;
maxEpoch   = netD.maxEpoch;
svinit     = netD.sv;

% Train net
netD.trainMode = 1;
[netD, statesD, maxIdxD, netDinfo] = network.initialization(netD);

% Initalize weights and bias
thetaD     = tagi.initializeWeightBias(netD);
normStatD  = tagi.createInitNormStat(netD);
netD.sv    = svinit;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
rumtime_s = tic;
stop  = 0;
epoch = 0;
while ~stop
    epoch   = epoch + 1;
    netD.sv = svinit;   %BD
    [thetaD, normStatD,Yn,~,sv] = network.regression(netD, thetaD, normStatD, statesD, maxIdxD, xtrain, ytrain);
    %figure
%     scatter(xtrain,ytrainTrue,'r');hold on
%     scatter(xtrain,Yn,'ok');hold on
%     scatter(xtrain,ytrain,'dm');
%     xlabel('x');
%     ylabel('y')
%     title(['Epoch',num2str(epoch)''])
%     drawnow
%     hold off
    %netD.sv = sv;
    if epoch >= maxEpoch; break;end
end
runtime_e = toc(rumtime_s);

%% Data for Heteroscedastic Noise
noise      = @(x) (3*(x).^4)+0.02;                  %f1: (3*(x).^4)+0.02, 0.45*(x+0.5).^2, f2: 0.5.*(x+1)
noiseTrain = noise(xtrain);
ytrain     = f(xtrain) + normrnd(zeros(length(noiseTrain), 1), sqrt(noiseTrain));
% figure;
subplot(1,2,1)
scatter(xtrain, noiseTrain)
subplot(1,2,2)
scatter(xtrain,ytrain)
%% Noise Net
% GPU 1: yes; 0: no
netN.task           = 'regression';
netN.saveModel      = 1;
% GPU
netN.gpu            = 0;
netN.numDevices     = 0;
% Data type object half or double precision
netN.dtype          = 'single';
% Number of input covariates
netN.nx             = netD.nodes(1); 
% Number of output responses
netN.nl             = ny;
netN.nv2            = ny;
netN.ny             = ny; 
% Batch size 
netN.batchSize      = netD.batchSize; 
netN.repBatchSize   = netD.repBatchSize; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
netN.imgSize        = [0 0 0];
netN.layer          = [1          1       1      ];
netN.nodes          = [netN.nx   50    netN.ny  ]; 
netN.actFunIdx      = [0          4       0      ];
netN.NoiseActFunIdx = 3;
% Observations standard deviation
netN.learnSv         = 0;% Online noise learning
netN.sv              = 0;%0.32*ones(1,1, netN.dtype);  
netN.lastLayerUpdate = 0;
netN.noiseType       = 'hete';      %'hete' 
% Parameter initialization
netN.initParamType  = 'He';
netN.gainS          = 2*ones(1, length(netN.layer)-1);
% Maximal number of epochs and splits
netN.maxEpoch       = 100;   
netN.numSplits      = 1;
%layerConct          = 1;     %BD
layerConct = 1;
%Train net
netN.trainMode = 1;
[netN, statesN, maxIdxN, netNinfo] = network.initialization(netN);

netN1              = netN;
netN1.trainMode    = 0;
netN1.batchSize    = 1;
netN1.repBatchSize = 1;
[netN1, statesN1, maxIdxN1] = network.initialization(netN1);
normStatN1 = tagi.createInitNormStat(netN1);

% Initalize weights and bias
thetaN    = tagi.initializeWeightBias(netN);
normStatN = tagi.createInitNormStat(netN);

% Training for Noise    
runtime_s2 = tic;
stop   = 0;
epoch  = 0;
maxEpoch = netN.maxEpoch;
while ~stop
    epoch = epoch + 1;
    [thetaN, normStatN, ~, ~, Prior_act_v2hat, Pos_act_v2hat] = network.regressionWithNI(netD, thetaD, normStatD, statesD, maxIdxD, netN, thetaN, normStatN, statesN, maxIdxN, xtrain, ytrain, layerConct);
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
runtime_e2 = toc(runtime_s2);
runtime_e  = runtime_e + runtime_e2;