% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet & Bhargob Deka
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clear 
% clc
% close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
rand_seed=1;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed)); 
%% Load Data
% path  = char([cd ,'/data/']);
% load(char([path, '/MotorCycle.mat']))
load('trainData.mat')
load('testData.mat')
xtrain = trainData(:,1);
ytrain = trainData(:,2);

xtest  = testData(:,1);
ytest  = testData(:,2);

nx     = size(xtrain, 2);
ny     = size(ytrain, 2);
%% Net
% GPU 1: yes; 0: no
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
net.layer          = [1         1      1          ];
net.nodes          = [net.nx   200     net.ny     ];
net.actFunIdx      = [0         4      0         ];
% Observations standard deviation
net.learnSv        = 1;% Online noise learning
net.sv             = 0;%0.32*ones(1,1, net.dtype);                         %BD
net.noiseType      = 'hete';      %'hom' or 'hete'                         %BD
net.NoiseActFunIdx = 1;                                                    %BD
net.NoiseActFun_LL_Idx = 4;
net.epoch          = 1;
net.v2hat_LL       = [0.2 0.2];
net.idx_LL_M       = [];
net.var_mode       = 0;

net.HP = 0;
% Parameter initialization
net.initParamType  = 'He';
net.gainS          = 1*ones(1, length(net.layer)-1);
net.gainS_v2hat    = 1;     %0.3 best for xavier and 0.15 best for He
net.gainSb_v2hat   = 1;
net.gainM_v2hat    = 1;
% Maximal number of epochs and splits
net.maxEpoch       = 500;
net.numSplits      = 1;
% Cross-validation for v2hat_LL
net.cv_v2hatLL  = 0;
net.numFolds    = 5;
net.permuteData = 1;    % 1 for split, else for fold
net.ratio       = 0.8;
% Cross-validation for HP
net.cv_HP       = 0;

% Grid Search for Gain factor
net.gs_Gain     = 1;

%% Hierarchical Prior for variance
net.HP      = 0;
net.HP_M    = 2;    % 1 for full , 2 for layerwise
net.xv_HP   = 0.005^2;
net.HP_BNI  = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid Search for Gain parameter

[Gain_factor, opt_E]      = opt.Gridsearch(net,xtrain,ytrain, []);
%run_time = toc(start);
% disp(['   Gain mean : ' num2str(run_time)])
Gains                     = [Gain_factor(1) Gain_factor(2)]; %0.5, 0.25
opt_Epochs                = opt_E; %82