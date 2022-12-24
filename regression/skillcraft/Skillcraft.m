%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         BostonHousing
% Description:  Apply TAGI to Boston Housing
% Authors:      Bhargob Deka, Luong-Ha Nguyen & James-A. Goulet
% Created:      July 21, 2020
% Updated:      December 02, 2020
% Contact:      bhargobdeka11@gmail.com & luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
% clc
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
% rng('default')
% rng(1223)
% rand_seed=2;
% RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
%% Data
load('skillcraft.mat')
load('loadGains.mat')
load('loadPatience.mat')
nobs  = size(data,1);
ncvr  = size(data,2)-1;
ratio = 0.8;
x     = data(:,1:ncvr);
y     = data(:,end); 
nx    = size(x, 2);        
ny    = size(y, 2);
%% Model
modelName        = 'FC';
dataName         = 'skillcraft';
%% Noise properties
net.noiseType = 'hete';   %'hete'
net.cv_Epoch_HomNoise = 0;
% Data type object half or double precision
net.dtype     = 'single';
if strcmp(net.noiseType,'homo')
    net.ny = ny;
    net.sv = [1  1];
elseif strcmp(net.noiseType,'hete')
    net.ny = 2*ny;
    net.sv = 0;
else
   net.ny = ny;
   net.sv = 0.32*ones(1,1, net.dtype);
end

%% Net
% GPU 1: yes; 0: no
net.task           = 'regression';
net.modelName      = modelName;
net.dataName       = dataName;
net.cd             = cd;
net.saveModel      = 1;
% GPU
net.gpu            = 1;
net.numDevices     = 0;
% Number of input covariates
net.nx             = nx; 
% Number of output responses
net.nl             = ny;
net.nv2            = ny;
% Batch size 
net.batchSize      = 10; %1
net.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = [0 0 0];
net.layer          = [1           1     1     1    1         ];
net.nodes          = [net.nx     1000  500    50   net.ny     ]; 
net.actFunIdx      = [0           4      4     4     0        ];
net.NoiseActFunIdx = 1;                                          %BD
net.NoiseActFun_LL_Idx = 4;
% Observations standard deviation
net.learnSv        = 1;% Online noise learning
net.epoch          = 1;
net.v2hat_LL       = [2 2];
net.idx_LL_M       = [];
net.var_mode       = 0;
% Parameter initialization
net.initParamType  = 'He';
% Gains              = [0.1  0.1];
net.Gains          = Gains;
% net.opt_Epochs     = opt_Epochs;
net.patience       = 3;

% Splits   
net.numSplits      = 10; 
% Cross-validation for v2hat_LL
net.cv_v2hatLL     = 0;
net.numFolds       = 1;
net.permuteData    = 1;    % 1 for split, else for fold
net.ratio          = 0.8;
% Cross-validation for HP
net.cv_HP          = 0;
%% Load Gain factors or HP
net.gs_Gain        = 1;    % 1 for grid-search, 0 for HP learning
if net.gs_Gain == 1
    net.gainS          = Gains(1)*ones(1,length(net.layer)-1);
    net.gainS_v2hat    = Gains(2);
    net.gainSb_v2hat   = 1;
    net.gainM_v2hat    = 1;
else
    net.gain_HP(1,:)   = [1e-02  1e-05];
    net.gain_HP(2,:)   = [Gains(2)  Gains(2)];
    net.gainSb_v2hat   = 1;
    net.gainM_v2hat    = 1;
    alpha = 0.01*(1-0.92); beta = 0.01*(1-0.92);
    net.m_w_v2hat      = [alpha*(1-1/net.nx)*Gains(1)*(1/net.nx)       1e-10       1e-10 ]; % 0.1*alpha*(1-1/50)*Gains(1)*(1/50) % alpha*Gains(2)*(1/50)
    net.m_b_v2hat      = [beta*(1/net.nx)                              1e-10        1e-10 ]; % beta*(1/50) % beta*(1/50)
    
    net.var_w_v2hat    = [(3*alpha*(1-1/net.nx)*Gains(1)*(1/net.nx))^2   1e-10       1e-10  ]; %(0.1*alpha*(1-1/50)*Gains(1)*(1/50))^2 %(alpha*Gains(2)*(1/50))^2
    net.var_b_v2hat    = [(3*beta*(1/net.nx))^2                          1e-10       1e-10  ]; %(beta*(1/50))^2 %(beta*(1/50))^2
end 
% Two layer properties
net.init = [];
%% Early Stopping
% net.epochlist = []; %BD
% filename1     = 'UCI_elevatorsEpochlist.txt';
% epochlist     = importdata(filename1);
% net.epochlist = epochlist;
net.epochlist = []; %BD
net.early_stop = 1;
if net.early_stop == 1
    net.maxEpoch       = 400;
    net.val_data       = 1;
else
    net.maxEpoch       = 40; %22
    net.val_data       = 0;
end
%% Hierarchical Prior for variance
net.HP   = 0;
net.HP_M = 2;    % 1 for full , 2 for layerwise
net.xv_HP   = 0.005^2;
net.HP_BNI  = [[0.5*ones(650,1);0.5*ones(50,1);2.7e-05*50*ones(50,1);ones(50,1);ones(2,1)] [(0.1/13)*ones(650,1);(0.1/50)*ones(50,1);1e-05*ones(50,1);(0.01/13)*ones(50,1);(1/50)*ones(2,1) ] ];  %[0.001^2*ones(650,1);0.001^2*ones(50,1);1e-05^2*ones(50,1);0.01^2*ones(50,1);0.005^2*ones(2,1) ]
%% GPU data transfer
% Transfer data to GPU
if net.gpu == 1
    x        = gpuArray(x);
    y        = gpuArray(y);
end
%% Run
task.runRegression(net, x, y, [], [])