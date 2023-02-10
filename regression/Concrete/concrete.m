%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         concrete
% Description:  Apply TAGI to concrete dataset
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      July 22, 2020
% Updated:      December 02, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
clc
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
rand_seed=1;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed)); 
%% Data
modelName        = 'FC';
dataName         = 'concrete';
path = char([cd ,'/data/']);
load(char([path, '/concrete.mat']))
load(char([path, '/concreteTestIndices.mat']))
load(char([path, '/concreteTrainIndices.mat']))
nobs  = size(data,1);
ncvr  = 8;
ratio = 0.9;
x     = data(:,1:ncvr);
y     = data(:,end);        
nx    = size(x, 2);        
ny    = size(y, 2);

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
net.gpu            = 0;
net.numDevices     = 0;
% Data type object half or double precision
net.dtype          = 'single';
% Number of input covariates
net.nx             = nx; 
% Number of output responses
net.nl             = ny;
net.nv2            = ny;
net.ny             = 2*ny; 
% Batch size 
net.batchSize      = 10; 
net.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = [0 0 0];
net.layer          = [1         1      1      ];
net.nodes          = [net.nx    50     net.ny ]; 
net.actFunIdx      = [0         4      0      ];
% net.NoiseActFunIdx = 5;
net.NoiseActFun_LL_Idx = 4;
net.NoiseActFunIdx = 5; %1-> exp, 5-> mrelu    
% Observations standard deviation
net.learnSv        = 1;% Online noise learning
net.sv             = 0;%0.32*ones(1,1, net.dtype);  
net.noiseType      = 'hete';
% Added terms
net.epoch          = 1;
net.v2hat_LL       = [0.2 0.2];
net.idx_LL_M       = [];
net.var_mode       = 0;
% Parameter initialization
net.initParamType  = 'He';
net.gainS          = 0.5*ones(1, length(net.layer)-1);
net.gainS_v2hat    = 1e-02;
net.gainSb_v2hat   = 1;
net.gainM_v2hat    = 1;

% Splits
net.numSplits      = 20;
% Cross-validation for v2hat_LL
net.cv_v2hatLL  = 0;
net.numFolds    = 5;
net.permuteData = 2;    % 1 for split, else for fold
net.ratio       = 0.8;
% Cross-validation for HP
net.cv_HP       = 0;
%% Load Gain factors or HP
load('loadGains.mat')
load('loadEpochs.mat')
load('loadPatience.mat')
net.Gains          = Gains;

net.patience       = patience;
net.gs_Gain        = 1;    % 1 for grid-search, 0 for HP learning
if net.gs_Gain == 1
    net.gainS          = Gains(1)*ones(1,length(net.layer)-1);
    net.gainS_v2hat    = Gains(2);
    net.gainSb_v2hat   = 1;
    net.gainM_v2hat    = 1;
end
%% Early Stopping
net.epochlist  = [];
net.early_stop = 0;
if net.early_stop == 1
    net.maxEpoch       = 100;
    net.val_data        = 1;
else
    net.maxEpoch       = 18; %18
    net.val_data        = 0;
end

% Two layer properties
net.init = [];

%% Run
task.runRegression(net, x, y, trainIdx, testIdx)

