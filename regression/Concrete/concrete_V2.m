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
rng(1223)
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

%% Data Net
% GPU 1: yes; 0: no
netD.task           = 'regression';
netD.modelName      = modelName;
netD.dataName       = dataName;
netD.cd             = cd;
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
netD.ny             = ny; 
% Batch size 
netD.batchSize      = 1; 
netD.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
netD.imgSize        = [0 0 0];
netD.layer          = [1         1      1    1];
netD.nodes          = [netD.nx   50     50  netD.ny]; 
netD.actFunIdx      = [0         4      4    0];
% Observations standard deviation
netD.learnSv        = 1;% Online noise learning
netD.sv             = [0.1 0.01];%0.32*ones(1,1, netD.dtype);  
netD.noiseType      = 'homo';
% Parameter initialization
netD.initParamType  = 'Xavier';
netD.gainS          = 0.25*ones(1, length(netD.layer)-1);
% Maximal number of epochs and splits
netD.maxEpoch       = 40;   
netD.numSplits      = 20;
netD.lastLayerUpdate = 1;

%% Noise Net
% GPU 1: yes; 0: no
netN.task           = 'regression';
netN.modelName      = modelName;
netN.dataName       = dataName;
netN.cd             = cd;
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
netN.batchSize      = netD.batchSize ; 
netN.repBatchSize   = netD.repBatchSize; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
netN.imgSize        = [0 0 0];
netN.layer          = [1         1      1];
netN.nodes          = [netN.nx   50     netN.ny]; 
netN.actFunIdx      = [0         4      0];
netN.NoiseActFunIdx = 1; 
% Observations standard deviation
netN.learnSv        = 0;% Online noise learning
netN.sv             = 0;%0.32*ones(1,1, netN.dtype);  
netN.lastLayerUpdate = 0;
% Parameter initialization
netN.initParamType  = 'Xavier';
netN.gainS          = 0.25*ones(1, length(netN.layer)-1);
% Maximal number of epochs and splits
netN.maxEpoch       = 40;   
netN.numSplits      = 20;

%% Run
task.runRegressionWithNI(netD, netN, x, y, trainIdx, testIdx)

