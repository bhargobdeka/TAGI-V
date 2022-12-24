%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         energy
% Description:  Apply TAGI to energy
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      July 21, 2020
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
% rng(123456789)
%% Data
modelName        = 'FC';
dataName         = 'yacht';
path                     = char([cd ,'/data/']);
load(char([path, '/yacht.mat']))
load(char([path, '/yachtTestIndices.mat']))
load(char([path, '/yachtTrainIndices.mat']))
nobs                     = size(data,1);
ncvr                     = 6;
% Input features
x                        = data(:,1:ncvr);
% Output targets
y                        = data(:,end); 
nx                       = size(x, 2);        
ny                       = size(y, 2); 
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
net.ny             = 1*ny; 
% Batch size 
net.batchSize      = 1; 
net.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = [0 0 0];

%% Net for mean
netD                 = net;
netD.lastLayerUpdate = 0;
netD.learnSv         = 0;% Online noise learning
netD.sv              = 0;%0.32*ones(1,1, net.dtype);  
netD.noiseType       = 'hete';
netD.layer           = [1         1      1     ];
netD.nodes           = [net.nx   50   netD.ny]; 
netD.actFunIdx       = [0         4     0      ];
% Parameter initialization
netD.initParamType  = 'He';
netD.gainS          = 0.5*ones(1, length(netD.layer)-1);
netD.gainS_v2hat     = 0.5;
netD.init           = 'D';
%% Net for v2hat
netW                   = net;
netW.lastLayerUpdate   = 0;
netW.learnSv           = 0;% Online noise learning
netW.sv                = 0;%0.32*ones(1,1, net.dtype); 
netW.layer             = [1         1      1     ];
netW.nodes             = [net.nx   50    netD.ny ]; 
netW.actFunIdx         = [0         4     0      ];
netW.NoiseActFunIdx    = 4;                                          %BD
net.NoiseActFun_LL_Idx = 4;
% Parameter initialization
netW.initParamType   = 'He';
netW.gainS           = (0.01)*ones(1, length(netD.layer)-1);
netW.gainS_v2hat     = 0.01;
netW.init            = 'W';
%% Observations standard deviation
net.learnSv        = 0;
net.sv             = 0;%0.32*ones(1,1, net.dtype);  
net.noiseType      = 'hete';
net.epoch          = 1;
net.v2hat_LL       = [0.2 0.2];
net.idx_LL_M       = 1;
net.var_mode       = 0;
% Maximal number of epochs and splits
net.maxEpoch       = 40;   
net.numSplits      = 20;


% Cross-validation for v2hat_LL
net.cv_v2hatLL  = 1;
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
task.runRegressionWithNI2(net,netD,netW, x, y, trainIdx, testIdx)