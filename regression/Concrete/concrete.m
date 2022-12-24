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
net.batchSize      = 32; 
net.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = [0 0 0];
net.layer          = [1         1      1      ];
net.nodes          = [net.nx    50     net.ny ]; 
net.actFunIdx      = [0         4      0      ];
net.NoiseActFunIdx = 1;
net.NoiseActFun_LL_Idx = 4;
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
net.gainS_v2hat    = 1e-04;
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
else
    net.gain_HP(1,:)   = [Gains(1)  Gains(1)];
    net.gain_HP(2,:)   = [Gains(2)  Gains(2)];
    net.gainSb_v2hat   = 1;
    net.gainM_v2hat    = 1;
    alpha = 0.01*(1-0.9); beta = 0.01*(1-0.9);
    net.m_w_v2hat      = [alpha*(1-1/net.nx)*Gains(1)*(1/net.nx)       1e-10       1e-10 ]; % 0.1*alpha*(1-1/50)*Gains(1)*(1/50) % alpha*Gains(2)*(1/50)
    net.m_b_v2hat      = [beta*(1/net.nx)                              1e-10        1e-10 ]; % beta*(1/50) % beta*(1/50)
    
    net.var_w_v2hat    = [(alpha*(1-1/net.nx)*Gains(1)*(1/net.nx))^2   1e-10       1e-10  ]; %(0.1*alpha*(1-1/50)*Gains(1)*(1/50))^2 %(alpha*Gains(2)*(1/50))^2
    net.var_b_v2hat    = [(beta*(1/net.nx))^2                          1e-10       1e-10  ]; %(beta*(1/50))^2 %(beta*(1/50))^2
end
% % creating mesh
load('EEW2g.mat');
load('SEW2g.mat');
load('aOg.mat');
net.EEW2g = EEW2g;
net.SEW2g = SEW2g;
net.aOg   = aOg;
%% Early Stopping
net.epochlist  = [];
net.early_stop = 0;
if net.early_stop == 1
    net.maxEpoch       = 100;
    net.val_data        = 1;
else
    net.maxEpoch       = 100; %18
    net.val_data        = 0;
end
%% Hierarchical Prior for variance
net.HP   = 0;
net.HP_M = 2;    % 1 for full , 2 for layerwise
net.xv_HP = 0.005^2;
net.HP_BNI  = [[0.5*ones(650,1);0.5*ones(50,1);2.7e-05*50*ones(50,1);ones(50,1);ones(2,1)] [(0.1/13)*ones(650,1);(0.1/50)*ones(50,1);1e-05*ones(50,1);(0.01/13)*ones(50,1);(1/50)*ones(2,1) ] ];  %[0.001^2*ones(650,1);0.001^2*ones(50,1);1e-05^2*ones(50,1);0.01^2*ones(50,1);0.005^2*ones(2,1) ]
% Two layer properties
net.init = [];

% net.patience = 3;
% syms a b;
% ndiv=100;
% Y1      = 1.5.^(linspace(log(0.0000001),log(30),ndiv));
% Y1(1:7) = 1e-05.*Y1(1:7);
% [net.EEW2g,net.SEW2g] = meshgrid(Y1);
% net.aOg=zeros(ndiv,ndiv);
% for i=1:ndiv
%     for j=1:ndiv
%         net.aOg(i,j)=single(root(net.SEW2g(i,j)*a^3 - 4*net.SEW2g(i,j)*a^2 - net.EEW2g(i,j)^2*a^2 - 2*net.EEW2g(i,j)^2*a + 5*net.SEW2g(i,j)*a - 2*net.SEW2g(i,j) - net.EEW2g(i,j)^2, a, 3));
%     end
% end
%% Run
task.runRegression(net, x, y, trainIdx, testIdx)

