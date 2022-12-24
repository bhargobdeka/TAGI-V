%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         toyExample with one network
% Description:  Apply TAGI to heteroscedasticity
% Authors:      Bhargob Deka & Luong-Ha Nguyen & James-A. Goulet
% Created:      Aug 18, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca &
% bhargob.deka@mail.polymtl.ca
% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet & Bhargob Deka
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc; 
% clc
% close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
rand_seed=1;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed)); %Initialize random stream number based on clock
%% Data
% f = @(x,w,z) z.*10.*cos(x) + (1-z).*10.*sin(x) + w;
% n = 2500;
% for i = 1:10
%     w_obs      = normrnd(zeros(n, 1), 1);
%     z_obs      = double(rand(1,n)<0.5)';
%     x_obs      = unifrnd(-2,2,[n,1]);
%     y_obs      = f(x_obs, w_obs, z_obs);
%     Data.x_obs = x_obs;
%     Data.y_obs = y_obs;
%     Data.w_obs = w_obs;
%     Data.z_obs = z_obs;
%     save(['dataset',num2str(i),'.mat'],'Data');
% end
%% Load Indices
path  = char([cd ,'/Indices/']);
load(char([path, '/DepewegBiModTestIndices.mat']))
load(char([path, '/DepewegBiModTrainIndices.mat']))
%% Load Datasets
for k = 1:10
    path  = char([cd ,'/Datasets/']);
    load(char([path, '/dataset',num2str(k),'.mat']))
    %% Net
    % GPU 1: yes; 0: no
    %% Hyper-parameters (user-defined)
    % Batch size
    net.batchSize      = 1;
    net.repBatchSize   = 1;
    % Maximal number of epochs and splits
    net.maxEpoch       = 100;
    net.numSplits      = 5;
    %% Fixed parameters
    nx                 = 1;
    ny                 = 1;
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
    
    % Layer| 1: FC; 2:conv; 3: max pooling;
    % 4: avg pooling; 5: layerNorm; 6: batchNorm
    % Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
    net.imgSize        = [0 0 0];
    net.layer          = [1         1     1        1       ];
    net.nodes          = [net.nx   20    20     net.ny     ];
    net.actFunIdx      = [0         4     4        0       ];
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
    % net.gainS          = 0.5*ones(1, length(net.layer)-1);
    % net.gainS_v2hat    = 1;     %0.3 best for xavier and 0.15 best for He
    net.gainSb_v2hat   = 1;
    net.gainM_v2hat    = 1;
    
    % Cross-validation for v2hat_LL
    net.cv_v2hatLL  = 0;
    net.numFolds    = 5;
    net.permuteData = 2;    % 1 for split, else for fold
    net.ratio       = 0.9;
    % Cross-validation for HP
    net.cv_HP       = 0;
    
    % Grid Search for Gain factor
    net.gs_Gain     = 1;
    %% Hierarchical Prior for variance
    net.HP          = 0;
    net.HP_M        = 2;    % 1 for full , 2 for layerwise
    net.xv_HP       = 0.005^2;
    net.HP_BNI      = [];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Grid Search for Gain parameter
    
    [Gain_factor, opt_E]      = opt.Gridsearch(net,Data.x_obs,Data.y_obs, trainIdx);
    % run_time = toc(start);
    % disp(['   Gain mean : ' num2str(run_time)])
    Gains                     = [Gain_factor(1) Gain_factor(2)];
    opt_Epochs                = opt_E;
    disp(['   Gain mean : ' num2str(Gain_factor(1))])
    disp(['   Gain v2hat : ' num2str(Gain_factor(2))])
    
    save([cd ,'/Gains/','Gains',num2str(k),'.mat'],'Gains')
    save([cd ,'/Opt_Epochs/','opt_Epochs',num2str(k),'.mat'],'opt_Epochs')
end
