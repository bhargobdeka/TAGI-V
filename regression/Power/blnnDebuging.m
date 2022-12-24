%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         testIndices4BLNN
% Description:  Test covariance indices for BLNN 
% Author:       Luong-Ha Nguyen & James-A. Goulet
% Date:         October 30, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all
% clc
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
% gpuDevice(2);
% warning off parallel:gpu:device:DeviceLibsNeedsRecompiling
% try
%     gpuArray.eye(2)^2;
% catch ME
% end
% try
%     nnet.internal.cnngpu.reluForward(1);
% catch ME
% end
%% Boston Housing
path                     = char([cd ,'/data/']);
load(char([path, '/powerPlant.mat']))
load(char([path, '/powerPlantTestIndices.mat']))
load(char([path, '/powerPlantTrainIndices.mat']))
useGPU                       = 1;
n_obs                        = size(data,1);
n_cvr                        = 4;
alpha                        = 0.9;
N_split                      = 21;

x                            = data(:,1:n_cvr);
y                            = data(:,end);

idx_obs                      = 1:n_obs;
idx_train_end                = round(0.9*n_obs)-1;
idx_train                    = idx_obs(1:idx_train_end);
idx_test                     = idx_obs((idx_train_end+1):n_obs);
y_train                      = y(idx_train,:);
x_train                      = x(idx_train,:);
y_test                       = y(idx_test,:);
x_test                       = x(idx_test,:);
n_x                          = size(x_train,2);         %Number of input covariates
n_y                          = size(y_train,2);         %Number of output responses
n_xtest                      = size(x_test,2);
n_ytest                      = size(y_test,2);
n_train                      = length(idx_train);
n_test                       = length(idx_test);

% Normalizing 
[x_train, y_train, x_test, ~, m_xtrain, s_xtrain, m_ytrain, s_ytrain] = dp.normalize(x_train, y_train, x_test, y_test);

% Optimization
delta                        = 1E-6; 
Niter                        = 10;
eta                          = 1;

%% Feed-Forward Network properties
NN.gpu                   = 0;
NN.nodes                     = [n_x 3 2 n_y];                %Number of nodes for each layer
% NN.n_layers                  = numel(NN.nodes);   %Number of hidden layers
NN.nx                        = n_x;                 %Number of input covariates
NN.ny                        = n_y;                 %Number of output responses
NN.batchSize                 = 2; 
NN.sw                        = 0 ;                  %Bias and weights prior standard deviation
NN.sx                        = nan;
NN.sv                        = 0.1;                   %Observations standard deviation
NN.rho_y                     = 0.01;                %Observation correlation
NN.convergence_tol           = 1E-3;                %Learning convergence tolerence (i.e, stopping criterion)
NN.max_epoch                 = 10;                   %Maximal number of learnign epoch
NN.factor4Bp                 = [1 1 1];
NN.factor4Wp                 = [1 1 1];
NN.hiddenLayerActivation     = 'relu';              %{'tanh','sigm','cdf','relu'}
NN.outputActivation          = 'linear';            %{'cdf'/'sigm'/'linear}'
NN.dtype                     = 'single';
%% Indices for each parameter group
% xloop= reshape(xtrain(1:NN.batchSize,:)', [8, 1]);
NN  = ctr.parameters(NN);
NNd = NN;
NNd = ctr.diagCovarianceIndices(NNd);
NN  = ctr.fullCovarianceIndices(NN);
NN  = ctr.transitionMatrix(NN);
NN.errorRateDisplay = 0;
% Initialize weights & bias
[mp, Sp, mpd, Spd] = ctr.initializeWeightBias(NN);
% Training
NN.trainMode = 1;
NNd.trainMode = 1;
stop         = 0;
epoch        = 0;
while ~stop
    if epoch>1
        idxtrain      = randperm(length(idxtrain));
        ytrain        = ytrain(idxtrain, :);
        xtrain        = xtrain(idxtrain, :);
    end
    epoch = epoch + 1;
    [mp, Sp, ~, ~] = ctr.network(NN, NNd, mp, Sp, mpd, Spd, x_train, y_train);
    if epoch == NN.maxEpoch
        stop = 1;
    end
end
