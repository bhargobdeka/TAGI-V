%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
rand_seed=4;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
% rng(123456)
%% Data
xtrain = load('xtrain_TY1_TAGI_BNI.mat');
ytrain = load('ytrain_TY1_TAGI_BNI.mat');
nx     = size(xtrain, 2);
ny     = size(ytrain, 2);

%% Neural Network properties
% GPU
NN.gpu                       = 0;
% Data type object single or double precision
NN.dtype                     = 'single';
% Number of input covariates
NN.nx                        = nx;
% Number of output responses
NN.ny                        = ny;
% Batch size
NN.batchSize                 = 1;
NN.errorRateDisplay          = 0;
% Number of nodes for each layer
NN.nodes                     = [NN.nx 128 128 NN.ny];
% Input standard deviation
NN.sx                        = nan;
% Observations standard deviation
% NN.sv                        = 0.2345;
% Maximal number of learnign epoch
NN.maxEpoch                  = 100;
% Factor for initializing weights & bias
NN.factor4Bp                 = 1E-2*[1 1 1 1 1];
NN.factor4Wp                 = 0.25*[1 1 1 1 1];
% Activation function for hidden layer {'tanh','sigm','cdf','relu'}
NN.hiddenLayerActivation     = 'relu';
% Activation function for hidden layer {'linear', 'tanh','sigm','cdf','relu'}
NN.outputActivation          = 'linear';
% Weight percentaga being set to 0
NN.dropWeight                = 1;
% Replicate a net for testing
NN.errorRateEval = 0;
NNtest                       = NN;
% Train network
% Indices for each parameter group
NN = indices.parameters(NN);
NN = indices.covariance(NN);
% Initialize weights & bias
[mp, Sp] = tagi.initializeWeightBias(NN);
%load('mp.mat')

% Test network
NNtest.batchSize = 1;
NNtest.trainMode = 0;
% Indices for each parameter group
NNtest = indices.parameters(NNtest);
NNtest = indices.covariance(NNtest);


% Loop initialization
NN.trainMode = 1;
stop         = 0;
epoch        = 0;
FigHandle = figure;
set(FigHandle, 'Position', [100, 100, 400, 200])