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
dataName         = 'wine';
path                     = char([cd ,'/data/']);
load(char([path, '/redWine.mat']))
load(char([path, '/redWineTestIndices.mat']))
load(char([path, '/redWineTrainIndices.mat']))
nobs                     = size(data,1);
ncvr                     = 11;
% Input features
x                        = data(:,1:ncvr);
% Output targets
y                        = data(:,end); 
nx                       = size(x, 2);        
ny                       = size(y, 2);

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
net.layer          = [1         1      1     ];
net.nodes          = [net.nx    50     net.ny]; 
net.actFunIdx      = [0         4      0     ];
net.NoiseActFunIdx = 1;
net.NoiseActFun_LL_Idx = 4;
% Observations standard deviation
net.learnSv        = 1;% Online noise learning
net.sv             = 0;%0.32*ones(1,1, net.dtype);  
net.noiseType      = 'hete';
% Added terms
net.epoch          = 1;
net.v2hat_LL       = [0.2  0.2];
net.idx_LL_M       = [];
net.var_mode       = 0;
% Parameter initialization
net.initParamType  = 'He';
net.gainSb_v2hat   = 1;
net.gainM_v2hat    = 1;
% Maximal number of epochs and splits
net.maxEpoch       = 100;   
net.numSplits      = 20;

% Cross-validation for v2hat_LL
net.cv_v2hatLL  = 0;
net.numFolds    = 1;
net.permuteData = 2;    % 1 for split, else for fold
net.ratio       = 0.8;
% Cross-validation for HP
net.cv_HP       = 0;
%% Hierarchical Prior for variance
net.HP    = 0;
net.HP_M  = 2;    % 1 for full , 2 for layerwise
net.xv_HP = 0.005^2;
net.HP_BNI  = [];
% Two layer properties
net.init = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid Search for Gain parameter

[Gain_factor, run_time, patience]  = opt.GridsearchV2(net,x,y, trainIdx);
Gains                    = [Gain_factor(1);Gain_factor(2)];

disp(['   Gain mean : ' num2str(Gain_factor(1))])
disp(['   Gain v2hat : ' num2str(Gain_factor(2))])

save('loadGains.mat','Gains')
save('loadPatience.mat','patience')
%% writing data to a text file
fileID = fopen('GS_Wine.txt','w');
fprintf(fileID, 'Grid search results\n\n');
fprintf(fileID,'%f %f\n',[Gains;run_time;patience]);
fclose(fileID);