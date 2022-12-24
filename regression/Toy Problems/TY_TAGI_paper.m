%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         toyExample with one network
% Description:  Apply TAGI to heteroscedasticity
% Authors:      Luong-Ha Nguyen & James-A. Goulet & Bhargob Deka
% Created:      Dec 9, 2020
% Updated:      Dec 9, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca &
% bhargob.deka@mail.polymtl.ca
% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet & Bhargob Deka
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
% clc
% close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
rand_seed=4;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
%% Data
rng(1223) % Seed
f           = @(x) (5*x).^3/50;
ntrain      = 40;
ntest       = 100;
nval        = 20;
sv          = 3/50;
xtrain      = (rand(ntrain, 1)*8 - 4)/5;
xtest       = linspace(-1,1,ntest)';

ytrainTrue  = f(xtrain);
ytrain      = f(xtrain) + normrnd(0,sv, [ntrain, 1]);

nx         = size(xtrain, 2);
ny         = size(ytrain, 2);
% figure;
% scatter(xtrain,ytrain,'ob');
% xlabel('x');
% ylabel('y');


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
net.ny             = 1*ny;             % 2*ny for 'hete'
% Batch size 
net.batchSize      = 1; 
net.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = [0 0 0];
net.layer          = [1          1      1     ];
net.nodes          = [net.nx     50    net.ny ]; 
net.actFunIdx      = [0          4      0     ];
% Observations standard deviation
net.learnSv        = 0;% Online noise learning
net.sv             = 3/50*ones(1,1,net.dtype);%0.32*ones(1,1, net.dtype);  %BD
net.noiseType      = 'non';      %'hom' or 'hete'                          %BD
net.NoiseActFunIdx = 2;                                                    %BD
net.NoiseActFun_LL_Idx = 4;
net.epoch          = 1;
net.v2hat_LL       = [0.2 0.2];
net.idx_LL_M       = [];
net.var_mode       = 0;
net.HP             = 1;
net.init           = [];
% Parameter initialization
net.initParamType  = 'He';
net.gainS          = 0.25*ones(1, length(net.layer)-1);
net.gainS_v2hat    = 0.5;     %0.3 best for xavier and 0.15 best for He
net.gainSb_v2hat   = 1;
net.gainM_v2hat    = 1;
% Maximal number of epochs and splits
net.maxEpoch       = 50;   
net.numSplits      = 1;
% Cross-validation for v2hat_LL
net.cv_v2hatLL  = 0;
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
% Initialization          
saveModel  = net.saveModel;
maxEpoch   = net.maxEpoch;
svinit     = net.sv;

% Train net
net.trainMode = 1;
[net, states, maxIdx, netInfo] = network.initialization(net);

% Test net
netT              = net;
netT.trainMode    = 0;
netT.batchSize    = 1;
netT.repBatchSize = 1;
normStatT         = tagi.createInitNormStat(netT); 
[netT, statesT, maxIdxT] = network.initialization(netT); 


% Initalize weights and bias
% if net.HP == 1
%     theta = tagi.initializeWeights_HP_MeanOnly(net);
% else
%     theta = tagi.initializeWeightBias(net);
% end
theta = tagi.initializeWeightBias(net);

normStat = tagi.createInitNormStat(net);
net.sv   = svinit;

% metrics
RMSE        = zeros(net.maxEpoch,1);
LLlist      = zeros(net.maxEpoch,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
rumtime_s = tic;
stop  = 0;
epoch = 0;
%Open the video writer object.
% v = VideoWriter('TY1_T4_He.avi');
% v.FrameRate = 1;   % showing one frame per sec
% open(v);
while ~stop
    epoch = epoch + 1;
    net.epoch   = epoch;      %BD
    net.epochCV = epoch;
%     [theta, normStat,zl,Szl,sv, Prior_act_v2hat, Pos_act_v2hat, K_Gain] = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
    [theta, normStat] = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
% Testing   
    [~, ~, zl, Szl] = network.regression(netT, theta, normStatT, statesT, maxIdxT, xtest, []);



    if epoch >= maxEpoch; break;end
end

% Testing
xtest      = linspace(-1,1,ntest)';
ytestTrue  = f(xtest);
ytest      = ytestTrue + normrnd(0,sv, [ntest, 1]);
nObsTest   = size(xtest, 1);

[~, ~, yp, Syp] = network.regression(netT, theta, normStatT, statesT, maxIdxT, xtest, []);
if net.learnSv==1&&strcmp(net.noiseType, 'hete') % Online noise inference
    ypM  = reshape(yp', [net.nl, 1, 2, nObsTest]);
    SypM = reshape(Syp', [net.nl, 1, 2, nObsTest]);
    
    mv2  = reshape(ypM(:,:,2,:), [net.nl*nObsTest, 1]);
    Sv2  = reshape(SypM(:,:,2,:), [net.nl*nObsTest, 1]);
    mv2a = act.NoiseActFun(mv2, Sv2, net.NoiseActFunIdx, net.gpu);
    
    ml   = reshape(ypM(:,:,1,:), [net.nl*nObsTest, 1]);
    Sl   = reshape(SypM(:,:,1,:), [net.nl*nObsTest, 1]) + mv2a;
else
    ml = yp;
    if strcmp(net.noiseType, 'homo')
        Sl = Syp+net.sv(1);
    else
        Sl = Syp+net.sv(1).^2;
    end
end

%%  Plotting Data
figure;
scatter(5*xtrain,50*ytrain,'+b');hold on
scatter(5*xtest,50*ytest,'dm');hold on
plot(5*xtest,50*ml,'k');hold on
patch(5*[xtest' fliplr(xtest')],50*[ml' + 3*sqrt(Sl') fliplr(ml' - 3*sqrt(Sl'))],'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.2)
set(gca,'ytick',[-50 0 50], 'xtick', [-5 0 5])
% h=legend('$y_{true}$','$y_{true} \pm \sigma_{true}$','train','test','$E[\hat{y}]$','$E[\hat{y}] \pm 1\sigma$');
% set(h,'Interpreter','latex')
% xlabel('x','Interpreter','latex')
% ylabel('y','Interpreter','latex')



set(gcf,'Color',[1 1 1])
opts=['scaled y ticks = false,',...
    'scaled x ticks = false,',...
    'x label style={font=\large},',...
    'y label style={font=\large},',...
    'z label style={font=\large},',...
    'legend style={font=\large},',...
    'title style={font=\large},',...
    'mark size=5',...
    ];
%matlab2tikz('figurehandle',gcf,'filename',[ 'regression_1D_early_stop.tex'] ,'standalone', true,'showInfo', false,'floatFormat','%.5g','extraTikzpictureOptions','font=\large','extraaxisoptions',opts);


