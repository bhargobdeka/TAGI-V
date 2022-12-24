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
clc
% clc
% close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
% rand_seed=1;
% RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));
%% Data
rng(1223) % Seed
ntrain     = 500;
f          = @(x) -(x+0.5).*sin(3*pi*x);
noise      = @(x) 0.45*(x+0.5).^2;

xtrain     = [rand(ntrain, 1)*1 - 0.5]; % Generate covariate between [-1, 1];
noiseTrain = noise(xtrain);
ytrain     = f(xtrain) + normrnd(zeros(length(noiseTrain), 1), sqrt(noiseTrain));

%% Load Train Data
load('xtrain_TY1_TAGI_BNI.mat')
load('ytrain_TY1_TAGI_BNI.mat')
% xtrain = xtrainData;
% ytrain = ytrainData';
% load('xtrain_DVI.mat')
% load('ytrain_DVI.mat')
% xtrain = xtrain_DVI;
% ytrain = ytrain_DVI';
xtrain_OGData = xtrain;
ytrain_OGData = ytrain;

%xtest  = linspace(-1,1,ntest)';
nx     = size(xtrain, 2);
ny     = size(ytrain, 2);
% load('xtest_DVI.mat')
% load('ytest_DVI.mat')
% xtest = xtest_DVI;
% ytest = ytest_DVI';

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
net.saveModel      = 1;
% GPU
net.gpu            = 0;
net.numDevices     = 0;
% Number of input covariates
net.nx             = nx; 
% Number of output responses
net.nl             = ny;
net.nv2            = ny;
% Batch size 
net.batchSize      = 1; %1
net.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = [0 0 0];
net.layer          = [1           1      1          1        ];
net.nodes          = [net.nx     128    128      net.ny      ]; 
net.actFunIdx      = [0           4      4        0          ];
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



% Splits   
net.numSplits      = 1; 
% Cross-validation for v2hat_LL
net.cv_v2hatLL     = 0;
net.numFolds       = 1;
net.permuteData    = 2;    % 1 for split, else for fold
net.ratio          = 0.8;
% Cross-validation for HP
net.cv_HP          = 0;
%% Load Gain factors or HP
net.gs_Gain        = 1;    % 1 for grid-search, 0 for HP learning
if net.gs_Gain == 1
    net.gainS          = 1*ones(1,length(net.layer)-1);
    net.gainS_v2hat    = 1;
    net.gainSb_v2hat   = 1;
    net.gainM_v2hat    = 1;
else
    net.gain_HP(1,:)   = [1  1];
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
net.early_stop = 0;
if net.early_stop == 1
    net.maxEpoch       = 200;
    net.val_data        = 1;
else
    net.maxEpoch       = 100;
    net.val_data        = 1;
end
%% Hierarchical Prior for variance
net.HP   = 0;
net.HP_M = 2;    % 1 for full , 2 for layerwise
net.xv_HP   = 0.005^2;
net.HP_BNI  = [[0.5*ones(650,1);0.5*ones(50,1);2.7e-05*50*ones(50,1);ones(50,1);ones(2,1)] [(0.1/13)*ones(650,1);(0.1/50)*ones(50,1);1e-05*ones(50,1);(0.01/13)*ones(50,1);(1/50)*ones(2,1) ] ];  %[0.001^2*ones(650,1);0.001^2*ones(50,1);1e-05^2*ones(50,1);0.01^2*ones(50,1);0.005^2*ones(2,1) ]

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

% Val net
netV              = net;
netV.trainMode    = 0;
netV.batchSize    = 1;
netV.repBatchSize = 1;
[netV, statesV, maxIdxV] = network.initialization(netV);
normStatV = tagi.createInitNormStat(netV);
%% validation set
if net.val_data == 1
    %  Validation set
    [xtrain, ytrain, xval, yval] = dp.split(xtrain, ytrain,0.8);
end
% Cross-Validation for Epoch for HP                           %BD
if net.cv_HP == 1
    maxEpoch          = opt.crossvalHP(net, xtrain, ytrain);
    disp([' No. of Epoch ' num2str(maxEpoch)])
end
% Initalize weights and bias
theta    = tagi.initializeWeightBias(net);
normStat = tagi.createInitNormStat(net);
net.sv   = svinit;

% Normalize Data
%xtest      = linspace(-5,5,200)';
% metrics
RMSE         = zeros(net.maxEpoch,1);
LLlist_test  = zeros(net.maxEpoch,1);
LLlist_train = zeros(net.maxEpoch,1);
%FigHandle   = figure;
%set(FigHandle, 'Position', [500, 100, 600, 400])
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
rumtime_s    = tic;
stop         = 0;
epoch        = 0;
delta        = 0.01;
patience     = 5;
best_LL_val  = nan;
counter      = 0;
LL_vals      = [];
%Open the video writer object.
% v = VideoWriter('TY1_T4_He.avi');
% v.FrameRate = 1;   % showing one frame per sec
% open(v);

while ~stop
    epoch = epoch + 1;
    epoch
    net.epoch   = epoch;      %BD
    net.epochCV = epoch;
    if epoch >1
        idxtrain = randperm(size(ytrain, 1));
        ytrain   = ytrain(idxtrain, :);
        xtrain   = xtrain(idxtrain, :);
    end

    % Training
    [theta, normStat, zl_train, Szl_train] = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
    if net.val_data == 1
        [~, ~, ml_val, Sl_val] = network.regression(netV, theta, normStatV, statesV, maxIdxV, xval, []);
        LL_val                 = mt.loglik(yval, ml_val(:,1), Sl_val(:,1));
        LL_vals                = [LL_vals;LL_val];
        if any(isnan(LL_val))
            LL_val = -inf;
        end
    end
    if net.early_stop == 1
        %% Early Stopping
        if ~isnan(LL_val)
            if isnan(best_LL_val)
                best_LL_val = LL_val;
            elseif LL_val - best_LL_val > delta
                best_LL_val = LL_val;
                counter   = 0;
            elseif LL_val - best_LL_val < delta
                counter = counter + 1;
                disp(['   counter #' num2str(counter) ' out of' num2str(patience)])
                if counter >= patience
                    maxEpoch = epoch;
                end
            end
        else
            break;
        end
    end
    if epoch >= maxEpoch; break;end
end
figure;
scatter(1:maxEpoch,LL_vals,'dm');
xlabel('epoch')
ylabel('test log-likelihood')
if net.early_stop == 0
    hold on;plot([28, 28],[min(LL_vals), max(LL_vals)],'-k')
end
drawnow
run_time_e = toc(rumtime_s);
% LLlist_test(end)
% Plotting
% figure;
% scatter(1:maxEpoch,LLlist_train,1,'ob');hold on;scatter(1:maxEpoch,LLlist_test,1,'dr')
% %ylim([-1, -0.4])
% xlabel('epoch')
% ylabel('LL')
% legend('train','test')
% [~,N_train] = max(LLlist_train);
% [~,N_test] = max(LLlist_test);

xtest  = linspace(-1,1,200)';
nObsTest   = size(xtest, 1);
[~, ~, yp, Syp] = network.regression(netT, theta, normStatT, statesT, maxIdxT, xtest, []);
if net.learnSv==1&&strcmp(net.noiseType, 'hete') % Online noise inference
    ypM  = reshape(yp', [net.nl, 1, 2, nObsTest]);
    SypM = reshape(Syp', [net.nl, 1, 2, nObsTest]);
    
    mv2  = reshape(ypM(:,:,2,:), [net.nl*nObsTest, 1]);
    Sv2  = reshape(SypM(:,:,2,:), [net.nl*nObsTest, 1]);
    
    ml   = reshape(ypM(:,:,1,:), [net.nl*nObsTest, 1]);
    Sl   = reshape(SypM(:,:,1,:), [net.nl*nObsTest, 1]);
else
    ml = yp;
    if strcmp(net.noiseType, 'homo')
        Sl = Syp+net.sv(1);
    else
        Sl = Syp+net.sv(1).^2;
    end
end
Pred_TAGI_BNI = [ml Sl];
%%  Plotting Data
xTrue = sort(xtrain_OGData);
yTrue = f(xTrue);
sTrue = noise(xTrue);
load('Pred_DNN.mat')
figure;
%scatter(xtrain,ytrain,'+m');hold on
plot(xTrue,yTrue,'r');hold on
patch([xTrue' fliplr(xTrue')],[yTrue' + sqrt(sTrue') fliplr(yTrue' - sqrt(sTrue'))],'red','EdgeColor','none','FaceColor','red','FaceAlpha',0.3)
plot(xtest,ml,'k');hold on
patch([xtest' fliplr(xtest')],[ml' + sqrt(Sl') fliplr(ml' - sqrt(Sl'))],'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.3);hold on
% plot(xtest,Pred_DNN(:,1),'b');hold on
% patch([xtest' fliplr(xtest')],[Pred_DNN(:,1)' + sqrt(Pred_DNN(:,2)') fliplr(Pred_DNN(:,1)' - sqrt(Pred_DNN(:,2)'))],'blue','EdgeColor','none','FaceColor','blue','FaceAlpha',0.3);
h=legend('ytrue','$\pm 1$ true stdv.','$\mu$','$\mu \pm \sigma$');
set(h,'Interpreter','latex')
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
title('TAGI-BNI')
% xlim([-5,5])
% ylim([-5,7])
set(gcf,'Color',[1 1 1])
opts=['scaled y ticks = false,',...
    'scaled x ticks = false,',...
    'x label style={font=\large},',...
    'y label style={font=\large},',...
    'z label style={font=\large},',...
    'legend style={font=\large},',...
    'title style={font=\Large},',...
    'mark size=5',...
    ];