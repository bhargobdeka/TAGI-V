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
rand_seed=1;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed)); 
%% Load Data
% path  = char([cd ,'/data/']);
% load(char([path, '/MotorCycle.mat']))
load('trainData.mat')
load('testData.mat')
xtrain = trainData(:,1);
ytrain = trainData(:,2);
xtrain_OGData = xtrain;
ytrain_OGData = ytrain;
% figure;
% scatter(xtrain_OGData,ytrain_OGData,'+r');
% h=legend('train');
% set(h,'Interpreter','latex')
% xlabel('x','Interpreter','latex')
% ylabel('y','Interpreter','latex')
% xlim([-5,5])
% ylim([-5,7])

xtest  = testData(:,1);
ytest  = testData(:,2);
nx     = size(xtrain, 2);
ny     = size(ytrain, 2);
%% Net
% GPU 1: yes; 0: no
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
net.ny             = 2*ny;             % 2*ny for 'hete'
% Batch size
net.batchSize      = 1;
net.repBatchSize   = 1;
% Layer| 1: FC; 2:conv; 3: max pooling;
% 4: avg pooling; 5: layerNorm; 6: batchNorm
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = [0 0 0];
net.layer          = [1         1      1          ];
net.nodes          = [net.nx   200     net.ny     ];
net.actFunIdx      = [0         4      0         ];
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
net.gainS          = 1*ones(1, length(net.layer)-1);
net.gainS_v2hat    = 1;     %0.3 best for xavier and 0.15 best for He
net.gainSb_v2hat   = 1;
net.gainM_v2hat    = 1;
% Maximal number of epochs and splits
net.maxEpoch       = 100;
net.numSplits      = 1;
% Cross-validation for v2hat_LL
net.cv_v2hatLL  = 0;
net.numFolds    = 5;
net.permuteData = 2;    % 1 for split, else for fold
net.ratio       = 0.8;
% Cross-validation for HP
net.cv_HP       = 0;

% Grid Search for Gain factor
net.gs_Gain     = 0;
%% Hierarchical Prior for variance
net.HP        = 0;
net.HP_M      = 2;    % 1 for full , 2 for layerwise
net.xv_HP     = 0.005^2;
net.HP_BNI    = [];
Nsplit        = net.numSplits;
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
rumtime_s = tic;
stop  = 0;
epoch = 0;
%Open the video writer object.
% v = VideoWriter('TY1_T4_He.avi');
% v.FrameRate = 1;   % showing one frame per sec
% open(v);
% Load Theta
[mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
load('TY_theta_100E_03.mat');
[mw_L, Sw_L, mb_L, Sb_L] = tagi.extractParameters(theta_learnt03);
theta = tagi.compressParameters(mw_L, Sw, mb_L, Sb, mwx, Swx, mbx, Sbx); 

%close all
%figure;
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
%     [theta, normStat,zl,Szl,sv, Prior_act_v2hat, Pos_act_v2hat, K_Gain] = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
    % Training
    [theta, normStat, zl_train, Szl_train] = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
    % Theta
    [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
%   mv2a_train = Prior_act_v2hat_train(:,1);
    ml_train   = zl_train(:,1);
    Sl_train   = Szl_train(:,1);
    LLlist_train(epoch)  = mt.loglik(ytrain, ml_train, Sl_train);
    % Testing
    %xtest      = linspace(-5,5,200)';
    [~, ~, zl_test, Szl_test,~, variance] = network.regression(netT, theta, normStatT, statesT, maxIdxT, xtest, []);
%     mv2a_test = Prior_act_v2hat_test(:,1);
    ml_test    = zl_test(:,1);
    Sl_test    = Szl_test(:,1);
    S_var_test = variance(:,1);
    Sl_z0_test = Sl_test - S_var_test;
    LLlist_test(epoch) = mt.loglik(ytest, ml_test, Sl_test);
    % Plot
%     subplot(1,2,1);
%     scatter(xtrain_OGData,ytrain_OGData,'+r');hold on
%     %scatter(xtest,ytest,'dm');hold on
%     plot(xtest,ml_test,'k');hold on
%     patch([xtest' fliplr(xtest')],[ml_test' + sqrt(Sl_test') fliplr(ml_test' - sqrt(Sl_test'))],'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.3);hold on
%     patch([xtest' fliplr(xtest')],[ml_test' + sqrt(Sl_z0_test') fliplr(ml_test' - sqrt(Sl_z0_test'))],'blue','EdgeColor','none','FaceColor','blue','FaceAlpha',0.3);hold on
% %     patch([xtest' fliplr(xtest')],[ml_test' + sqrt(Sl_z0_test') fliplr(ml_test' - sqrt(S_var_test'))],'red','EdgeColor','none','FaceColor','red','FaceAlpha',0.3);
% %     h=legend('train','$\mu$','$\mu \pm \sigma$');
% %     set(h,'Interpreter','latex')
%     xlabel('x','Interpreter','latex')
%     ylabel('y','Interpreter','latex')
%     title('epistemic')
%     xlim([-5,5])
%     ylim([-5,7])
%     hold off
%     subplot(1,2,2);
%     scatter(xtrain_OGData,ytrain_OGData,'+r');hold on
%     %scatter(xtest,ytest,'dm');hold on
%     plot(xtest,ml_test,'k');hold on
%     patch([xtest' fliplr(xtest')],[ml_test' + sqrt(Sl_test') fliplr(ml_test' - sqrt(Sl_test'))],'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.3);hold on
%     %patch([xtest' fliplr(xtest')],[ml_test' + sqrt(Sl_z0_test') fliplr(ml_test' - sqrt(Sl_z0_test'))],'blue','EdgeColor','none','FaceColor','blue','FaceAlpha',0.3);hold on
%     patch([xtest' fliplr(xtest')],[ml_test' + sqrt(S_var_test') fliplr(ml_test' - sqrt(S_var_test'))],'red','EdgeColor','none','FaceColor','red','FaceAlpha',0.3);
% %     h=legend('train','$\mu$','$\mu \pm \sigma$');
% %     set(h,'Interpreter','latex')
%     xlabel('x','Interpreter','latex')
%     ylabel('y','Interpreter','latex')
%     title('aleatory')
%     xlim([-5,5])
%     ylim([-5,7])
%     
%     drawnow
%     pause(0.2)
%     hold off
    %LLlist_test(epoch) = mt.loglik(ytest, ml_test, Sl_test);
    if epoch >= maxEpoch; break;end
end
% figure;
% scatter(1:maxEpoch, ( len_L1_w*100/length(Sw_L1)),'db');hold on
% scatter(1:maxEpoch, ( len_L2_w*100/length(Sw_L2)), 'or');
% xlabel('epoch')
% ylabel('removed weights')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting
 figure;
scatter(1:maxEpoch,LLlist_train,1,'ob');hold on;scatter(1:maxEpoch,LLlist_test,1,'dr')
%ylim([-1, -0.4])
xlabel('epoch')
ylabel('LL')
legend('train','test')
[~,N_train] = max(LLlist_train);
[~,N_test] = max(LLlist_test);
% Testing
xtest      = linspace(-5,5,200)';
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

%%  Plotting Data
figure;
scatter(xtrain_OGData,ytrain_OGData,'+r');hold on
%scatter(xtest,ytest,'dm');hold on
plot(xtest,ml,'k');hold on
patch([xtest' fliplr(xtest')],[ml' + sqrt(Sl') fliplr(ml' - sqrt(Sl'))],'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.3)
h=legend('train','$\mu$','$\mu \pm \sigma$');
set(h,'Interpreter','latex')
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
title('TAGI-BNI')
xlim([-5,5])
ylim([-5,7])
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