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
ntrain     = 400;
ntest      = 100;
f          = @(x) (3*x).^3/50;
xtrain     = unifrnd(-1,1,[ntrain,1]);
noise_het  = @(x) (0.5*(x).^4)+0.02;
noise_hom  = 0.8;
noiseTrain = noise_het(xtrain).*noise_hom;
xtest      = linspace(-1,1,ntest)';
noiseTest  = noise_het(xtest).*noise_hom;
ytrainTrue = f(xtrain);
ytrain     = f(xtrain) + normrnd(0, sqrt(noiseTrain),[ntrain,1]);

ytestTrue  = f(xtest);
ytest      = ytestTrue + normrnd(0, sqrt(noiseTest),[ntest,1]);
nx         = size(xtrain, 2);
ny         = size(ytrain, 2);

subplot(1,2,1)
scatter(xtrain,noise_het(xtrain));
subplot(1,2,2)
scatter(xtrain,noiseTrain);

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
net.ny             = 2*ny; 
% Batch size 
net.batchSize      = 1; 
net.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = [0 0 0];
net.layer          = [1         1         1      1     ];
net.nodes          = [net.nx   125       125    net.ny ]; 
net.actFunIdx      = [0         4         4      0     ];
net.NoiseActFunIdx     = 4;
net.NoiseActFun_LL_Idx = 4;
% Observations standard deviation
net.learnSv        = 1;% Online noise learning
net.sv             = 0;%0.32*ones(1,1, net.dtype);  
net.noiseType      = 'hete';
net.epoch          = 1;
net.v2hat_LL       = [0.5  1];
net.idx_LL_M       = 2;
% Parameter initialization
net.initParamType  = 'He';
net.gainS          = 0.25*ones(1, length(net.layer)-1);
net.gainS_v2hat    = 0.5;    %0.02/13
net.gainSb_v2hat   = 1;
net.gainM_v2hat    = 1;
% Maximal number of epochs and splits
net.maxEpoch       = 100;   
net.numSplits      = 1;

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
theta    = tagi.initializeWeightBias(net);
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
sv_LL = zeros(maxEpoch+1,2);
sv_LL(1,:) = net.v2hat_LL;
while ~stop
    epoch = epoch + 1;
    [theta, normStat,~,~,sv, ~, ~, ~, svresult_LL] = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
    sv_LL(epoch+1,:) = svresult_LL;
    % testing
    netT.v2hat_LL = svresult_LL;
    [~, ~, zl, Szl,~, Prior_act_v2hat] = network.regression(netT, theta, normStatT, statesT, maxIdxT, xtest, []);
    mv2a = Prior_act_v2hat(:,1);
    ml   = zl(:,1);
    Sl   = Szl(:,1) + mv2a;

% Figure
subplot(1,2,1)
Pr_msv = Prior_act_v2hat(:,1);
Pr_psv = sqrt(Prior_act_v2hat(:,2));
plot(xtest,Prior_act_v2hat(:,1),'k');hold on;  %plot the line first
patch([xtest' fliplr(xtest')],[Pr_msv'+Pr_psv',fliplr(Pr_msv'-Pr_psv')],'g','EdgeColor','none','FaceColor','g','FaceAlpha',0.2);hold on;
scatter(xtest,noiseTest,'dr');hold on;
xlabel('x');
ylabel('var (Prior)')
title(['Epoch',num2str(epoch)''])
hold off
subplot(1,2,2)
scatter(xtest,ml,'k');hold on;
scatter(xtest,ytest,'db');hold on;
scatter(xtest,ytestTrue,'r');
xlabel('x');
ylabel('mean')
title(['Epoch',num2str(epoch)''])
drawnow
hold off
pause(0.01)
% Evaluation
RMSE(epoch)          = mt.computeError(ytest, ml);
LLlist(epoch)        = mt.loglik(ytest, ml, Sl);

disp(' ')
disp(['      RMSE : ' num2str(RMSE(epoch)) ])
disp(['      LL   : ' num2str(LLlist(epoch))])
    if epoch >= maxEpoch; break;end
end
figure;
xp=(0:size(sv_LL,1)-1)';
mv=sv_LL(:,1);
sv2=sv_LL(:,2);
plot(xp,noise_hom*ones(length(xp),1),'r');hold on;
%mvt=sv_results(:,3);
%patch([xp' fliplr(xp')],sqrt([mv' + 3*sqrt(sv2') fliplr(mv' - 3*sqrt(sv2'))]./([mvt',fliplr(mvt')])),'red','EdgeColor','none','FaceColor','blue','FaceAlpha',0.2)
patch([xp' fliplr(xp')],[mv' + sqrt(sv2') fliplr(mv' - sqrt(sv2'))],'red','EdgeColor','none','FaceColor','blue','FaceAlpha',0.2);
hold on;
plot(xp,mv,'b');hold on
%     plot(xp,noise_sv^2,'r');
xlabel('epoch')
ylabel('sv')

