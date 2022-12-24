%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         toyExample with two networks
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
ntrain     = 4000;
ntest      = 1000;
dim        = 10;

noise      = @(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) 0.01*(x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+0.2).^3+0.1;                   %0.1*(x)+0.2, 0.45.*(x+0.5).^2+0.02, 0.018(x1+x2+x3+0.2).^3+0.1
noise_LL   = 0.2;
xtrain     = unifrnd(-1,1,[ntrain,dim]); % Generate covariate between [-1, 1];
noiseTrain = noise(xtrain(:,1),xtrain(:,2),xtrain(:,3),xtrain(:,4),xtrain(:,5),xtrain(:,6),xtrain(:,7),xtrain(:,8),xtrain(:,9),xtrain(:,10))+noise_LL;
if any(noiseTrain<0)
    noiseTrain(noiseTrain<0) = 0.01;
end
xtest      = unifrnd(-1,1,[ntest,dim]);
noiseTest  = noise(xtest(:,1),xtest(:,2),xtest(:,3),xtest(:,4),xtest(:,5),xtest(:,6),xtest(:,7),xtest(:,8),xtest(:,9),xtest(:,10))+noise_LL;
if any(noiseTest<0)
    noiseTrain(noiseTest<0) = 0.01;
end

ytrainTrue = 0;
ytrain     = ytrainTrue + normrnd(zeros(length(xtrain), 1), sqrt(noiseTrain));


ytestTrue  = 0;
ytest      = ytestTrue + normrnd(zeros(length(xtest), 1), sqrt(noiseTest));
nx         = size(xtrain, 2);
ny         = size(ytrain, 2);

subplot(1,1,1)
scatter(xtrain(:,1),noiseTrain,'r');
%% Noise Net
% GPU 1: yes; 0: no
net.task            = 'regression';
net.saveModel       = 1;
% GPU
net.gpu             = 0;
net.numDevices      = 0;
% Data type object half or double precision
net.dtype           = 'single';
% Number of input covariates
net.nx              = nx; 
% Number of output responses
net.nl              = ny;
net.nv2             = ny;
net.ny              = ny; 
% Batch size 
net.batchSize      = 1; 
net.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = [0 0 0];
net.layer          = [1              1        1         ];
net.nodes          = [net.nx        100      net.ny     ]; 
net.actFunIdx      = [0              4        0         ];
net.NoiseActFunIdx = 4;
net.NoiseActFun_LL_Idx = 4;%BD
net.idx_LL_M       = 1;    %BD
% Copy these terms
net.v2hat_LL       = [1  1];
net.epoch           = 1;
% Observations standard deviation
net.learnSv         = 0;% Online noise learning
net.sv              = 0;%0.32*ones(1,1, netN.dtype);  
net.lastLayerUpdate = 0;
net.noiseType       = 'hete';      %'hete' 
net.var_mode        = 1;
% Parameter initialization
net.initParamType  = 'He';      %'Xavier', 'He'
net.gainS          = 0.25*ones(1, length(net.layer)-1);     % Y = X:  G = 0.05, Epoch = 75 , other: 1.75 for original func 
net.gainS_v2hat    = 0.5;    %0.02/13
net.gainSb_v2hat   = 1;
net.gainM_v2hat    = 1;
% Maximal number of epochs and splits
net.maxEpoch       = 1;   
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
[net, states, maxIdx, netinfo] = network.initialization(net);

% Test net
net_test              = net;
net_test.trainMode    = 0;
net_test.batchSize    = 1;
net_test.repBatchSize = 1;
[net_test, statesTest, maxIdx_statesTest] = network.initialization(net_test);
normStat_test = tagi.createInitNormStat(net_test);

% Initalize weights and bias
theta     = tagi.initializeWeightBias(net);
normStat  = tagi.createInitNormStat(net);
net.sv    = svinit;

% metrics
RMSE        = zeros(net.maxEpoch,1);
LLlist      = zeros(net.maxEpoch,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
rumtime_s = tic;
stop  = 0;
epoch = 0;
% sv_result = zeros(maxEpoch+1,2);
% sv_result(1,:) = net.sv;
while ~stop
    epoch = epoch + 1;
    [theta, normStat,~,~,~,~,~,svresult_LL] = network.regressionOnlyNI(net, theta, normStat, states, maxIdx, xtrain, ytrain);
    sv_LL = [net.v2hat_LL;svresult_LL];
    %Testing
    net_test.v2hat_LL = svresult_LL(end,:);
    [~, ~, zl, Szl,Prior_act_v2hat] = network.regressionOnlyNI(net_test, theta, normStat_test, statesTest, maxIdx_statesTest, xtest, []);
    % Denormalizing
    mv2a = Prior_act_v2hat(:,1);
    ml   = zl(:,1);
    Sl   = Szl(:,1);
    % Plotting
    if dim == 1
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
        scatter(xtest,ytestTrue*ones(length(xtest),1),'r');
        xlabel('x');
        ylabel('mean')
        title(['Epoch',num2str(epoch)''])
        drawnow
        hold off
        pause(0.01)
    end
    % Evaluation
    RMSE(epoch)          = mt.computeError(ytest, zl);
    LLlist(epoch)        = mt.loglik(ytest, zl, Szl);
    
    disp(' ')
    disp(['      RMSE : ' num2str(RMSE(epoch)) ])
    disp(['      LL   : ' num2str(LLlist(epoch))])
    if epoch >= maxEpoch; break;end
end
runtime_e = toc(rumtime_s);
if ~isempty(net.idx_LL_M)
    figure;
    xp=(0:size(sv_LL,1)-1)';
    mv=sv_LL(:,1);
    sv2=sv_LL(:,2);
    plot(xp,noise_LL*ones(length(xp),1),'r');hold on;
    %mvt=sv_results(:,3);
    %patch([xp' fliplr(xp')],sqrt([mv' + 3*sqrt(sv2') fliplr(mv' - 3*sqrt(sv2'))]./([mvt',fliplr(mvt')])),'red','EdgeColor','none','FaceColor','blue','FaceAlpha',0.2)
    patch([xp' fliplr(xp')],[mv' + sqrt(sv2') fliplr(mv' - sqrt(sv2'))],'red','EdgeColor','none','FaceColor','blue','FaceAlpha',0.2);
    hold on;
    plot(xp,mv,'b');hold on
    %     plot(xp,noise_sv^2,'r');
    xlabel('obs')
    ylabel('sv')
end
plot(LLlist)

