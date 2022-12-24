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
% rand_seed=4;
% RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
%% Data
rng(1223) % Seed
ntrain     = 400;
ntest      = 100;
f          =@(x) (5*x).^3/50;
noise      =@(x) (3*(x).^4)+0.02;
xtrain     = [rand(ntrain, 1)*1 - 0.5]; % Generate covariate between [-1, 1];
wtrain     = normrnd(zeros(length(xtrain), 1), 0.1);
xtest      = linspace(-1, 1, ntest)';
noiseTrain = noise(xtrain);
ytrainTrue = f(xtrain);
ytrain     = f(xtrain) + normrnd(zeros(length(noiseTrain), 1), sqrt(noiseTrain));
wtest      = normrnd(zeros(length(xtest), 1), 0.1);
noiseTest  = noise(xtest);
ytestTrue  = f(xtest);
ytest      = ytestTrue + normrnd(zeros(length(noiseTest), 1), sqrt(noiseTest));
nx         = size(xtrain, 2);
ny         = size(ytrain, 2);

scatter(ytrain,noiseTrain,'ok');
xlabel('x');
ylabel('y');
% Data Loading
% load('Data_Hetero_BNI.mat')
% ntrain     = 400;
% ntest      = 100;
% nx         = size(xtrain, 2);
% ny         = size(ytrain, 2);

%% Net
% GPU 1: yes; 0: no
netD.task           = 'regression';
netD.saveModel      = 1;
% GPU
netD.gpu            = 0;
netD.numDevices     = 0;
% Data type object half or double precision
netD.dtype          = 'single';
% Number of input covariates
netD.nx             = nx; 
% Number of output responses
netD.nl             = ny;
netD.nv2            = ny;
netD.ny             = ny;             % 2 for 'hete'
% Batch size 
netD.batchSize      = 1; 
netD.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
netD.imgSize        = [0 0 0];
netD.layer          = [1          1       1       1    ];
netD.nodes          = [netD.nx   125     125    netD.ny]; 
netD.actFunIdx      = [0          4       4       0    ];
% Observations standard deviation
netD.learnSv        = 1;% Online noise learning
netD.sv             = [0.1  0.1];%0.32*ones(1,1, net.dtype);               %BD
netD.noiseType      = 'homo';      %'homo' or 'hete'                         %BD
netD.NoiseActFunIdx = 2;
% Parameter initialization
netD.initParamType  = 'He';
netD.gainS          = 0.25*ones(1, length(netD.layer)-1);
% Maximal number of epochs and splits
netD.maxEpoch       = 50;   
netD.numSplits      = 1;

%% Noise Net
% GPU 1: yes; 0: no
netN.task           = 'regression';
netN.saveModel      = 1;
% GPU
netN.gpu            = 0;
netN.numDevices     = 0;
% Data type object half or double precision
netN.dtype          = 'single';
% Number of input covariates
netN.nx             = netD.nodes(1); 
% Number of output responses
netN.nl             = ny;
netN.nv2            = ny;
netN.ny             = ny; 
% Batch size 
netN.batchSize      = netD.batchSize; 
netN.repBatchSize   = netD.repBatchSize; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
netN.imgSize        = [0 0 0];
netN.layer          = [1         1      1    ];
netN.nodes          = [netN.nx  200   netN.ny]; 
netN.actFunIdx      = [0         4      0    ];
netN.NoiseActFunIdx = 3;
% Observations standard deviation
netN.learnSv         = 0;% Online noise learning
netN.sv              = 0;%0.32*ones(1,1, netN.dtype);  
netN.lastLayerUpdate = 0;
netN.noiseType       = 'hete';      %'hete' 
% Parameter initialization
netN.initParamType  = 'He';
netN.gainS          = 2*ones(1, length(netN.layer)-1);
% Maximal number of epochs and splits
netN.maxEpoch       = 50;   
netN.numSplits      = 1;
%% Run
% Initialization          
saveModel  = netD.saveModel;
maxEpoch   = netD.maxEpoch;
svinit     = netD.sv;
layerConct = 1;
%layerConct = length(netD.layer) - 1;

% Train net
netD.trainMode = 1;
[netD, statesD, maxIdxD, netDinfo] = network.initialization(netD);
netN.trainMode = 1;
[netN, statesN, maxIdxN, netNinfo] = network.initialization(netN);

% Test net
netD1              = netD;
netD1.trainMode    = 0;
netD1.batchSize    = 1;
netD1.repBatchSize = 1;
[netD1, statesD1, maxIdxD1] = network.initialization(netD1);
normStatD1 = tagi.createInitNormStat(netD1);

netN1              = netN;
netN1.trainMode    = 0;
netN1.batchSize    = 1;
netN1.repBatchSize = 1;
[netN1, statesN1, maxIdxN1] = network.initialization(netN1);
normStatN1 = tagi.createInitNormStat(netN1);

% Initalize weights and bias
thetaD    = tagi.initializeWeightBias(netD);
normStatD = tagi.createInitNormStat(netD);
thetaN    = tagi.initializeWeightBias(netN);
normStatN = tagi.createInitNormStat(netN);
netD.sv    = svinit;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
rumtime_s = tic;
stop  = 0;
epoch = 0;
while ~stop
    epoch = epoch + 1;
    netD.sv = svinit;
    [thetaD, normStatD,Yn,Syn,sv] = network.regression(netD, thetaD, normStatD, statesD, maxIdxD, xtrain, ytrain);
%     %figure
%     subplot(1,1,1)
%     scatter(xtrain,ytrainTrue,'r');hold on
%     scatter(xtrain,Yn,'ok');hold on
%     scatter(xtrain,ytrain,'dm');
%     xlabel('x');
%     ylabel('y')
%     title(['Epoch',num2str(epoch)''])
%     drawnow
%     hold off
%     netD.sv = sv;
    if epoch >= maxEpoch; break;end
end
runtime_e = toc(rumtime_s);
% Training for Noise 
maxEpoch   = netN.maxEpoch;
runtime_s2 = tic;
stop   = 0;
epoch  = 0;
while ~stop
    epoch = epoch + 1;
    [thetaN, normStatN, ~, ~, Prior_act_v2hat, Pos_act_v2hat] = network.regressionWithNI(netD, thetaD, normStatD, statesD, maxIdxD, netN, thetaN, normStatN, statesN, maxIdxN, xtrain, ytrain, layerConct);
    % Figure
    subplot(1,2,1)
    xpr=(1:length(xtrain))';
    Pos_msv = Pos_act_v2hat(:,1);
    Pos_psv = sqrt(Pos_act_v2hat(:,1));
    %patch([xtrain' fliplr(xtrain')],[Pos_msv'+Pos_psv',fliplr(Pos_msv'-Pos_psv')],'g','EdgeColor','none','FaceColor','g','FaceAlpha',0.2);hold on
    scatter(xtrain,Pos_act_v2hat(:,1),'ok');hold on
    scatter(xtrain,noiseTrain,'dr');
    xlabel('x');
    ylabel('var (Pos)')
    title(['Epoch',num2str(epoch)''])
    hold off
    subplot(1,2,2)
    Pr_msv = Prior_act_v2hat(:,1);
    Pr_psv = sqrt(Prior_act_v2hat(:,1));
    %patch([xpr' fliplr(xpr')],[Pr_msv'+Pr_psv',fliplr(Pr_msv'-Pr_psv')],'g','EdgeColor','none','FaceColor','g','FaceAlpha',0.2);hold on
    scatter(xtrain,Prior_act_v2hat(:,1),'ok');hold on
    scatter(xtrain,noiseTrain,'dr');
    xlabel('x');
    ylabel('var (Prior)')
    title(['Epoch',num2str(epoch)''])
    drawnow
    hold off
    if epoch >= maxEpoch; break;end
end
runtime_e2 = toc(runtime_s2);
runtime_e  = runtime_e + runtime_e2;
% Figure
%     subplot(1,2,1)
%     xpr=(1:length(xtrain))';
%     Pos_msv = Pos_act_v2hat(:,1);
%     Pos_psv = sqrt(Pos_act_v2hat(:,1));
%     %patch([xtrain' fliplr(xtrain')],[Pos_msv'+Pos_psv',fliplr(Pos_msv'-Pos_psv')],'g','EdgeColor','none','FaceColor','g','FaceAlpha',0.2);hold on
%     scatter(xtrain,Pos_act_v2hat(:,1),'ok');hold on
%     scatter(xtrain,noiseTrain,'dr');
%     xlabel('x');
%     ylabel('var (Pos)')
%     title(['Epoch',num2str(epoch)''])
%     hold off
%     subplot(1,2,2)
%     Pr_msv = Prior_act_v2hat(:,1);
%     Pr_psv = sqrt(Prior_act_v2hat(:,1));
%     %patch([xpr' fliplr(xpr')],[Pr_msv'+Pr_psv',fliplr(Pr_msv'-Pr_psv')],'g','EdgeColor','none','FaceColor','g','FaceAlpha',0.2);hold on
%     scatter(xtrain,Prior_act_v2hat(:,1),'ok');hold on
%     scatter(xtrain,noiseTrain,'dr');
%     xlabel('x');
%     ylabel('var (Prior)')
%     title(['Epoch',num2str(epoch)''])
%     drawnow
%     hold off
    


%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing
nObsTest = size(xtest, 1);
[~, ~, yp, Syp] = network.regression(netT, theta, normStatT, statesT, maxIdxT, xtest, []);
if netD.learnSv==1&&strcmp(netD.noiseType, 'hete') % Online noise inference
    ypM  = reshape(yp', [netD.nl, 1, 2, nObsTest]);
    SypM = reshape(Syp', [netD.nl, 1, 2, nObsTest]);
    
    mv2  = reshape(ypM(:,:,2,:), [netD.nl*nObsTest, 1]);
    Sv2  = reshape(SypM(:,:,2,:), [netD.nl*nObsTest, 1]);
    mv2a = act.NoiseActFun(mv2, Sv2, netD.NoiseActFunIdx, netD.gpu);
    
    ml   = reshape(ypM(:,:,1,:), [netD.nl*nObsTest, 1]);
    Sl   = reshape(SypM(:,:,1,:), [netD.nl*nObsTest, 1]) + mv2a;
else
    ml = yp;
    if strcmp(netD.noiseType, 'homo')
        Sl = Syp+netD.sv(1);
    else
        Sl = Syp+netD.sv(1).^2;
    end
end

%%  Plotting Data
figure;
plot(xtest,ytestTrue,'r');hold on
patch([xtest' fliplr(xtest')],[ytestTrue' + sqrt(noiseTest') fliplr(ytestTrue' - sqrt(noiseTest'))],'red','EdgeColor','none','FaceColor','red','FaceAlpha',0.2);hold on
scatter(xtrain,ytrain,'+b');hold on
scatter(xtest,ytest,'dm');hold on
plot(xtest,ml,'k');hold on
patch([xtest' fliplr(xtest')],[ml' + sqrt(Sl') fliplr(ml' - sqrt(Sl'))],'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.2)
h=legend('$y_{true}$','$y_{true} \pm \sigma_{true}$','train','test','$E[\hat{y}]$','$E[\hat{y}] \pm 1\sigma$');
set(h,'Interpreter','latex')
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')



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


