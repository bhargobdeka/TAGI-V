%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         toyExample with one network
% Description:  Apply TAGI to heteroscedasticity
% Authors:      Bhargob Deka & Luong-Ha Nguyen & James-A. Goulet
% Created:      Aug 18, 2020
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
%rng('default')
rand_seed=1;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed)); %Initialize random stream number based on clock
RMSE = zeros(10,1);
LL   = zeros(10,1);
path  = char([cd ,'/Indices/']);
load(char([path, '/DepewegTestIndices.mat']))
load(char([path, '/DepewegTrainIndices.mat']))
for i = 1:10
    %% Load Indices
    path1  = char([cd ,'/Gains/']);
    load(char([path1, '/Gains', num2str(i) '.mat']))
    path2  = char([cd ,'/opt_Epochs/']);
    load(char([path2, '/opt_Epochs', num2str(i) '.mat']))
    %% Data
    path  = char([cd ,'/Datasets/']);
    load(char([path, '/dataset', num2str(i) '.mat']))
    x_obs     = Data.x_obs;
    y_obs     = Data.y_obs;
    nx        = size(Data.x_obs, 2);
    ny        = size(Data.y_obs, 2);

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
    net.gainS          = Gains(1)*ones(1, length(net.layer)-1);
    net.gainS_v2hat    = Gains(2);     %0.3 best for xavier and 0.15 best for He
    net.gainSb_v2hat   = 1;
    net.gainM_v2hat    = 1;
    % splits
    net.numSplits      = 5;
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
    RMSElist      = zeros(Nsplit,1);
    LLlist        = zeros(Nsplit,1);
    trainTimelist = zeros(Nsplit,1);
    for s = 1:Nsplit
        disp(['   Split : ' num2str(s)])
        % Data
        if net.permuteData == 1
            [xtrain, ytrain, xtest, ytest] = dp.split(x, y, 0.9);
        else
            xtrain = x_obs(trainIdx{s}, :);
            ytrain = y_obs(trainIdx{s}, :);
            xtest  = x_obs(testIdx{s}, :);
            ytest  = y_obs(testIdx{s}, :);
        end
        % scatter(xtrain,ytrain)

        %% Run
        % Initialization
        saveModel  = net.saveModel;
        maxEpoch   = opt_Epochs(s);   
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
        % Early Stopping for MaxEpochs
        if net.cv_HP == 1
            maxEpoch          = opt.crossvalHP(net, xtrain, ytrain);
            disp([' No. of Epoch ' num2str(maxEpoch)])
        end
        % Initalize weights and bias
        theta = tagi.initializeWeightBias(net);

        normStat = tagi.createInitNormStat(net);
        net.sv   = svinit;
        % Normalize Data
        [xtrainN, ytrainN, xtestN, ~, ~, ~, mytrain, sytrain] = dp.normalize(xtrain, ytrain, xtest, ytest);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Training
        rumtime_s = tic;
        stop  = 0;
        epoch = 0;

        while ~stop
            epoch = epoch + 1;
            disp(['Epoch: ' num2str(epoch)])
            net.epoch   = epoch;      %BD
            net.epochCV = epoch;
            %     [theta, normStat,zl,Szl,sv, Prior_act_v2hat, Pos_act_v2hat, K_Gain] = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
            [theta, normStat] = network.regression(net, theta, normStat, states, maxIdx, xtrainN, ytrainN);
            % % Testing
            %     [~, ~, zl, Szl,~, Prior_act_v2hat] = network.regression(netT, theta, normStatT, statesT, maxIdxT, xtestN, []);
            % % Denormalizing
            % mv2a = Prior_act_v2hat(:,1);
            % ml   = zl(:,1);
            % Sl   = Szl(:,1);
            %
            % [ml, Sl] = dp.denormalize(ml, Sl, mytrain, sytrain);
            %
            % subplot(1,1,1)
            % Pzl = sqrt(Sl);
            % scatter(xtestN,ml,'k');hold on;
            % patch([xtestN' fliplr(xtestN')],[ml'+Pzl',fliplr(ml'-Pzl')],'g','EdgeColor','none','FaceColor','g','FaceAlpha',0.2);hold on;
            % scatter(xtestN,ytest,'db');hold on;
            % xlabel('x');
            % ylabel('mean')
            % title(['Epoch',num2str(epoch)''])
            % drawnow
            % hold off
            if epoch >= maxEpoch; break;end
        end
        runtime_e = toc(rumtime_s);
        % Testing
        nObsTest = size(xtest, 1);
        [~, ~, yp, Syp] = network.regression(netT, theta, normStatT, statesT, maxIdxT, xtestN, []);
        [yp, Syp]       = dp.denormalize(yp, Syp, mytrain, sytrain);
        ypM             = reshape(yp', [net.nl, 1, 2, nObsTest]);
        SypM            = reshape(Syp', [net.nl, 1, 2, nObsTest]);
        ml              = reshape(ypM(:,:,1,:), [net.nl*nObsTest, 1]);
        Sl              = reshape(SypM(:,:,1,:), [net.nl*nObsTest, 1]);
        % Evaluation
        RMSElist(s)     = mt.computeError(ytest, ml);
        LLlist(s)       = mt.loglik(ytest, ml, Sl);
        trainTimelist(s) = runtime_e; 
        disp(' Results each split ')
        disp(['      RMSE : ' num2str(RMSElist(s)) ])
        disp(['      LL   : ' num2str(LLlist(s))])
    end
    RMSE(i) = nanmean(RMSElist);
    LL(i)   = nanmean(LLlist);
    disp(' Results each run')
    disp(['  Avg. RMSE     : ' num2str(nanmean(RMSElist)) ' +- ' num2str(nanstd(RMSElist))])
    disp(['  Avg. LL       : ' num2str(nanmean(LLlist)) ' +- ' num2str(nanstd(LLlist))])
end
avg_RMSE_per_run = nanmean(RMSE);
avg_LL_per_run = nanmean(LL);
disp(' Final avg. results after 10 random run')
disp(['  Avg. RMSE per run    : ' num2str(nanmean(RMSE)) ' +- ' num2str(nanstd(RMSE))])
disp(['  Avg. LL per run       : ' num2str(nanmean(LL)) ' +- ' num2str(nanstd(LL))])