%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         opt
% Description:  Optimize hyperparameters  for TAGI
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      December 9, 2019
% Updated:      April 15, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef opt
    methods (Static)
        function [optEpoch, optMtrainlist, optMvallist] = chpTuning(NN, mp, Sp, xtrain, ytrain, trainLabels, trainEncoderIdx)
            NNval            = NN;
            NNval.trainMode  = 0;
            [xtrainHp, ytrainHp, trainHpLabels, trainHpEncoderIdx] = dp.selectData(xtrain, ytrain, trainLabels, trainEncoderIdx, NN.trainIdx);
            [xvalHp, yvalHp, valHpLabels, ~] = dp.selectData(xtrain, ytrain, trainLabels, trainEncoderIdx, NN.valIdx);
            NN.labels        = trainHpLabels;
            NN.encoderIdx    = trainHpEncoderIdx;
            NNval.labels     = valHpLabels;
            [optEpoch, optMtrainlist, optMvallist] = opt.numEpochs(NN.maxEpoch, NN.sv, NN, NNval, mp, Sp, xtrainHp, ytrainHp, xvalHp, yvalHp);
        end
        function [bestmp, bestSp, Mvallist, optEpoch] = earlyStop(NN, mp, Sp, x, y)
            % Initializtion
            NNval           = NN;
            NNval.trainMode = 0;
            % Data
            trainIdx        = NN.trainIdx;
            valIdx          = NN.valIdx;
            if strcmp(NN.task, 'classification')
                [xtrain, ytrain, trainLabels, trainEncoderIdx] = dp.selectData(x, y, NN.labels, NN.encoderIdx, trainIdx);
                [xval, ~, valLabels, ~] = dp.selectData(x, y, NN.labels, NN.encoderIdx, valIdx);
                NN.encoderIdx = trainEncoderIdx;
                NN.labels     = trainLabels;
                NNval.labels  = valLabels;  
            elseif strcmp(NN.task, 'regression')
                [xtrain, ytrain, ~, ~] = dp.selectData(x, y, [], [], trainIdx);
                [xval, yval, ~, ~]     = dp.selectData(x, y, [], [], valIdx);
            end
            % Loop
            if NN.gpu == 1
                Mvallist = zeros(NN.numEpochs, 1, 'gpuArray');
            else
                Mvallist = zeros(NN.numEpochs, 1, NN.dtype);
            end
            stop     = 0;
            epoch    = 0;
            bestMval = 1;
            bestmp   = mp;
            bestSp   = Sp;
            optEpoch = 0;
            timeTot  = 0;
            while ~stop
                ts    = tic;
                epoch = epoch + 1;
                % Shuffle data
                if epoch >= 1
                    idxtrain      = randperm(size(ytrain, 1));
                    ytrain        = ytrain(idxtrain, :);
                    xtrain        = xtrain(idxtrain, :);
                    NN.encoderIdx = NN.encoderIdx(idxtrain, :);
                    NN.labels     = NN.labels(idxtrain);                    
                end 
                disp('#########################')
                disp(['Epoch #' num2str(epoch)])
                if strcmp(NN.task, 'classification')
                    NN.errorRateEval    = 1;
                    NNval.errorRateEval = 1;
                    [~, Mval, mp, Sp]   = opt.computeErrorRate(NN, NNval, mp, Sp,  xtrain, ytrain, xval); 
                elseif strcmp(NN.task, 'regression')
                    [~, Mval, mp, Sp, ~, ~] = opt.computeMetric(NN.sv, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval);  
                end
                Mvallist(epoch) = Mval;
                if Mval < bestMval
                    bestMval = Mval;
                    bestmp   = mp;
                    bestSp   = Sp;
                    optEpoch = epoch;
                end
                if epoch >= NN.numEpochs
                    stop = 1;
                end
                timeTot = timeTot + toc(ts);
                timeRem = timeTot/epoch*(NN.numEpochs-epoch)/60;
                disp(' ')
                disp(['Remaining time (mins): ' sprintf('%0.2f',timeRem)]);
            end
        end
        function [sv, N]             = crossValidation(NN, NNval, mp, Sp, x, y, optNumEpoch, numRuns)          
            % Split data into different folds
            numFolds = NN.numFolds;
            numEpochs= NN.numEpochs;
            numObs   = size(x, 1);
            initsv   = NN.sv;
            foldIdx  = dp.kfolds(numObs, numFolds);
            if strcmp(NN.task, 'classification')
                labels     = NN.labels;
                encoderIdx = NN.encoderIdx;
            end
            % Loop initialization
            if NN.gpu == 1
                 Nlist  = zeros(numRuns, 1, 'gpuArray');
                svlist  = zeros(numRuns, length(NN.sv), 'gpuArray');
            else
                Nlist  = zeros(numRuns, 1, NN.dtype);
                svlist = zeros(numRuns, length(NN.sv), NN.dtype);
            end                      
            dispFlag = 1;
            for n = 1:numRuns
                NN.sv    = initsv;
                if dispFlag == 1
                    disp('   ------')
                    disp(['   Fold #' num2str(n)])
                end
                [xtrain, xval]       = dp.regroup(x, foldIdx, n);
                [ytrain, yval]       = dp.regroup(y, foldIdx, n);
                if strcmp(NN.task, 'classification')
                    [trainLabels, valLabels] = dp.regroup(labels, foldIdx, n);
                    [trainEncoderIdx, valEncoderIdx] = dp.regroup(encoderIdx, foldIdx, n);
                    NN.labels        = trainLabels;
                    NNval.labels     = valLabels;
                    NN.encoderIdx    = trainEncoderIdx;
                    NNval.encoderIdx = valEncoderIdx;
                end
                [svlist(n, :), ~, ~] = opt.BGA(initsv, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval);
                if optNumEpoch == 1
                    Nlist(n)         = opt.numEpochs(numEpochs, svlist(n, :), NN, NNval, mp, Sp, xtrain, ytrain, xval, yval);
                end
                if dispFlag == 1
                    fprintf(['   sigma_v      : ' repmat(['%#-+10.2e' ' '],[1, length(NN.sv)-1]) '%#-+10.2e\n'], svlist(n, :))
                    if optNumEpoch == 1
                        disp(['   Num. of epoch: ' num2str(Nlist(n))])
                    end
                end
            end
            sv = mean(svlist);
            N  = round(mean(Nlist));
        end
        function [N, Mtrainlist, Mvallist] = numEpochs(E, sv, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval)
            if NN.gpu == 1
                Mvallist = zeros(E, 1, 'gpuArray');
                Mtrainlist = zeros(E, 1, 'gpuArray');
            else
                Mvallist = zeros(E, 1, NN.dtype);
                Mtrainlist = zeros(E, 1, 'gpuArray');
            end
            timeTot  = 0;
            for e = 1:E
                ts = tic;
                if e > 1
                    idxtrain = randperm(size(ytrain, 1));
                    ytrain   = ytrain(idxtrain, :);
                    xtrain   = xtrain(idxtrain, :);
                    NN.encoderIdx = NN.encoderIdx(idxtrain, :);
                    NN.labels = NN.labels(idxtrain);                  
                end                
                if strcmp(NN.task, 'classification')
                    if NN.displayMode == 1
                        disp('   #########################')
                        disp(['   Epoch #' num2str(e)])
                    end
                    NNval.errorRateEval = 1;
                    [Mtrainlist(e), Mvallist(e), mp, Sp] = opt.computeErrorRate(NN, NNval, mp, Sp, xtrain, ytrain, xval);
                    timeTot = timeTot + toc(ts);
                    timeRem = timeTot/e*(E-e)/60;
                    if NN.displayMode == 1
                        disp(' ')
                        disp(['   Remaining time (mins): ' sprintf('%0.2f',timeRem)]);
                    end
                    [~, N] = min(Mvallist); 
                else
                    [~,  Mvallist(e), mp, Sp, ~, ~] = opt.computeMetric(sv, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval); 
                    [~, N] = max(Mvallist); 
                end
            end                   
        end
        function [thetaOpt, mp, Sp]  = BGA(theta, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval)
            % NR: Newton-Raphson; MMT: Momentum; ADAM: Adaptive moment
            % estimation
            optimizer           = 'ADAM'; 
            % Hyperparameters
            N                   = 50;
            displayFlag         = 0;
            learningRate        = 0.2;
            beta                = 0.9;
            beta1               = 0.9;
            beta2               = 0.999;
            epsilon             = 1E-8;
            tol                 = 1E-6;
            % Parameter setup
            numParams           = length(theta);
            idxParam            = 1*ones(numParams, 1);
            if NN.gpu == 1
                theta           = gpuArray(theta);
                thetaloop       = theta;
                thetaTR         = zeros(numParams, 1, 'gpuArray');
                momentumTR      = zeros(numParams, 1, 'gpuArray');
                vTR             = zeros(numParams, 1, 'gpuArray');
                sTR             = zeros(numParams, 1, 'gpuArray');
                gradientTR2OR   = zeros(numParams, 1, 'gpuArray');
                Mlist           = zeros(N+1, 1, 'gpuArray');
                Mvallist        = zeros(N+1, 1, 'gpuArray');
                thetalist       = zeros(N+1, numParams, 'gpuArray');
            else
                thetaloop       = theta;
                thetaTR         = zeros(numParams, 1);
                momentumTR      = zeros(numParams, 1);
                vTR             = zeros(numParams, 1);
                sTR             = zeros(numParams, 1);
                gradientTR2OR   = zeros(numParams, 1);
                Mlist           = zeros(N+1, 1);
                Mvallist        = zeros(N+1, 1);
                thetalist       = zeros(N+1, numParams);
            end
            funOR2TR            = cell(numParams, 1);
            funTR2OR            = cell(numParams, 1);
            funGradTR2OR        = cell(numParams, 1);
            for n = 1: numParams
                [funOR2TR{n}, funTR2OR{n}, funGradTR2OR{n}] = opt.getTransfFun(idxParam(n));
                thetaTR(n)  = funOR2TR{n}(theta(n));
            end
            % Compute inital metric
            NN.sv               = theta;
            NNval.sv            = theta;
            [M, Mval, mp, Sp, yf, Vf] = opt.computeMetric(theta, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval);                      
            % Loop initialization
            converged           = 0;
            loop                = 0;
            count               = 0;
            evalGradient        = 1;           
            Mlist(1)            = M;
            Mvallist(1)         = Mval;
            thetalist(1, :)     = theta;
            if displayFlag == 1
                figure
%                 plot(Mlist(1), 'ok')
                hold on
                plot(Mvallist(1), 'om')
                xlabel('Num. of epoch', 'interpreter', 'latex')
                ylabel('Log-likelihood')
                xlim([1, N])
            end
            while ~converged
                loop     = loop + 1;
                if displayFlag == 1
                    disp(' ')
                    disp(['   Iteration #' num2str(loop)])
                end
                thetaRef = theta;
                % Compute gradient
                if evalGradient == 1
                    for n = 1:numParams
                        gradientTR2OR(n) = funGradTR2OR{n}(thetaTR(n));
                    end
                    if isfield(NN, 'encoderIdx')
                        ygrad = ytrain(NN.encoderIdx);
                    else
                        ygrad = ytrain;
                    end
                    [gradient, hessian]  = opt.computeGradient(ygrad, yf, Vf, theta);
                    gradientTR           = gradient.*gradientTR2OR;
                    hessianTR            = abs(hessian.*gradientTR2OR.^2);
                end
                % Update parameters
                if strcmp(optimizer, 'NR')
                    thetaTRloop                     = opt.NR(thetaTR, gradientTR, hessianTR);
                    momentumTRloop                  = nan;
                    vTRloop                         = nan;
                    sTRloop                         = nan;
                elseif strcmp(optimizer, 'MMT')
                    [thetaTRloop, momentumTRloop]   = opt.MMT(thetaTR, gradientTR, momentumTR, learningRate, beta);
                    vTRloop                         = nan;
                    sTRloop                         = nan;
                elseif strcmp(optimizer, 'ADAM')
                    [thetaTRloop, vTRloop, sTRloop] = opt.ADAM(thetaTR, sTR, vTR, gradientTR, learningRate, beta1, beta2, epsilon, loop);
                    momentumTRloop                  = nan;
                else
                    error ('The optimizer does not exist')
                end
                % Transform to original space
                for n = 1:numParams
                    thetaloop(n) = funTR2OR{n}(thetaTRloop(n));
                end
                % Compute metric w.r.t the new parameters
                [Mloop, Mvalloop, mploop, Sploop, yfloop, Vfloop] = opt.computeMetric(thetaloop, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval);
                % Update new parameter values for the next iteration 
                if Mloop > M
                    % Convergence check
                    if abs((M - Mloop) / M) < tol
                        converged = 1;
                    end
                    M           = Mloop;
                    Mval        = Mvalloop;
                    theta       = thetaloop;
                    thetaTR     = thetaTRloop;
                    mp          = mploop;
                    Sp          = Sploop;
                    momentumTR  = momentumTRloop;
                    vTR         = vTRloop;
                    sTR         = sTRloop;
                    yf          = yfloop;
                    Vf          = Vfloop;
                else
                    learningRate = learningRate/2; 
                    count        = count + 1;
                    evalGradient = 0;
                end
                % Savel to list
                Mlist(loop+1)         = M;
                Mvallist(loop+1)      = Mval;
                thetalist(loop+1, : ) = theta;
                % Convergence check
                if loop == N || count > 3 || converged == 1
                    [~, idx] = max(Mvallist(1:loop+1));  
                    thetaOpt = thetalist(idx, :);
                    break                   
                end
                % Display the results
                if displayFlag == 1
                    disp(['    Log likelihood: ' num2str(M)])
                    fprintf(['    current values: ' repmat(['%#-+15.2e' ' '],[1, numParams-1]) '%#-+15.2e\n',...
                        '      param change: ' repmat(['%#-+15.2e' ' '],[1, numParams-1]) '%#-+15.2e\n'], theta, theta-thetaRef)                   
%                     plot(loop+1, Mlist(loop+1), 'ok')
                    hold on
                    plot(loop+1, Mvallist(loop+1), 'om')
                    pause(0.01)
                end                
            end
        end      
        function [g, h]              = computeGradient(y, yf, Vf, sv)   
            d      = size(y, 2);
            sv     = sv.*ones(size(Vf));
            sigma  = Vf + sv.^2;
            B      = (y - yf).^2;
            if d == 1
                g  = mean(-sv./sigma + (sv./(sigma.^2)).*B);
                h  = mean(((sv.^2) - Vf)./(sigma.^2) + ((-3*(sv.^4) - 2*(sv.^2).*Vf + (Vf.^2))./(sigma.^4)).*B);
            else
                g  = mean(sum(d*(-sv)./sigma + (sv./(sigma.^2)).*B, 2));
                h  = mean(sum(((d*sv.^2) - Vf)./(sigma.^2) + ((-3*(sv.^4) - 2*(sv.^2).*Vf + (Vf.^2))./(sigma.^4)).*B, 2));
            end
        end
        function [M, Mval, mp, Sp, yf, Vf] = computeMetric(theta, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval)
            NN.sv    = theta;
            NNval.sv = theta;
            % Training
            NN.trainMode            = 1;
            [mp, Sp, yf, Vf, ~, ~]     = tagi.network(NN, mp, Sp, xtrain, ytrain);
            if isfield(NN, 'encoderIdx')
                yf = yf(NN.encoderIdx);
                Vf = Vf(NN.encoderIdx);
                ytrain = ytrain(NN.encoderIdx);
            end
            Vf = Vf + (NN.sv.^2).*ones(size(Vf), NN.dtype);
            M  = mt.loglik(ytrain, yf, Vf);
            % Validation
            NNval.trainMode         = 0;
            [~, ~, yfval, Vfval, ~, ~] = tagi.network(NNval, mp, Sp, xval, []);
            if isfield(NNval, 'encoderIdx')
                yfval = yfval(NNval.encoderIdx);
                Vfval = Vfval(NNval.encoderIdx);
                yval  = yval(NNval.encoderIdx);
            end
            Vfval = Vfval + (NNval.sv.^2).*ones(size(Vfval));
            Mval  = mt.loglik(yval, yfval, Vfval);
        end
        function [M, Mval, mp, Sp]   = computeErrorRate(NN, NNval, mp, Sp,  xtrain, ytrain, xval)
            % Training
            NN.trainMode = 1;         
            [mp, Sp, ~, ~, ~, erTrain] = tagi.network(NN, mp, Sp, xtrain, ytrain);
            M = mean(erTrain);
            % Validation
            NNval.trainMode = 0;                   
            [~, ~, ~, ~, ~, erVal] = tagi.network(NNval, mp, Sp, xval, []);
            Mval = mean(erVal);
        end
        function theta               = NR(prevtheta, gradient, hessian)
            theta = prevtheta + gradient./hessian;
        end
        function [theta, momentum]   = MMT(prevtheta, gradient, prevMomentum, learningRate, beta)
            momentum = beta*prevMomentum + (1 - beta)*gradient;
            theta    = prevtheta + learningRate*momentum;
        end
        function [theta, v, s]       = ADAM(prevtheta, prevs, prevv, gradient, learningRate, beta1, beta2, epsilon, N)
            v       = beta2*prevv + (1 - beta2)*(gradient).^2;
            s       = beta1*prevs + (1 - beta1)*gradient;
            vhat    = v./(1-(beta2)^N);
            shat    = s./(1-(beta1)^N);
            theta   = prevtheta + learningRate*shat./(sqrt(vhat) + epsilon);
        end
        function [funOR2TR, funTR2OR, funGradTR2OR] = getTransfFun(idxParam)       
            if idxParam == 1  % loge
                transfOR2TR     = @(p) log(p);
                transfTR2OR     = @(p) exp(p);
                
                funOR2TR        = @(p) transfOR2TR(p);
                funTR2OR        = @(p) transfTR2OR(p);
                funGradTR2OR    = @(p) exp(p);                             
            elseif idxParam == 2  % log10
                transfOR2TR     = @(p) log10(p);
                transfTR2OR     = @(p) 10.^p;     
                
                funOR2TR        = @(p) transfOR2TR(p);
                funTR2OR        = @(p) transfTR2OR(p);
                funGradTR2OR    = @(p) log(10)*10.^p;                
            elseif idxParam == 3 % None
                transfOR2TR     = @(p) p;
                transfTR2OR     = @(p) p;
                
                funOR2TR        = @(p) transfOR2TR(p);
                funTR2OR        = @(p) transfTR2OR(p);
                funGradTR2OR    = @(p) 1;
            else
                error('Parameter transformation function are not properly defined in: config file')
            end
        end
        % Optimization for no of Epochs for v2hat_LL
        function NEpoch            = crossvalV2hatLL(net, x, y)
            % Split data into different folds
            numFolds    = net.numFolds;
            numObs      = size(x, 1);
            c           = cvpartition(numObs,'KFold',5);
%             foldIdx     = dp.kfolds(numObs, numFolds);
            permuteData = net.permuteData;
            ratio       = net.ratio;
            E           = zeros(numFolds, 1, net.dtype);
            
            dispFlag = 1;
            for n = 1:numFolds
                if dispFlag == 1
                    disp('   ------')
                    disp(['   Fold #' num2str(n)])
                end
                % Splitting data into train and val
                if permuteData == 1
                    [xtrain, ytrain, xval, yval] = dp.split(x, y, ratio);
                else
                [xtrain, ytrain, xval, yval] = dp.partition(x, y, c, n);
                end
                % Normalize data
                [xtrain, ytrain, xval, ~, ~, ~, mytrain, sytrain] = dp.normalize(xtrain, ytrain, xval, yval);
                %% Network
                % Train net
                net.trainMode = 1;
                [net, states, maxIdx] = network.initialization(net);
                maxEpoch = net.maxEpoch;
                % Val net
                netV              = net;
                netV.trainMode    = 0;
                netV.batchSize    = 1;
                netV.repBatchSize = 1;
                [netV, statesV, maxIdxV] = network.initialization(netV);
                normStatV = tagi.createInitNormStat(netV);
            
                % Initalize weights and bias
                theta    = tagi.initializeWeightBias(net);
                normStat = tagi.createInitNormStat(net);
                
                % Training
                stop     = 0;
                epoch    = 0;
                LL_val   = zeros(maxEpoch,1);
                while ~stop
                    if epoch >1
                        idxtrain = randperm(size(ytrain, 1));
                        ytrain   = ytrain(idxtrain, :);
                        xtrain   = xtrain(idxtrain, :);
                    end
                    epoch = epoch + 1;
                    net.epoch = epoch;      %BD
                    net.epochCV = epoch;
                    [theta, normStat, ~, ~, ~, ~, ~, ~, svresult_LL] = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
                    net.v2hat_LL = svresult_LL;
                    
                    % Validation
                    netV.v2hat_LL = svresult_LL;
                    [~, ~, mzval, Szval] = network.regression(netV, theta, normStatV, statesV, maxIdxV, xval, []);
                    [ml_val, Sl_val]     = dp.denormalize(mzval(:,1), Szval(:,1), mytrain, sytrain);
                    LL_val(epoch,:)      = mt.loglik(yval, ml_val, Sl_val);
                    if epoch >= maxEpoch; break;end
                end
                [~, N] = max(LL_val);
                E(n)   = N;
            end
            NEpoch  = round(mean(E));
        end
        % Optimization for no of Epochs in Heteroscedastic noise inference
        function NEpoch            = crossvalHP(net, x, y)
            % Split data into different folds
            numFolds    = net.numFolds;
            numObs      = size(x, 1);
%             foldIdx     = dp.kfolds(numObs, numFolds);
            c           = cvpartition(numObs,'KFold',5);
            permuteData = net.permuteData;
            ratio       = net.ratio;
            E           = zeros(numFolds, 1, net.dtype);
            idx = randperm(size(x,1));
            x   = x(idx,:);
            y   = y(idx);
            dispFlag = 1;
            for n = 1:numFolds
                if dispFlag == 1
                    disp('   ------')
                    disp(['   Fold #' num2str(n)])
                end
                
%                 if n == 2
%                     disp('fold 2');
%                 end
                % Splitting data into train and val
                if permuteData == 1
                    [xtrain, ytrain, xval, yval] = dp.split(x, y, ratio);
                else
                    [xtrain, ytrain, xval, yval] = dp.partition(x, y, c, n);
%                 [xtrain, xval]       = dp.regroup(x, foldIdx, n);
%                 [ytrain, yval]       = dp.regroup(y, foldIdx, n);
                end
                % Normalize data
                [xtrain, ytrain, xval, ~, ~, ~, mytrain, sytrain] = dp.normalize(xtrain, ytrain, xval, yval);
                %% Network
                % Train net
                net.trainMode = 1;
                [net, states, maxIdx] = network.initialization(net);
                maxEpoch = net.maxEpoch;
                % Val net
                netV              = net;
                netV.trainMode    = 0;
                netV.batchSize    = 1;
                netV.repBatchSize = 1;
                [netV, statesV, maxIdxV] = network.initialization(netV);
                normStatV = tagi.createInitNormStat(netV);
                % Test net
                netT              = net;
                netT.trainMode    = 0;
                netT.batchSize    = 1;
                netT.repBatchSize = 1;
                [netT, statesT, maxIdxT] = network.initialization(netT);
                normStatT = tagi.createInitNormStat(netT);
            
                % Initalize weights and bias
                if net.HP == 1
                    if net.HP_M == 3
                        theta    = tagi.initializeWeights_HP_BNI(net);
                    else
                        theta    = tagi.initializeWeightBias(net);
                    end
                else
                    theta    = tagi.initializeWeightBias(net);
                end
                normStat = tagi.createInitNormStat(net);
                
                % Training
                stop     = 0;
                epoch    = 0;
                LL_val   = zeros(maxEpoch,1);
                while ~stop
                    if epoch >1
                        idxtrain = randperm(size(ytrain, 1));
                        ytrain   = ytrain(idxtrain, :);
                        xtrain   = xtrain(idxtrain, :);
                    end
                    epoch = epoch + 1;
                    net.epoch = epoch;      %BD
                    net.epochCV = epoch;
                    [theta, normStat]    = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
                    % Validation
                    [~, ~, mzval, Szval] = network.regression(netV, theta, normStatV, statesV, maxIdxV, xval, []);
                    [ml_val, Sl_val]     = dp.denormalize(mzval(:,1), Szval(:,1), mytrain, sytrain);
                    LL_val(epoch,:)      = mt.loglik(yval, ml_val, Sl_val);
                    if epoch >= maxEpoch; break;end
                end
                % scatter plot
%                 plot(LL_val,'g');
                [~, N] = max(LL_val);
%                 if N < 10
%                     N1 = N + 20;
%                 elseif N < 20
%                     N1 = N + 10;
%                 elseif N < 30
%                     N1 = N + 5;
%                 else
%                     N1 = N;
%                 end
                 E(n) = N;
%                 E(n)   = N;
%                 i = N;
%                 Err = zeros(maxEpoch-N,1);
%                 while i < 100
%                     Err(i+1-N) = M - LL_val(i+1);
%                     i = i+1;
%                 end
%                 % Using moving average
%                 if length(Err) ~= 1
%                     span = round(length(Err)/3);
%                 else
%                     span = 1;
%                 end
%                 
%                 try
%                     mavg   = movmean(Err,span);
%                     [~,L]  = min(mavg);
%                 catch
%                     warning('Problem using mavg.  Assigning a value of 0.');
%                     L = 0;
%                 end
%                 [~,L]  = min(mavg);
%                 E(n)   = N+L;
            end
            NEpoch  = round(mean(E));
            disp(['   opt Epoch : ' num2str(NEpoch)])
        end
        function [Gain_factor, opt_Epoch ]   =  Gridsearch(net,x,y, trainIdx)
            % Initalize weights and bias
            alpha        = [0.1 0.5 1];     %0.1:0.1:1
            beta         = [1e-03 1e-02 1e-01 ];
            dict         = combvec(alpha, beta);
            avg_split_LL = zeros(1,size(dict,2));
            opt_epoch_k  = zeros(net.numSplits,size(dict,2));
                for k = 1:size(dict,2)
                    Gain = dict(:,k);
                    disp([' Search loop #' num2str(k)])
                    dispFlag = 0;
                    Nsplit        = net.numSplits; 
                    avg_fold_E    = zeros(Nsplit,1,net.dtype);
                    avg_fold_LL   = zeros(Nsplit,1,net.dtype);
                    for s = 1:Nsplit
                        disp('**********************')
                        disp([' Run time #' num2str(s)])
                        if net.permuteData == 1
                            [x_new, y_new] = dp.split(x, y, 0.9);
                        else
                            x_new   = x(trainIdx{s}, :);
                            y_new   = y(trainIdx{s}, :);
                        end
                        idx     = randperm(size(x_new,1));
                        x_new   = x_new(idx,:);
                        y_new   = y_new(idx);
                        numFolds    = net.numFolds;
                        E       = zeros(numFolds, 1, net.dtype);
                        max_LL  = zeros(numFolds, 1, net.dtype);
                        % Split data into different folds
                        numObs        = size(x_new, 1);
                        c             = cvpartition(numObs,'KFold',5);
                        partitionData = 1; %default
                        ratio         = net.ratio;
                        for n = 1:numFolds
                            if dispFlag == 1
                                disp('   ------')
                                disp(['   Fold #' num2str(n)])
                            end
                            % Splitting data into train and val
                            if partitionData == 1
                                [xtrain, ytrain, xval, yval] = dp.split(x_new, y_new, ratio);
                            else
                                [xtrain, ytrain, xval, yval] = dp.partition(x_new, y_new, c,n);
                            end
                            % Normalize data
                            [xtrain, ytrain, xval, ~, ~, ~, mytrain, sytrain] = dp.normalize(xtrain, ytrain, xval, yval);
                            %% Network
                            % Train net
                            net.trainMode = 1;
                            [net, states, maxIdx] = network.initialization(net);
                            maxEpoch = net.maxEpoch;
                            % Val net
                            netV              = net;
                            netV.trainMode    = 0;
                            netV.batchSize    = 1;
                            netV.repBatchSize = 1;
                            [netV, statesV, maxIdxV] = network.initialization(netV);
                            normStatV = tagi.createInitNormStat(netV);
                            % Initalize weights and bias
                            net.gainS          = Gain(1)*ones(1,length(net.layer)-1);
                            net.gainS_v2hat    = Gain(2);     %0.02/13
                            theta    = tagi.initializeWeightBias(net);
                            normStat = tagi.createInitNormStat(net);
                            % Training
                            stop     = 0;
                            epoch    = 0;
                            LL_val   = zeros(maxEpoch,1);
                            best_LL  = -inf;
                            patience = 10;
                            delta    = -0.01;
                            while ~stop
                                if epoch >1
                                    idxtrain = randperm(size(ytrain, 1));
                                    ytrain   = ytrain(idxtrain, :);
                                    xtrain   = xtrain(idxtrain, :);
                                end
                                epoch = epoch + 1;
                                disp(['   epoch #' num2str(epoch)])
                                net.epoch = epoch;      %BD
                                net.epochCV = epoch;
                                [theta, normStat]    = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
                                % Validation
                                [~, ~, mzval, Szval] = network.regression(netV, theta, normStatV, statesV, maxIdxV, xval, []);
                                [ml_val, Sl_val]     = dp.denormalize(mzval(:,1), Szval(:,1), mytrain, sytrain);
                                LL_val(epoch,:)      = mt.loglik(yval, ml_val, Sl_val);
                                if any(isnan(LL_val))
                                    LL_val(epoch) = 0;
                                end
                                % Early Stopping
                                if LL_val < best_LL + delta
                                    counter = counter + 1;
                                    disp(['   counter #' num2str(counter) ' out of' num2str(patience)])
                                    if counter == patience
                                        maxEpoch = epoch;
                                        break;
                                    end
                                else
                                    best_LL_val = LL_val;
                                    counter     = 0;
                                end
                                
                                if epoch >= maxEpoch;break;end
                           
                            end
                            % Plot
%                             plot(LL_val)
                            [M, N]    = max(LL_val);
%                             %%
%                             while ~stop
% %                                 if epoch >1
% %                                     idxtrain = randperm(size(ytrain, 1));
% %                                     ytrain   = ytrain(idxtrain, :);
% %                                     xtrain   = xtrain(idxtrain, :);
% %                                 end
%                                 epoch = epoch + 1;
%                                 net.epoch = epoch;      %BD
%                                 net.epochCV = epoch;
%                                 [theta, normStat]    = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
%                                 if epoch >= N; break;end
%                             end
%                             % Validation
%                             [~, ~, mzval, Szval] = network.regression(netV, theta, normStatV, statesV, maxIdxV, xval, []);
%                             [ml_val, Sl_val]     = dp.denormalize(mzval(:,1), Szval(:,1), mytrain, sytrain);
%                             LL_val_opt           = mt.loglik(yval, ml_val, Sl_val);
                            E(n)      = N;
                            max_LL(n) = M;
                        end
                        avg_fold_E(s)    = round(mean(E));
%                         disp(['   opt Epoch #' num2str(avg_fold_E(s))])
%                         if avg_fold_E(s) <= 10
%                             check;
%                         end
                        avg_fold_LL(s)   = mean(max_LL);
                    end
                    avg_split_LL(k)  = mean(avg_fold_LL);
                    opt_epoch_k(:,k) = avg_fold_E;
                end
                [~,index_gs]     = max(avg_split_LL);
                Gain_factor      = dict(:,index_gs);
                opt_Epoch        = opt_epoch_k(:,index_gs);
        end
        function [best_gains, run_time, best_patience]   =  GridsearchV2(net,x,y, trainIdx)
%             rng('default') 
            best_LL       = -inf;
            best_gains    = [0;0];
            best_patience = [];
            delta         = 0.01;
            counter       = 0;
            Nsplit        = net.numSplits;
            start_time    = tic;
%             runtime_persplit = zeros(Nsplit,1);
            for s = 1:Nsplit
%                 disp('**********************')
%                 disp([' Run time #' num2str(s)])
                net.permuteData   = 0;                             %BD changed from 0 to 1
                if isempty(trainIdx)
                    net.permuteData    = 1;
                end
                % Data
                if net.permuteData == 1
                    [x_new, y_new] = dp.split(x, y, 0.9);
                else
                    if net.gpu == 1
                       trainId = gpuArray(trainIdx{s});
                      
                    else
                        trainId = trainIdx{s};
                       
                    end
                    x_new = xtrain(trainId, :);
                    y_new = ytrain(trainId, :);
                    
                end
                
                [xtrain, ytrain, xval, yval] = dp.split(x_new, y_new, 0.8);
                % Normalize data
                [xtrain, ytrain, xval, ~, ~, ~, mytrain, sytrain] = dp.normalize(xtrain, ytrain, xval, yval);
                %% Network
                % Train net
                net.trainMode = 1;
                [net, states, maxIdx] = network.initialization(net);
%                 maxEpoch = net.maxEpoch;
                % Val net
                netV              = net;
                netV.trainMode    = 0;
                netV.batchSize    = 1;
                netV.repBatchSize = 1;
                [netV, statesV, maxIdxV] = network.initialization(netV);
                normStatV = tagi.createInitNormStat(netV);
                % Initalize hypers
                p_list       = [3];
                alpha        = [0.1 0.5 ];     %0.1:0.1:1
                beta         = [1e-03 1e-04];
                dict         = combvec(alpha, beta);
%                 start_time_split = tic;
                for p = 1:length(p_list)
                    patience = p_list(p);
                    for k = 1:size(dict,2)
                        Gain = dict(:,k);
                        disp([' Search loop #' num2str(k)])
                        % Initalize weights and bias
                        net.gainS          = Gain(1)*ones(1,length(net.layer)-1);
                        net.gainS_v2hat    = Gain(2);     %0.02/13
                        theta    = tagi.initializeWeightBias(net);
                        normStat = tagi.createInitNormStat(net);
                        % Training
                        stop         = 0;
                        epoch        = 0;
                        best_LL_val  = nan;
                        loglk        = [];
                        while ~stop
                            if epoch >1
                                idxtrain = randperm(size(ytrain, 1));
                                ytrain   = ytrain(idxtrain, :);
                                xtrain   = xtrain(idxtrain, :);
                            end
                            epoch = epoch + 1;
                            disp(['   epoch #' num2str(epoch)])
                            maxEpoch  = net.maxEpoch;
                            net.epoch = epoch;      %BD
                            net.epochCV = epoch;

                            [theta, normStat]    = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
                            % Validation
                            [~, ~, mzval, Szval] = network.regression(netV, theta, normStatV, statesV, maxIdxV, xval, []);
                            [ml_val, Sl_val]     = dp.denormalize(mzval(:,1), Szval(:,1), mytrain, sytrain);
                            LL_val               = mt.loglik(yval, ml_val, Sl_val);
                            loglk                = [loglk; LL_val];
                            if isnan(LL_val)
                               break;
                            end
                               
%                             
                            %% Early Stopping
                            early_stop = 1;
                            if early_stop == 1
%                             if ~isnan(LL_val)
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
                            end
%                             else
%                                 break;
%                             end
                            %%     
                            if epoch >= maxEpoch
                                LL = LL_val;
                                %disp([' early stopping epoch is:' num2str(maxEpoch)])
                                break;
                            end

                        end
                        scatter(1:epoch, loglk)
                        drawnow
                        if LL > best_LL && ~isnan(LL_val)
                            best_LL       = LL;
                            best_gains    = Gain;
                            best_patience = patience;
                        end
                    end
                end
%              runtime_persplit(s) = toc(start_time_split);  
            end
            run_time = toc(start_time);
            disp([' Best Gain 1:' num2str(best_gains(1))])
            disp([' Best Gain 2:' num2str(best_gains(2))])
            disp([' Best patience:' num2str(best_patience)])
            disp([' total runtime:' num2str(run_time)])
        end
        
        function NEpoch            = crossvalHomo(net, x, y)
            % Split data into different folds
            numFolds    = net.numFolds;
            numObs      = size(x, 1);
            c           = cvpartition(numObs,'KFold',5);
%             foldIdx     = dp.kfolds(numObs, numFolds);
            permuteData = net.permuteData;
            ratio       = net.ratio;
            E           = zeros(numFolds, 1, net.dtype);
            
            dispFlag = 1;
            for n = 1:numFolds
                if dispFlag == 1
                    disp('   ------')
                    disp(['   Fold #' num2str(n)])
                end
                % Splitting data into train and val
                if permuteData == 1
                    [xtrain, ytrain, xval, yval] = dp.split(x, y, ratio);
                else
                [xtrain, ytrain, xval, yval] = dp.partition(x, y, c, n);
                end
                % Normalize data
                [xtrain, ytrain, xval, ~, ~, ~, mytrain, sytrain] = dp.normalize(xtrain, ytrain, xval, yval);
                %% Network
                % Train net
                net.trainMode = 1;
                [net, states, maxIdx] = network.initialization(net);
                maxEpoch = net.maxEpoch;
                % Val net
                netV              = net;
                netV.trainMode    = 0;
                netV.batchSize    = 1;
                netV.repBatchSize = 1;
                [netV, statesV, maxIdxV] = network.initialization(netV);
                normStatV = tagi.createInitNormStat(netV);
            
                % Initalize weights and bias
                theta    = tagi.initializeWeightBias(net);
                normStat = tagi.createInitNormStat(net);
                
                % Training
                stop     = 0;
                epoch    = 0;
                LL_val   = zeros(maxEpoch,1);
                while ~stop
                    if epoch >1
                        idxtrain = randperm(size(ytrain, 1));
                        ytrain   = ytrain(idxtrain, :);
                        xtrain   = xtrain(idxtrain, :);
                    end
                    epoch = epoch + 1;
                    net.epoch = epoch;      %BD
                    net.epochCV = epoch;
                    [theta, normStat, ~, ~, sv_hom] = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
%                     net.v2hat_LL = svresult_LL;
                    
                    % Validation
                    netV.sv = sv_hom;
                    [~, ~, mzval, Szval] = network.regression(netV, theta, normStatV, statesV, maxIdxV, xval, []);
                    [ml_val, Sl_val]     = dp.denormalize(mzval(:,1), Szval(:,1), mytrain, sytrain);
                    LL_val(epoch,:)      = mt.loglik(yval, ml_val, Sl_val+netV.sv(1));
                    if epoch >= maxEpoch; break;end
                end
                [~, N] = max(LL_val);
                E(n)   = N;
            end
            NEpoch  = round(mean(E));
        end
    end
end
