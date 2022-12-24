%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         task
% Description:  Run different tasks such as classification, regression, etc
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      July 02, 2020
% Updated:      December 02, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef task
    methods (Static)                
        % Classification
        function runClassification(net, x, y, trainIdx, testIdx, trainedModelDir)
            % Initialization          
            cdresults  = net.cd;
            modelName  = net.modelName;
            dataName   = net.dataName;
            savedEpoch = net.savedEpoch;
            maxEpoch   = net.maxEpoch;
            
            % Initialize net
            net.trainMode = 1;
            [net, states, maxIdx, netInfo] = network.initialization(net); 
            if isempty(trainedModelDir)                  
                theta    = tagi.initializeWeightBias(net); 
                normStat = tagi.createInitNormStat(net); 
            else
                l        = load(trainedModelDir);
                theta    = l.theta;
                normStat = l.normStat;
            end
            % Data
            [xtrain, ytrain, trainLabels, trainEncoderIdx] = dp.selectData(x, y, net.labels, net.encoderIdx, trainIdx);  
            [xtest, ~, testLabels, testEncoderIdx] = dp.selectData(x, y, net.labels, net.encoderIdx, testIdx);
            clear x y
            % Training
            if net.displayMode == 1
                disp(' ')
                disp('Training... ')
            end
            net.trainMode = 1;
            daEnable     = net.da.enable;
            erTrain      = zeros(size(xtrain, 4), net.maxEpoch, net.dtype);
            PnTrain      = zeros(size(xtrain, 4), net.numClasses, net.maxEpoch, net.dtype);
            erTest       = zeros(size(xtest, 4), net.maxEpoch, net.dtype);
            PnTest       = zeros(size(xtest, 4), net.numClasses, net.maxEpoch, net.dtype);

            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            timeTest  = 0;
            while ~stop
                ts = tic;
                epoch = epoch + 1;
                if epoch == 1
                    net.da.enable = 1;
                else
                    net.da.enable   = daEnable;
                    idxtrain        = randperm(size(ytrain, 1));
                    ytrain          = ytrain(idxtrain, :);
                    xtrain          = xtrain(:,:,:,idxtrain);
                    trainLabels     = trainLabels(idxtrain);
                    trainEncoderIdx = trainEncoderIdx(idxtrain, :);
                    net.sv          = net.sv*net.svDecayFactor;
                    net.normMomentum= 0.9;
                    if net.sv < net.svmin
                        net.sv = net.svmin;
                    end
                end
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(net.maxEpoch)])
                net.trainMode  = 1;
                net.labels     = trainLabels;
                net.encoderIdx = trainEncoderIdx;
                if any(net.learningRateSchedule == epoch)
                    net.sv = net.scheduledSv(net.learningRateSchedule==epoch);
                end
                [theta, normStat, PnTrain(:,:,epoch), erTrain(:,epoch)] = network.classification(net, theta, normStat, states, maxIdx, xtrain, ytrain);                             
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime/epoch*(net.maxEpoch-epoch)/60;
                tt = tic;
                disp('   Testing... ')
                net.trainMode     = 0;
                net.labels        = testLabels;
                net.encoderIdx    = testEncoderIdx;
                net.errorRateEval = 1;
                [~, ~, PnTest(:,:,epoch), erTest(:, epoch)] = network.classification(net, theta, normStat, states, maxIdx, xtest, []);
                disp(['   Error rate : ' num2str(100*mean(erTest(:, epoch))) '%'])
                timeTest    = timeTest + toc(tt);
                timeTestRem = timeTest/epoch*(net.maxEpoch-epoch)/60;
                
                if epoch >= maxEpoch
                    stop = 1;
                    disp(' ')
                    disp('Done.')
                end
                if stop ~= 1 && net.displayMode == 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem+timeTestRem) ' mins']);
                end
                if mod(epoch, savedEpoch)==0||epoch==maxEpoch
                    metric.erTest   = erTest;
                    metric.PnTest   = PnTest;
                    metric.erTrain  = erTrain;
                    metric.PnTrain  = PnTrain;
                    trainTimeEpoch  = [trainTime, timeTest];
                    task.saveClassificationNet(cdresults, modelName, dataName, theta, normStat, metric, trainTimeEpoch, netInfo, epoch)
                end
            end
        end
        
        % Regression
        function runRegression(net, x, y, trainIdx, testIdx)
            
            % Initialization          
            cdresults     = net.cd;
            modelName     = net.modelName;
            dataName      = net.dataName;
            saveModel     = net.saveModel;
            maxEpoch      = net.maxEpoch;
            svinit        = net.sv;
%             net.gainS          = 0.5*ones(1,length(net.layer)-1);
%             net.gainS_v2hat    = 1e-04;
            % Train net
            net.trainMode = 1;
            [net, states, maxIdx, netInfo] = network.initialization(net);
            normStat = tagi.createInitNormStat(net); 
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
            
            % Loop
            Nsplit        = net.numSplits;
            RMSElist      = zeros(Nsplit, 1);
            LLlist        = zeros(Nsplit, 1);
            LLlist_N      = zeros(Nsplit, 1);
            trainTimelist = zeros(Nsplit, 1);
            permuteData   = 0;                             %BD changed from 0 to 1
            if isempty(trainIdx) || isempty(testIdx)
                permuteData    = 1;
            end
            LLlist_trains  = zeros(maxEpoch,Nsplit);
            LLlist_tests   = zeros(maxEpoch,Nsplit);
            RMSElist_tests = zeros(maxEpoch,Nsplit);
%             Epochs         = zeros(Nsplit,1);
            stop_Epoch     = zeros(Nsplit,1);
            runtime_oneE   = zeros(maxEpoch,Nsplit);
            for s = 1:Nsplit
                disp('**********************')
                disp([' Run time #' num2str(s)])
                % Data
                if permuteData == 1
                    [xtrain, ytrain, xtest, ytest] = dp.split(x, y, 0.9);
                else
                    if net.gpu == 1
                       trainId = gpuArray(trainIdx{s});
                       testId  = gpuArray(testIdx{s});
                    else
                        trainId = trainIdx{s};
                        testId  = testIdx{s};
                    end
                    xtrain = x(trainId, :);
                    ytrain = y(trainId, :);
                    xtest  = x(testId, :);
                    ytest  = y(testId, :);
                end
                
               
                if net.val_data == 1
                    %  Validation set
                    [xtrain, ytrain, xval, yval] = dp.split(xtrain, ytrain,0.8);
                    
                    %  Normalize data
                    [~,~, xval] = dp.normalize(xtrain, ytrain, xval, yval);
                end
                
                if net.gs_Gain == 1
                    net.gainS          = net.Gains(1)*ones(1,length(net.layer)-1); % net.Gains(1) --BD
                     net.gainS_v2hat   = net.Gains(2);       % net.Gains(2)

                else
                    net.gainS         = net.gain_HP(1,1)*ones(1,length(net.layer)-1);
                    net.gainS_v2hat   = net.gain_HP(2,1);
                end
                % Cross-Validation for Epoch for Homo noise                               %BD
                if net.cv_Epoch_HomNoise == 1
                    maxEpoch          = opt.crossvalHomo(net, xtrain, ytrain);
                    net.epochCV       = maxEpoch;
                    disp([' No. of Epoch ' num2str(maxEpoch)])
                
                else
                    net.epochCV       = 1;
                    NEpoch            = 1;
                end
                [xtrain, ytrain, xtest, ~, ~, ~, mytrain, sytrain] = dp.normalize(xtrain, ytrain, xtest, ytest);
                %% Initalize weights and bias
                theta    = tagi.initializeWeightBias(net);
                normStat = tagi.createInitNormStat(net);
                net.sv   = svinit;
                %%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Training
                rumtime_s    = tic;
                stop         = 0;
                epoch        = 0;
                best_LL_val  = nan;
                patience     = net.patience;
                delta        = 0.01;
                counter      = 0;
                if ~isempty(net.epochlist)
                    net.maxEpoch = net.epochlist(s);
                end

                LLlist_test   = zeros(net.maxEpoch,1);
                RMSElist_test = zeros(net.maxEpoch,1);
                while ~stop
                    if epoch >1
                        idxtrain = randperm(size(ytrain, 1));
                        ytrain   = ytrain(idxtrain, :);
                        xtrain   = xtrain(idxtrain, :);
                    end
                    epoch    = epoch + 1;
                    maxEpoch = net.maxEpoch;
%                     epoch
                    net.epoch   = epoch;          %BD
                    if net.gs_Gain == 1
                        start_time_oneE = tic;    %BD
                        [theta, normStat] = network.regression(net, theta, normStat, states, maxIdx, xtrain, ytrain);
                        runtime_oneE(epoch,s) = toc(start_time_oneE);
                    end
%                   % Validation
                    if net.val_data == 1
                        [~, ~, mzval, Szval] = network.regression(netV, theta, normStatV, statesV, maxIdxV, xval, []);
                        [ml_val, Sl_val]     = dp.denormalize(mzval(:,1), Szval(:,1), mytrain, sytrain);
                        LL_val               = mt.loglik(yval, ml_val, Sl_val);
                        if any(isnan(LL_val))
                            LL_val = -inf;
                        end
                    end
%                     loglk(epoch)         = LL_val;
                    %% Early Stopping
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
                    %%
                    if net.early_stop == 1
                        if epoch >= maxEpoch
                            stop_Epoch(s,1) = maxEpoch;
                            disp([' early stopping epoch is:' num2str(maxEpoch)])
                            break;
                        end
                    end

                    %% Testing after each Epoch
                    %test_after_each_epoch = 1;
                    if net.early_stop == 0              %BD
                        nObsTest = size(xtest, 1);
                        [~, ~, yp, Syp] = network.regression(netT, theta, normStatT, statesT, maxIdxT, xtest, []);
                        if net.learnSv==1&&strcmp(net.noiseType, 'hete') % Online noise inference
                            ypM  = reshape(yp', [net.nl, 1, 2, nObsTest]);
                            SypM = reshape(Syp', [net.nl, 1, 2, nObsTest]);
                            
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
                        [ml, Sl]             = dp.denormalize(ml, Sl, mytrain, sytrain);
                        RMSElist_test(epoch) = mt.computeError(ytest, ml);
                        LLlist_test(epoch)   = mt.loglik(ytest, ml, Sl);
%                         if epoch == NEpoch
%                             net.v2hat_LL = svresult_LL;
%                         end
                        if (epoch == maxEpoch)
                            %                         Epochs(s) = maxEpoch;
                            break;
                        end
                    end
                    
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Finished running till maxEpoch
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
%                 scatter(1:length(LL_train), LL_train)
%                 drawnow
                %% Plotting
%                 figure;
% % %                 scatter(1:maxEpoch,LLlist_train,1,'ob');hold on;
%                 scatter(1:maxEpoch,LLlist_test,1,'dr')
%                 xlabel('epoch')
%                 ylabel('LL')
%                 legend('test')
%                 drawnow
% %                 [~,N_train] = max(LLlist_train);
%                 [~,N_test] = max(LLlist_test);
%                 stop_Epoch(s,1) = N_test;
%                 N_test
                % Write training time
                runtime_e = toc(rumtime_s);
                
%                 clear ml; clear Sl; clear yp; clear Syp;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Testing
                netT.v2hat_LL  = [2 2];
                net.sv = 0;
                
                nObsTest = size(xtest, 1);
                [~, ~, yp, Syp] = network.regression(netT, theta, normStatT, statesT, maxIdxT, xtest, []);               
                if net.learnSv==1&&strcmp(net.noiseType, 'hete') % Online noise inference
                    ypM  = reshape(yp', [net.nl, 1, 2, nObsTest]);
                    SypM = reshape(Syp', [net.nl, 1, 2, nObsTest]);
                    

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
                ml_N = ml; Sl_N = Sl; %BD
                ytest_N  = (ytest - nanmean(ytest))./nanstd(ytest);%BD
                [ml, Sl] = dp.denormalize(ml, Sl, mytrain, sytrain);
                
                % Evaluation
                RMSElist(s)      = mt.computeError(ytest, ml);
                LLlist(s)        = mt.loglik(ytest, ml, Sl);
                LLlist_N(s)      = mt.loglik(ytest_N, ml_N, Sl_N);%BD
                if isinf(LLlist(s)) %BD
                    LLlist(s) =   NaN;
                    RMSElist(s) = NaN;
                end
                if ~isreal(LLlist(s)) %BD
                    LLlist(s) =   NaN;
                    RMSElist(s) = NaN;
                end
                trainTimelist(s) = runtime_e;              
                disp(' ')
                disp(['      RMSE : ' num2str(RMSElist(s)) ])
                disp(['      LL   : ' num2str(LLlist(s))])
                disp(['      LL normal   : ' num2str(LLlist_N(s))])
%                 LLlist_trains(:,s) = LLlist_train;
%                 test_after_each_epoch = 1;
                if net.early_stop == 0 
                    if ~isreal(LLlist_test)
                        LLlist_test   = zeros(100,1);
                        RMSElist_test = zeros(100,1);
                    end
                    LLlist_tests(:,s) = LLlist_test;
                    RMSElist_tests(:,s) = RMSElist_test;
                end
                
                
            end
            
%             Disp_LLlist = 1;
            if net.early_stop == 0 
                mean_LLtest   = nanmean(LLlist_tests')';
                mean_RMSEtest = nanmean(RMSElist_tests')';
                figure;
                subplot(1,2,1)
    %             scatter(1:maxEpoch,mean(LLlist_trains'),1,'ob');hold on;
                scatter(1:maxEpoch,nanmean(LLlist_tests'),1,'dr')
                xlabel('epoch')
                ylabel('LL')
                legend('test')
                subplot(1,2,2)
                scatter(1:maxEpoch,nanmean(RMSElist_tests'),1,'dr')
                xlabel('epoch')
                ylabel('RMSE')
                legend('test')
                % compute the epoch and max value
    %             [M_train,N_train] = max(mean(LLlist_trains'));
                [M_test,N_test] = max(nanmean(LLlist_tests'));
                % output
                M_test
                N_test
                write_result = 1;
                if write_result == 1
                    dlmwrite(strcat('UCI_',net.dataName,'LLtest',num2str(net.maxEpoch),'Epoch','.txt'),mean_LLtest, 'delimiter','\t','newline','pc');
                    dlmwrite(strcat('UCI_',net.dataName,'RMSEtest',num2str(net.maxEpoch),'Epoch','.txt'),mean_RMSEtest, 'delimiter','\t','newline','pc')
                end
            else
                dlmwrite(strcat('UCI_',net.dataName,'Epochlist','.txt'),stop_Epoch, 'delimiter','\t','newline','pc');
            end

            if ~isnan(saveModel)
                disp('Saving model...')
                metric.RMSElist    = RMSElist;
                metric.LLlist      = LLlist;
                metric.LLlist_N    = LLlist_N;
                task.saveRegressionNet(cdresults, modelName, dataName, theta, normStat, metric, trainTimelist, netInfo, maxEpoch)
            end
            % Saving results to a text file
            save_file = 1;
            if save_file == 1
                if net.gs_Gain == 1
                    if net.early_stop == 1
                        fileID = fopen(append('UCI_',net.dataName,'.txt'),'w');
                        fprintf(fileID, 'Results\n\n');
                        fprintf(fileID,'%f +- %f\n',[[nanmean(metric.RMSElist) nanstd(metric.RMSElist)].'; [nanmean(metric.LLlist_N) nanstd(metric.LLlist_N)].';...
                            [nanmean(trainTimelist) nanstd(trainTimelist)].']);
                        fclose(fileID);
                    else
                        fileID = fopen(append('UCI_',net.dataName,'100E','.txt'),'w');
                        fprintf(fileID, 'Results\n\n');
                        fprintf(fileID,'%f +- %f\n',[[nanmean(metric.RMSElist) nanstd(metric.RMSElist)].'; [nanmean(metric.LLlist) nanstd(metric.LLlist)].';...
                            [nanmean(metric.LLlist_N) nanstd(metric.LLlist_N)].'; [nanmean(trainTimelist) nanstd(trainTimelist)].';...
                            [nanmean(trainTimelist./maxEpoch) nanstd(trainTimelist./maxEpoch)].';[nanmean(nanmean(runtime_oneE)) nanstd(nanmean(runtime_oneE))].';]);
                        fclose(fileID);
                    end
                end
            end

            % Display final results
            disp('###################')
            disp(' Final results')
            disp(['  Avg. RMSE     : ' num2str(nanmean(metric.RMSElist)) ' +- ' num2str(nanstd(metric.RMSElist))])
            disp(['  Avg. LL       : ' num2str(nanmean(metric.LLlist)) ' +- ' num2str(nanstd(metric.LLlist))])
            disp(['  Avg. LL normal: ' num2str(nanmean(metric.LLlist_N)) ' +- ' num2str(nanstd(metric.LLlist_N))])
            disp(['  Avg. Time     : ' num2str(nanmean(trainTimelist)) ' +- ' num2str(nanstd(trainTimelist))])
            disp(['  Avg. Time for one epoch     : ' num2str(mean(mean(runtime_oneE))) ' +- ' num2str(std(mean(runtime_oneE)))])
            if net.early_stop == 1
                disp(['  Avg. opt. Epoch     : ' num2str(nanmean(stop_Epoch)) ' +- ' num2str(nanstd(stop_Epoch))])
            end
            disp('Done.')
        end
        
        function runRegressionWithNI(netD, netN, x, y, trainIdx, testIdx)
            % Initialization          
            cdresults  = netD.cd;
            modelName  = netD.modelName;
            dataName   = netD.dataName;
            saveModel  = netD.saveModel;
            maxEpoch   = netD.maxEpoch;
            svinit     = netD.sv;
%             layerConct = length(netD.layer) - 1;
            layerConct = 1;
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
            
            % Loop
            Nsplit        = netD.numSplits;
            RMSElist      = zeros(Nsplit, 1);
            LLlist        = zeros(Nsplit, 1);
            trainTimelist = zeros(Nsplit, 1);
            permuteData   = 0;
            if isempty(trainIdx) || isempty(testIdx)
                permuteData    = 1;
            end
            for s = 1:Nsplit
                disp('**********************')
                disp([' Run time #' num2str(s)])
                
                % Data
                if permuteData == 1
                    [xtrain, ytrain, xtest, ytest] = dp.split(x, y, 0.9);
                else
                    xtrain = x(trainIdx{s}, :);
                    ytrain = y(trainIdx{s}, :);
                    xtest  = x(testIdx{s}, :);
                    ytest  = y(testIdx{s}, :);
                end
                [xtrain, ytrain, xtest, ~, ~, ~, mytrain, sytrain] = dp.normalize(xtrain, ytrain, xtest, ytest);
                xtrainref = xtrain;
                ytrainref = ytrain;
                
                % Initalize weights and bias
                thetaD    = tagi.initializeWeightBias(netD);
                normStatD = tagi.createInitNormStat(netD);
                thetaN    = tagi.initializeWeightBias(netN);
                normStatN = tagi.createInitNormStat(netN);
                netD.sv    = svinit;
%                 
                %%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Training
                rumtime_s = tic;
                stop  = 0;
                epoch = 0;
                while ~stop
                    if epoch >1
                        idxtrain = randperm(size(ytrain, 1));
                        idxtrainCell{epoch+1} = idxtrain;
                        ytrain   = ytrain(idxtrain, :);
                        xtrain   = xtrain(idxtrain, :);
                    end
                    epoch = epoch + 1;
                    [thetaD, normStatD,~,~, sv] = network.regression(netD, thetaD, normStatD, statesD, maxIdxD, xtrain, ytrain);
                    %netD.sv = sv;
                    if epoch >= maxEpoch; break;end
                end
                runtime_e = toc(rumtime_s);
%                 
% %                 %%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                 % Testing (disable network.regressionWithNI)
% %                 [~, ~, ml, Sl] = network.regressionWithNI(netD1, thetaD, normStatD1, statesD1, maxIdxD1, netN1, thetaN, normStatN1, statesN1, maxIdxN1, xtest, [], layerConct);   
% %                 Sl = Sl + netD.sv(1);
% %                 [ml, Sl] = dp.denormalize(ml, Sl, mytrain, sytrain);
%                 
%                 % Evaluation
%                 RMSElist(s)      = mt.computeError(ytest, ml);
%                 LLlist(s)        = mt.loglik(ytest, ml, Sl);
%                 trainTimelist(s) = runtime_e;               
%                 disp(' ')
%                 disp(['      RMSE : ' num2str(RMSElist(s)) ])
%                 disp(['      LL   : ' num2str(LLlist(s))])  
                
                rumtime_s2 = tic;
                stop   = 0;
                epoch  = 0;      
                xtrain = xtrainref;
                ytrain = ytrainref;
                while ~stop
                    if epoch >1
                        idxtrain = idxtrainCell{epoch+1};                       
                        ytrain   = ytrain(idxtrain, :);
                        xtrain   = xtrain(idxtrain, :);
                    end
                    epoch = epoch + 1;
                    netN.epoch = epoch;        %BD
                    [thetaN, normStatN] = network.regressionWithNI(netD, thetaD, normStatD, statesD, maxIdxD, netN, thetaN, normStatN, statesN, maxIdxN, xtrain, ytrain, layerConct);%BD
                    if epoch >= maxEpoch; break;end
                end
                runtime_e2 = toc(rumtime_s2);
                runtime_e  = runtime_e + runtime_e2;
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Testing
                [~, ~,  ml, Sl] = network.regressionWithNI(netD1, thetaD, normStatD1, statesD1, maxIdxD1, netN1, thetaN, normStatN1, statesN1, maxIdxN1, xtest, [], layerConct);                       
                [ml, Sl] = dp.denormalize(ml, Sl, mytrain, sytrain);
                
                % Evaluation
                RMSElist(s)      = mt.computeError(ytest, ml);
                LLlist(s)        = mt.loglik(ytest, ml, Sl);
                trainTimelist(s) = runtime_e;               
                disp(' ')
                disp(['      RMSE : ' num2str(RMSElist(s)) ])
                disp(['      LL   : ' num2str(LLlist(s))])                
            end
            metric.RMSElist    = RMSElist;
            metric.LLlist      = LLlist;
%             if ~isnan(saveModel)
%                 disp('Saving model...')
%                 metric.RMSElist    = RMSElist;
%                 metric.LLlist      = LLlist;
%                 task.saveRegressionNet(cdresults, modelName, dataName, theta, normStat, metric, trainTimelist, netInfo, maxEpoch)
%             end
            
            % Display final results
            disp('###################')
            disp(' Final results')
            disp(['  Avg. RMSE     : ' num2str(nanmean(metric.RMSElist)) ' +- ' num2str(nanstd(metric.RMSElist))])
            disp(['  Avg. LL       : ' num2str(nanmean(metric.LLlist)) ' +- ' num2str(nanstd(metric.LLlist))])
            disp(['  Avg. Time     : ' num2str(nanmean(trainTimelist)) ' +- ' num2str(nanstd(trainTimelist))])
            disp('Done.')
        end
        function runRegressionWithNI2(net,netD,netW, x, y, trainIdx, testIdx)
            % Initialization          
            cdresults     = net.cd;
            modelName     = net.modelName;
            dataName      = net.dataName;
            saveModel     = net.saveModel;
            maxEpoch      = net.maxEpoch;
            netD.sv       = net.sv;
            netW.sv       = net.sv;
            net.trainMode  = 1;
            netT           = net;
            netT.trainMode = 0;
            % Train net mean
            netD.trainMode = 1;
            [netD, statesD, maxIdxD] = network.initialization(netD);
             
            % Test net mean
            netT_D              = netD;
            netT_D.trainMode    = 0;
            netT_D.batchSize    = 1;
            netT_D.repBatchSize = 1;
            [netT_D] = network.initialization(netT_D); 
            
            % Train net mean
            netW.trainMode = 1;
            [netW, statesW, maxIdxW] = network.initialization(netW);
            
            % Test net mean
            netT_W              = netW;
            netT_W.trainMode    = 0;
            netT_W.batchSize    = 1;
            netT_W.repBatchSize = 1;
            [netT_W] = network.initialization(netT_W); 
            
            % Loop
            Nsplit        = net.numSplits;
            RMSElist      = zeros(Nsplit, 1);
            LLlist        = zeros(Nsplit, 1);
            trainTimelist = zeros(Nsplit, 1);
            permuteData   = 0;                             %BD changed from 0 to 1
            if isempty(trainIdx) || isempty(testIdx)
                permuteData    = 1;
            end
            for s = 1:Nsplit
                disp('**********************')
                disp([' Run time #' num2str(s)])
                
                % Data
                if permuteData == 1
                    [xtrain, ytrain, xtest, ytest] = dp.split(x, y, 0.9);
                else
                    xtrain = x(trainIdx{s}, :);
                    ytrain = y(trainIdx{s}, :);
                    xtest  = x(testIdx{s}, :);
                    ytest  = y(testIdx{s}, :);
                end
                
                [xtrain, ytrain, xtest, ~, ~, ~, mytrain, sytrain] = dp.normalize(xtrain, ytrain, xtest, ytest);
                % Initalize weights and bias
                thetaD    = tagi.initializeWeightBias(netD);
                normStatD = tagi.createInitNormStat(netD);
                thetaW    = tagi.initializeWeightBias(netW);
                normStatW = tagi.createInitNormStat(netW);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Training
                rumtime_s = tic;
                stop  = 0;
                epoch = 0;
                while ~stop
                    if epoch >1
                        idxtrain = randperm(size(ytrain, 1));
                        ytrain   = ytrain(idxtrain, :);
                        xtrain   = xtrain(idxtrain, :);
                    end
                    epoch = epoch + 1;
                    net.epoch   = epoch;      %BD
                    net.epochCV = 1;
                    [thetaD, thetaW, normStatD, normStatW, ~, ~, svresult_LL] = network.regressionWithNI2(net, netD, netW, thetaD, normStatD, statesD, maxIdxD, thetaW, normStatW, statesW, maxIdxW, xtrain, ytrain);
                    if epoch == net.epochCV
                        net.v2hat_LL = svresult_LL;
                    end
                    if epoch >= maxEpoch; break;end
                end
                runtime_e = toc(rumtime_s);
                % Testing
                
                netT.v2hat_LL = svresult_LL;
                nObsTest = size(xtest, 1);
                [~, ~, ~, ~, yp,Syp,svresult_LL] = network.regressionWithNI2(netT, netT_D, netT_W, thetaD, normStatD, statesD, maxIdxD, thetaW, normStatW, statesW, maxIdxW, xtest, []); 
                if net.learnSv==1&&strcmp(net.noiseType, 'hete') % Online noise inference
                    ypM  = reshape(yp', [net.nl, 1, 2, nObsTest]);
                    SypM = reshape(Syp', [net.nl, 1, 2, nObsTest]);

                    %                     mv2  = reshape(ypM(:,:,2,:), [net.nl*nObsTest, 1]);
                    %                     Sv2  = reshape(SypM(:,:,2,:), [net.nl*nObsTest, 1]);
                    %                     mv2a = act.expFun(mv2, Sv2, net.gpu);
                    %                     [mv2a,~] = act.NoiseActFun(mv2, Sv2, net.NoiseActFunIdx, net.gpu);
                    ml   = reshape(ypM(:,:,1,:), [net.nl*nObsTest, 1]);
                    Sl   = reshape(SypM(:,:,1,:), [net.nl*nObsTest, 1]);
                elseif net.learnSv==0 && strcmp(net.noiseType, 'hete')
                    ml  = yp;
                    Sl  = Syp;
                else
                    ml = yp;
                    if strcmp(net.noiseType, 'homo')
                        Sl = Syp+net.sv(1);
                    else
                        Sl = Syp+net.sv(1).^2;
                    end
                end
                [ml, Sl] = dp.denormalize(ml, Sl, mytrain, sytrain);
                % Evaluation
                RMSElist(s)      = mt.computeError(ytest, ml);
                LLlist(s)        = mt.loglik(ytest, ml, Sl);
                if ~isreal(LLlist(s))
                    check;
                end
                trainTimelist(s) = runtime_e;               
                disp(' ')
                disp(['      RMSE : ' num2str(RMSElist(s)) ])
                disp(['      LL   : ' num2str(LLlist(s))])                
            end
            if ~isnan(saveModel)
                disp('Saving model...')
                metric.RMSElist    = RMSElist;
                metric.LLlist      = LLlist;
%                 task.saveRegressionNet(cdresults, modelName, dataName, thetaD, thetaW, normStatD, normStatW, metric, trainTimelist, maxEpoch)
            end
            
            % Display final results
            disp('###################')
            disp(' Final results')
            disp(['  Avg. RMSE     : ' num2str(nanmean(metric.RMSElist)) ' +- ' num2str(nanstd(metric.RMSElist))])
            disp(['  Avg. LL       : ' num2str(nanmean(metric.LLlist)) ' +- ' num2str(nanstd(metric.LLlist))])
            disp(['  Avg. Time     : ' num2str(nanmean(trainTimelist)) ' +- ' num2str(nanstd(trainTimelist))])
            disp('Done.')        
        end
        % Autoencoder
        function runAE(netE, netD, x, trainIdx, trainedModelDir)
            % Initialization            
            cd         = netE.cd;
            modelName  = netE.modelName;
            dataName   = netE.dataName;
            savedEpoch = netE.savedEpoch;
            maxEpoch   = netE.maxEpoch;
            
            netE.trainMode = 1;
            netD.trainMode = 1;
            [netE, statesE, maxIdxE, netEinfo] = network.initialization(netE); 
            [netD, statesD, maxIdxD, netDinfo] = network.initialization(netD); 
            if isempty(trainedModelDir)                  
                thetaE    = tagi.initializeWeightBias(netE); 
                normStatE = tagi.createInitNormStat(netE); 
                thetaD    = tagi.initializeWeightBias(netD); 
                normStatD = tagi.createInitNormStat(netD);
            else
                l         = load(trainedModelDir);
                thetaE    = l.thetaE;
                normStatE = l.normStatE;
                thetaD    = l.thetaD;
                normStatD = l.normStatD;
            end
            % Data
            xtrain = dp.selectData(x, [], [], [], trainIdx);
            % Training
            if netE.displayMode == 1
                disp(' ')
                disp('Training... ')
            end
            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            while ~stop
                ts    = tic;
                epoch = epoch +1;
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(netE.maxEpoch)])
                if epoch > 1
                    idxtrain = randperm(size(xtrain, 4));
                    xtrain   = xtrain(:,:,:,idxtrain);
                    netE.sv  = netE.sv*netE.svDecayFactor;
                    netD.sv  = netD.sv*netD.svDecayFactor;
                    if netE.sv < netE.svmin
                        netE.sv = netE.svmin;
                    end
                    if netD.sv < netD.svmin
                        netD.sv = netD.svmin;
                    end
                end
                [thetaE, thetaD, normStatE, normStatD] = network.AE(netE, thetaE, normStatE, statesE, maxIdxE, netD, thetaD, normStatD, statesD, maxIdxD, xtrain);
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime/epoch*(netE.maxEpoch-epoch)/60;
                
                % Save results after E epochs
                if mod(epoch, savedEpoch)==0||epoch==maxEpoch
                    task.saveAEnet(cd, modelName, dataName, thetaE, thetaD, normStatE, normStatD, netEinfo, netDinfo, epoch);
                end
                if epoch >= maxEpoch
                    stop = 1;
                    disp(' ')
                    disp('Done.')
                end
                if stop ~= 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem) ' mins']);
                end
            end
            % Output
            netE.theta    = thetaE;
            netE.normStat = normStatE;
            netD.theta    = thetaD;
            netD.normStat = normStatD;
        end
        function [thetaE, thetaEM, thetaES, thetaD, normStatE, normStatEM, normStatES, normStatD] = runAE_V2(netE, EMnet, ESnet, netD, thetaE, thetaEM, thetaES, thetaD, normStatE, normStatEM, normStatES, normStatD, x, Sx, trainIdx)
            % Data
            xtrain = dp.selectData(x, [], [], [], trainIdx);
            if ~isempty(Sx)
                Sxtrain = Sx(trainIdx);
            else
                Sxtrain = [];
            end
            % Training
            if netE.displayMode == 1
                disp(' ')
                disp('   Training... ')
            end
            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            while ~stop
                ts    = tic;
                epoch = epoch +1;
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(netE.maxEpoch)])
                if epoch > 1
                    idxtrain = randperm(size(xtrain, 4));
                    xtrain   = xtrain(:,:,:,idxtrain);
                    netD.sv  = netD.sv*netD.svDecayFactor;
                    EMnet.sv = EMnet.sv*EMnet.svDecayFactor;
                    if netD.sv < netD.svmin
                        netD.sv = netD.svmin;
                    end
                    if EMnet.sv < EMnet.svmin
                        EMnet.sv = EMnet.svmin;
                    end
                end
                [thetaE, thetaEM, thetaES, thetaD, normStatE, normStatEM, normStatES, normStatD] = network.AE_V2(netE, EMnet, ESnet, netD, thetaE, thetaEM, thetaES, thetaD, normStatE, normStatEM, normStatES, normStatD, xtrain, Sxtrain);
                if epoch >= netE.maxEpoch
                    stop = 1;
                end
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime/epoch*(netE.maxEpoch-epoch)/60;
                if stop ~= 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem) ' mins']);
                end
            end
        end
        function [thetaE, thetaD, thetaC, normStatE, normStatD, normStatC, metric, time] = runSSAE(netE, netD, Cnet,  thetaE, thetaD, thetaC, normStatE, normStatD, normStatC, x, Sx, xlabelled, Sxlabelled, ylabelled, updateIdx, trainIdx, testIdx)
            % Data
            % Data
            [xtrain, ~, trainLabels, ~] = dp.selectData(x, [], Cnet.labels, [], trainIdx);
            [xtest, ~, testLabels, ~] = dp.selectData(x, [], Cnet.labels, [], testIdx);
            if ~isempty(Sx)
                Sxtrain = Sx(trainIdx);
                Sxtest = Sx(testIdx);
            else
                Sxtrain = [];
                Sxtest  = [];
            end
            clear x y Sx ytrain
            % Training
            if Cnet.displayMode == 1
                disp(' ')
                disp('   Training... ')
            end
            daEnable  = netE.da.enable;
            runtime_s = tic;
            erTrain   = zeros(size(xtrain, 4), Cnet.maxEpoch, Cnet.dtype);
            PnTrain   = zeros(size(xtrain, 4), Cnet.numClasses, Cnet.maxEpoch, Cnet.dtype);
            erTest    = zeros(size(xtest, 4), Cnet.maxEpoch, Cnet.dtype);
            PnTest    = zeros(size(xtest, 4), Cnet.numClasses, Cnet.maxEpoch, Cnet.dtype);
            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            timeTest  = 0;
            while ~stop
                ts = tic;
                epoch = epoch + 1;
                if epoch == 1
                    netE.da.enable = 0;
                else
                    netE.da.enable  = daEnable;
                    idxtrain        = randperm(size(xtrain, 4));
                    xtrain          = xtrain(:,:,:,idxtrain);
                    trainLabels     = trainLabels(idxtrain);
                    netE.sv         = netE.sv*netE.svDecayFactor;
                    netD.sv         = netD.sv*netD.svDecayFactor;
                    Cnet.sv         = Cnet.sv*Cnet.svDecayFactor;
                    if netE.sv < netE.svmin
                        netE.sv = netE.svmin;
                    end
                    if netD.sv < netD.svmin
                        netD.sv = netD.svmin;
                    end
                    if Cnet.sv < Cnet.svmin
                        Cnet.sv = Cnet.svmin;
                    end
                end
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(netE.maxEpoch)])
                netE.trainMode  = 1;
                netD.trainMode  = 1;
                Cnet.trainMode  = 1;
                Cnet.labels     = trainLabels;
                [thetaE, thetaD, thetaC, normStatE, normStatD, normStatC, PnTrain(:,:,epoch), erTrain(:,epoch)] = network.SSAE(netE, netD, Cnet, thetaE, thetaD, thetaC, normStatE, normStatD, normStatC, xtrain, Sxtrain, xlabelled, Sxlabelled, ylabelled, updateIdx);
                if epoch >= netE.maxEpoch
                    stop = 1;
                end
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime/epoch*(netE.maxEpoch-epoch)/60;                
                tt        = tic;
                disp('   Testing... ')
                netE.trainMode    = 0;
                netD.trainMode    = 0;
                Cnet.trainMode    = 0;
                Cnet.labels       = testLabels;
                Cnet.errorRateEval= 1;
                [~, ~, ~, ~, ~, ~, PnTest(:,:,epoch), erTest(:, epoch)] = network.SSAE(netE, netD, Cnet, thetaE, thetaD, thetaC, normStatE, normStatD, normStatC, xtest, Sxtest, [], [], [], []);
                disp(['   Error rate : ' num2str(100*mean(erTest(:, epoch))) '%'])
                timeTest    = timeTest + toc(tt);
                timeTestRem = timeTest/epoch*(netE.maxEpoch-epoch)/60;
                if stop ~= 1 && netE.displayMode == 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem+timeTestRem) ' mins']);
                end
            end
            runtime_e = toc(runtime_s);
            % Outputs
            metric.erTest   = erTest;
            metric.PnTest   = PnTest;
            metric.erTrain  = erTrain;
            metric.PnTrain  = PnTrain;
            time            = [runtime_e, trainTime, timeTest];
        end   
                       
        % GAN
        function runGAN(netD, netG, x, trainIdx, trainedModelDir)
            % Initialization                       
            cdresults = netD.cd;
            modelName = netD.modelName;
            dataName  = netD.dataName;
            savedEpoch = netD.savedEpoch;
            maxEpoch  = netD.maxEpoch;
            
            netD.trainMode = 1;
            netG.trainMode = 1;
            [netD, statesD, maxIdxD, netDinfo] = network.initialization(netD);
            [netG, statesG, maxIdxG, netGinfo] = network.initialization(netG);              
            if isempty(trainedModelDir)                                  
                thetaD    = tagi.initializeWeightBias(netD); 
                normStatD = tagi.createInitNormStat(netD);
                thetaG    = tagi.initializeWeightBias(netG); 
                normStatG = tagi.createInitNormStat(netG);
            else
                l         = load(trainedModelDir);                
                thetaD    = l.thetaD;
                normStatD = l.normStatD;
                thetaG    = l.thetaG;
                normStatG = l.normStatG;
            end
            
            % Data
            xtrain = dp.selectData(x, [], [], [], trainIdx);
            % Training
            if netD.displayMode == 1
                disp(' ')
                disp('Training... ')
            end
            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            while ~stop
                ts    = tic;
                epoch = epoch +1;
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(netD.maxEpoch)])
                if epoch > 1
                    idxtrain  = randperm(size(xtrain, 4));
                    xtrain    = xtrain(:,:,:,idxtrain);
                    netD.sv = netD.sv*netD.svDecayFactor;
                    if netD.sv < netD.svmin
                        netD.sv = netD.svmin;
                    end
                    netG.sv = netG.sv*netG.svDecayFactor;
                    if netG.sv < netG.svmin
                        netG.sv = netG.svmin;
                    end
                end
                [thetaD, thetaG, normStatD, normStatG] = network.GAN(netD, thetaD, normStatD, statesD, maxIdxD, netG, thetaG, normStatG, statesG, maxIdxG, xtrain);
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime/epoch*(netD.maxEpoch-epoch)/60;
                
                % Save net after E epochs
                if mod(epoch, savedEpoch)==0||epoch==maxEpoch
                    task.saveGANnet(cdresults, modelName, dataName, thetaD, thetaG, normStatD, normStatG, trainTime, netDinfo, netGinfo, epoch)
                end
                
                if epoch >= maxEpoch
                    stop = 1;
                    disp(' ')
                    disp('Done.')
                end
                if stop ~= 1 && netD.displayMode == 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem) ' mins']);
                end
            end
            % Output 
            netD.theta    = thetaD;
            netD.normStat = normStatD;
            netG.theta    = thetaG;
            netG.normStat = normStatG;
        end        
        function [thetaD, thetaG, thetaC, thetaP, normStatD, normStatG, normStatC, normStatP, metric, time] = runSSGAN(netD, netG, Cnet, netP, thetaD, thetaG, thetaC, thetaP, normStatD, normStatG, normStatC, normStatP, x, Sx, xlabelled, Sxlabelled, ylabelled, updateIdx, trainIdx, testIdx)
            % Data
            [xtrain, ~, trainLabels, ~] = dp.selectData(x, [], Cnet.labels, [], trainIdx);
            [xtest, ~, testLabels, ~] = dp.selectData(x, [], Cnet.labels, [], testIdx);
            if ~isempty(Sx)
                Sxtrain = Sx(trainIdx);
                Sxtest = Sx(testIdx);
            else
                Sxtrain = [];
                Sxtest  = [];
            end
            clear x y Sx ytrain
            % Training
            if Cnet.displayMode == 1
                disp(' ')
                disp('   Training... ')
            end
            daEnable  = netD.da.enable;
            runtime_s = tic;
            erTrain   = zeros(size(xtrain, 4), Cnet.maxEpoch, Cnet.dtype);
            PnTrain   = zeros(size(xtrain, 4), Cnet.numClasses, Cnet.maxEpoch, Cnet.dtype);
            erTest    = zeros(size(xtest, 4), Cnet.maxEpoch, Cnet.dtype);
            PnTest    = zeros(size(xtest, 4), Cnet.numClasses, Cnet.maxEpoch, Cnet.dtype);
            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            timeTest  = 0;
            while ~stop
                ts = tic;
                epoch = epoch + 1;
                if epoch == 1
                    netD.da.enable = 0;
                else
                    netD.da.enable = daEnable;
                    idxtrain       = randperm(size(xtrain, 4));
                    xtrain         = xtrain(:,:,:,idxtrain);
                    trainLabels    = trainLabels(idxtrain);                   
                    netD.sv        = netD.sv*netD.svDecayFactor;
                    netG.sv        = netG.sv*netG.svDecayFactor;
                    Cnet.sv        = Cnet.sv*Cnet.svDecayFactor;
                    netP.sv        = Cnet.sv*Cnet.svDecayFactor;                   
                    if netD.sv < netD.svmin
                        netD.sv = netD.svmin;
                    end
                    if netG.sv < netG.svmin
                        netG.sv = netG.svmin;
                    end
                    if Cnet.sv < Cnet.svmin
                        Cnet.sv = Cnet.svmin;
                    end
                    if netP.sv < netP.svmin
                        netP.sv = netP.svmin;
                    end
                end
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(netD.maxEpoch)])                
                netD.trainMode  = 1;
                netG.trainMode  = 1;
                Cnet.trainMode  = 1;
                netP.trainMode  = 1;
                Cnet.labels     = trainLabels;
                [thetaD, thetaG, thetaC, thetaP, normStatD, normStatG, normStatC, normStatP, PnTrain(:,:,epoch), erTrain(:,epoch)] = network.SSGAN(netD, netG, Cnet, netP, thetaD, thetaG, thetaC, thetaP, normStatD, normStatG, normStatC, normStatP, xtrain, Sxtrain, xlabelled, Sxlabelled, ylabelled, updateIdx);
                if epoch >= netD.maxEpoch
                    stop = 1;
                end
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime/epoch*(netD.maxEpoch-epoch)/60;                
                tt        = tic;
                disp('   Testing... ')                
                netD.trainMode = 0;
                netG.trainMode = 0;
                Cnet.trainMode = 0;
                netP.trainMode = 0;
                Cnet.labels    = testLabels;
                Cnet.errorRateEval = 1;
                [~, ~, ~, ~, ~, ~, ~, ~, PnTest(:,:,epoch), erTest(:, epoch)] = network.SSGAN(netD, netG, Cnet, netP, thetaD, thetaG, thetaC, thetaP, normStatD, normStatG, normStatC, normStatP, xtest, Sxtest, [], [], [], []);
                disp(['   Error rate : ' num2str(100*mean(erTest(:, epoch))) '%'])
                timeTest    = timeTest + toc(tt);
                timeTestRem = timeTest/epoch*(netD.maxEpoch-epoch)/60;
                if stop ~= 1 && netD.displayMode == 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem+timeTestRem) ' mins']);
                end
            end
            runtime_e = toc(runtime_s);
            % Outputs
            metric.erTest  = erTest;
            metric.PnTest  = PnTest;
            metric.erTrain = erTrain;
            metric.PnTrain = PnTrain;
            time           = [runtime_e, trainTime, timeTest];
        end 
        
        function runInfoGAN(netD, netG, netQ, netP, x, trainIdx, trainedModelDir)
            % Initialization 
            cdresults  = netD.cd;
            modelName  = netD.modelName;
            dataName   = netD.dataName;
            savedEpoch  = netD.savedEpoch;
            maxEpoch   = netD.maxEpoch;
            
            netD.trainMode = 1;
            netG.trainMode = 1;
            netQ.trainMode = 1;
            netP.trainMode = 1;
            [netD, statesD, maxIdxD, netDinfo] = network.initialization(netD);
            [netG, statesG, maxIdxG, netGinfo] = network.initialization(netG);
            [netQ, statesQ, maxIdxQ, netQinfo] = network.initialization(netQ);
            [netP, statesP, maxIdxP, netPinfo] = network.initialization(netP);
            if isempty(trainedModelDir)                                  
                thetaD    = tagi.initializeWeightBias(netD); 
                normStatD = tagi.createInitNormStat(netD);
                thetaG    = tagi.initializeWeightBias(netG); 
                normStatG = tagi.createInitNormStat(netG);
                thetaQ    = tagi.initializeWeightBias(netQ); 
                normStatQ = tagi.createInitNormStat(netQ);
                thetaP    = tagi.initializeWeightBias(netP); 
                normStatP = tagi.createInitNormStat(netP);
            else
                l         = load(trainedModelDir);                
                thetaD    = l.thetaD;
                normStatD = l.normStatD;
                thetaG    = l.thetaG;
                normStatG = l.normStatG;
                thetaQ    = l.thetaQ;
                normStatQ = l.normStatQ;
                thetaP    = l.thetaP;
                normStatP = l.normStatP;
            end           
            % Data
            xtrain = dp.selectData(x, [], [], [], trainIdx);    
            % Training
            if netD.displayMode == 1
                disp(' ')
                disp('Training... ')
            end
            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            while ~stop
                ts    = tic;
                epoch = epoch +1;
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(netD.maxEpoch)])
                if epoch >= 1
                    idxtrain = randperm(size(xtrain, 4));
                    xtrain   = xtrain(:,:,:,idxtrain);
                    netD.sv  = netD.sv*netD.svDecayFactor;
                    netG.sv  = netG.sv*netG.svDecayFactor;
                    netQ.sv  = netQ.sv*netQ.svDecayFactor;
                    netP.sv  = netP.sv*netP.svDecayFactor;                   
                    if netD.sv < netD.svmin
                        netD.sv = netD.svmin;
                    end
                    if netG.sv < netG.svmin
                        netG.sv = netG.svmin;
                    end
                    if netQ.sv < netQ.svmin
                        netQ.sv = netQ.svmin;
                    end
                    if netP.sv < netP.svmin
                        netP.sv = netP.svmin;
                    end
                end
                [thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG, normStatQ, normStatP] = network.infoGAN(netD, thetaD, normStatD, statesD, maxIdxD, ...
                netG, thetaG, normStatG, statesG, maxIdxG, netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netP, thetaP, normStatP, statesP, maxIdxP, xtrain);
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime/epoch*(netD.maxEpoch-epoch)/60;
                
                if mod(epoch, savedEpoch)==0|| epoch==maxEpoch
                    task.saveinfoGANnet(cdresults, modelName, dataName, thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG, normStatQ, normStatP, trainTime, netDinfo, netGinfo, netQinfo, netPinfo, epoch)
                end
                if epoch >= maxEpoch
                    stop = 1;
                    disp(' ')
                    disp('Done.')
                end
                if stop ~= 1 && netD.displayMode == 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem) ' mins']);
                end
            end
        end        
        function runInfoGAN_V2(netD, netG, netQ, netQc, netP, x, trainIdx, trainedModelDir)
            % Initialization             
            cdresults  = netD.cd;
            modelName  = netD.modelName;
            dataName   = netD.dataName;
            savedEpoch  = netD.savedEpoch;
            maxEpoch   = netD.maxEpoch;
            
            netD.trainMode  = 1;
            netG.trainMode  = 1;
            netQ.trainMode  = 1;
            netQc.trainMode = 1;
            netP.trainMode  = 1;
            [netD, statesD, maxIdxD, netDinfo]     = network.initialization(netD);
            [netG, statesG, maxIdxG, netGinfo]     = network.initialization(netG);
            [netQ, statesQ, maxIdxQ, netQinfo]     = network.initialization(netQ);
            [netQc, statesQc, maxIdxQc, netQcinfo] = network.initialization(netQc);
            [netP, statesP, maxIdxP, netPinfo]     = network.initialization(netP);
            if isempty(trainedModelDir)                                  
                thetaD     = tagi.initializeWeightBias(netD); 
                normStatD  = tagi.createInitNormStat(netD);
                thetaG     = tagi.initializeWeightBias(netG); 
                normStatG  = tagi.createInitNormStat(netG);
                thetaQ     = tagi.initializeWeightBias(netQ); 
                normStatQ  = tagi.createInitNormStat(netQ);
                thetaQc    = tagi.initializeWeightBias(netQc); 
                normStatQc = tagi.createInitNormStat(netQc);
                thetaP     = tagi.initializeWeightBias(netP); 
                normStatP  = tagi.createInitNormStat(netP);
            else
                l          = load(trainedModelDir);                
                thetaD     = l.thetaD;
                normStatD  = l.normStatD;
                thetaG     = l.thetaG;
                normStatG  = l.normStatG;
                thetaQ     = l.thetaQ;
                normStatQ  = l.normStatQ;
                thetaQc    = l.thetaQc;
                normStatQc = l.normStatQc;
                thetaP     = l.thetaP;
                normStatP  = l.normStatP;
            end            
            % Data
            xtrain = dp.selectData(x, [], [], [], trainIdx);    
            % Training
            if netD.displayMode == 1
                disp(' ')
                disp('   Training... ')
            end
            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            while ~stop
                ts    = tic;
                epoch = epoch +1;
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(netD.maxEpoch)])
                if epoch >= 1
                    idxtrain = randperm(size(xtrain, 4));
                    xtrain   = xtrain(:,:,:,idxtrain);
                    netD.sv  = netD.sv*netD.svDecayFactor;
                    netG.sv  = netG.sv*netG.svDecayFactor;
                    netQ.sv  = netQ.sv*netQ.svDecayFactor;
                    netP.sv  = netP.sv*netP.svDecayFactor;                   
                    if netD.sv < netD.svmin
                        netD.sv = netD.svmin;
                    end
                    if netG.sv < netG.svmin
                        netG.sv = netG.svmin;
                    end
                    if netQ.sv < netQ.svmin
                        netQ.sv = netQ.svmin;
                    end
                    if netP.sv < netP.svmin
                        netP.sv = netP.svmin;
                    end
                end
                [thetaD, thetaG, thetaQ, thetaQc, thetaP, normStatD, normStatG, normStatQ, normStatQc, normStatP] = network.infoGAN_V2(netD, thetaD, normStatD, statesD, maxIdxD, ...
                    netG, thetaG, normStatG, statesG, maxIdxG, netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netQc, thetaQc, normStatQc, statesQc, maxIdxQc, netP, thetaP, normStatP, statesP, maxIdxP, xtrain);
                if epoch>=netD.maxEpoch
                    stop = 1;
                end
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime/epoch*(netD.maxEpoch-epoch)/60;
                
                % Save results after E epochs
                if mod(epoch, savedEpoch)==0||epoch==maxEpoch
                    task.saveinfoGANnet_V2(cdresults, modelName, dataName, thetaD, thetaG, thetaQ, thetaQc, thetaP, normStatD, normStatG, normStatQ, normStatQc, normStatP, trainTime, netDinfo, netGinfo, netQinfo, netQcinfo, netPinfo, epoch);
                end
                if stop ~= 1 && netD.displayMode == 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem) ' mins']);
                end
            end
        end
                
        function runACGAN(netD, netG, netQ, netP, x, y, updateIdx, trainIdx, trainedModelDir)
            % Initialization 
            cdresults = netD.cd;
            modelName = netD.modelName;
            dataName  = netD.dataName;
            savedEpoch  = netD.savedEpoch;
            maxEpoch   = netD.maxEpoch;
            
            netD.trainMode = 1;
            netG.trainMode = 1;
            netQ.trainMode = 1;
            netP.trainMode = 1;
            [netD, statesD, maxIdxD, netDinfo] = network.initialization(netD);
            [netG, statesG, maxIdxG, netGinfo] = network.initialization(netG);
            [netQ, statesQ, maxIdxQ, netQinfo] = network.initialization(netQ);
            [netP, statesP, maxIdxP, netPinfo] = network.initialization(netP);
            if isempty(trainedModelDir)                                  
                thetaD    = tagi.initializeWeightBias(netD); 
                normStatD = tagi.createInitNormStat(netD);
                thetaG    = tagi.initializeWeightBias(netG); 
                normStatG = tagi.createInitNormStat(netG);
                thetaQ    = tagi.initializeWeightBias(netQ); 
                normStatQ = tagi.createInitNormStat(netQ);
                thetaP    = tagi.initializeWeightBias(netP); 
                normStatP = tagi.createInitNormStat(netP);
            else
                l         = load(trainedModelDir);                
                thetaD    = l.thetaD;
                normStatD = l.normStatD;
                thetaG    = l.thetaG;
                normStatG = l.normStatG;
                thetaQ    = l.thetaQ;
                normStatQ = l.normStatQ;
                thetaP    = l.thetaP;
                normStatP = l.normStatP;
            end
            % Data
            [xtrain, ytrain, ~, updateIdxtrain] = dp.selectData(x, y, [], updateIdx, trainIdx);
            % Training
            if netD.displayMode == 1
                disp(' ')
                disp('Training... ')
            end
            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            while ~stop
                ts    = tic;
                epoch = epoch +1;
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(netD.maxEpoch)])
                if epoch > 1
                    idxtrain = randperm(size(xtrain, 4));
                    xtrain   = xtrain(:, :, :, idxtrain);
                    ytrain   = ytrain(idxtrain, :);
                    updateIdxtrain = updateIdxtrain(idxtrain, :);
                    netD.sv  = netD.sv*netD.svDecayFactor;
                    netG.sv  = netG.sv*netG.svDecayFactor;
                    netQ.sv  = netQ.sv*netQ.svDecayFactor;
                    netP.sv  = netP.sv*netP.svDecayFactor;                   
                    if netD.sv < netD.svmin
                        netD.sv = netD.svmin;
                    end
                    if netG.sv < netG.svmin
                        netG.sv = netG.svmin;
                    end
                    if netQ.sv < netQ.svmin
                        netQ.sv = netQ.svmin;
                    end
                    if netP.sv < netP.svmin
                        netP.sv = netP.svmin;
                    end
                end
                [thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG, normStatQ, normStatP] = network.ACGAN(netD, thetaD, normStatD, statesD, maxIdxD, ...
                netG, thetaG, normStatG, statesG, maxIdxG, netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netP, thetaP, normStatP, statesP, maxIdxP, xtrain, ytrain, updateIdxtrain);
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime/epoch*(netD.maxEpoch-epoch)/60;
                % Save after E epochs
                if mod(epoch, savedEpoch)==0||epoch==maxEpoch
                    task.saveinfoGANnet(cdresults, modelName, dataName, thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG, normStatQ, normStatP, trainTime, netDinfo, netGinfo, netQinfo, netPinfo, epoch)
                end
                
                if epoch >= maxEpoch
                    stop = 1;
                    disp(' ')
                    disp('Done.')
                end
                if stop ~= 1 && netD.displayMode == 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem) ' mins']);
                end
            end
            % Output 
            netD.theta    = thetaD;
            netD.normStat = normStatD;
            netG.theta    = thetaG;
            netG.normStat = normStatG;
            netQ.theta    = thetaQ;
            netQ.normStat = normStatQ;
            netP.theta    = thetaP;
            netP.normStat = normStatP;
        end 
        
        % Save functions
        function saveRegressionNet(cd, modelName, dataName, theta, normStat, metric, trainTime, netInfo, epoch)
            filename = [modelName, '_', 'E', num2str(epoch), '_', dataName];
            folder   = char([cd ,'/results/']);
            save([folder filename], 'theta', 'normStat', 'metric', 'trainTime', 'netInfo')
        end
        function saveClassificationNet(cd, modelName, dataName, theta, normStat, metric, trainTime, netInfo, epoch)
            filename = [modelName, '_', 'E', num2str(epoch), '_', dataName];
            folder   = char([cd ,'/results/']);
            save([folder filename], 'theta', 'normStat', 'metric', 'trainTime', 'netInfo')
        end
        function saveinfoGANnet(cd, modelName, dataName, thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG, normStatQ, normStatP, trainTime, netDinfo, netGinfo, netQinfo, netPinfo, epoch)
            filename       = [modelName, '_', 'E', num2str(epoch), '_', dataName];
            folder         = char([cd ,'/results/']);
            save([folder filename], 'thetaD',    'thetaG',    'thetaQ',    'thetaP',...
                'normStatD', 'normStatG', 'normStatQ', 'normStatP',...
                'netDinfo',  'netGinfo',  'netQinfo',  'netPinfo', 'trainTime');
        end
        function saveinfoGANnet_V2(cd, modelName, dataName, thetaD, thetaG, thetaQ, thetaQc, thetaP, normStatD, normStatG, normStatQ, normStatQc, normStatP, trainTime, netDinfo, netGinfo, netQinfo, netQcinfo, netPinfo, epoch)
            filename       = [modelName, '_', 'E', num2str(epoch), '_', dataName];
            folder         = char([cd ,'/results/']);
            save([folder filename], 'thetaD',    'thetaG',    'thetaQ',    'thetaQc', 'thetaP',...
                                    'normStatD', 'normStatG', 'normStatQ', 'normStatQc', 'normStatP',...
                                    'netDinfo',  'netGinfo',  'netQinfo',  'netQcinfo', 'netPinfo', 'trainTime');
        end
        function saveGANnet(cd, modelName, dataName, thetaD, thetaG, normStatD, normStatG, trainTime, netDinfo, netGinfo, epoch)
            filename  = [modelName, '_', 'E', num2str(epoch), '_', dataName];
            folder    = char([cd ,'/results/']);
            save([folder filename],  'thetaD', 'thetaG', 'normStatD', 'normStatG', 'netDinfo', 'netGinfo', 'trainTime')
        end
        function saveAEnet(cd, modelName, dataName, thetaE, thetaD, normStatE, normStatD, netEinfo, netDinfo, epoch)
            filename = [modelName, '_', 'E', num2str(epoch), '_', dataName];
            folder   = char([cd ,'/results/']);
            save([folder filename], 'thetaE', 'thetaD', 'normStatE', 'normStatD', 'netDinfo', 'netEinfo')
        end   
        
    end
end