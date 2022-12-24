%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         network
% Description:  Build networks relating to each task
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      July 02, 2020
% Updated:      December 02, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef network
    methods (Static)      
        % Classification
        function [theta, normStat, Pn, er] = classification(net, theta, normStat, states, maxIdx, x, y)
            % Initialization           
            numObs = size(x, 4);
            numDataPerBatch = net.batchSize*net.repBatchSize;
            if net.learnSv == 1
                ny = net.ny-net.nv2;
            else
                ny = net.ny;
            end
            if net.errorRateEval == 1
                Pn = zeros(numObs, net.numClasses, net.dtype);
                er = zeros(numObs, 1, net.dtype);
                [classObs, classIdx] = dp.class_encoding(net.numClasses);
                addIdx   = reshape(repmat(colon(0, ny, (numDataPerBatch-1)*ny), [net.numClasses, 1]), [net.numClasses*numDataPerBatch, 1]);
                classObs = repmat(classObs, [numDataPerBatch, 1]);
                classIdx = repmat(classIdx, [numDataPerBatch, 1]) + cast(addIdx, class(classIdx));
            else
                er = nan;
                Pn = nan;
            end
            % Loop
            loop     = 0;
            time_tot = 0;
            for i = 1:numDataPerBatch:numObs
                timeval  = tic;
                loop     = loop + 1;
                idxBatch = i:i+numDataPerBatch-1;
                xloop    = dp.dataLoader(x(:,:,:,idxBatch), net.da, net.batchSize, net.repBatchSize, net.trainMode);
                xloop    = reshape(xloop, [net.batchSize*net.nodes(1), net.repBatchSize]);
                states   = tagi.initializeInputs(states, xloop, [], [], [], [], [], [], [], [], net.xsc);
                % Training
                udIdx = dp.selectIndices(net.encoderIdx(idxBatch, :), numDataPerBatch, ny, net.dtype);
                if net.trainMode == 1
                    yloop                      = reshape(y(idxBatch, :)', [net.batchSize*ny, net.repBatchSize]);
                    [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta, normStat, states, maxIdx); 
                    [deltaM, deltaS,...
                        deltaMx, deltaSx]      = tagi.hiddenStateBackwardPass(net, theta, normStat, states, yloop, [], udIdx, maxIdx);
                    deltaTheta                 = tagi.parameterBackwardPass(net, theta, normStat, states, deltaM, deltaS, deltaMx, deltaSx);
                    theta                      = tagi.globalParameterUpdate(theta, deltaTheta, net.gpu);                    
                else 
                    states = tagi.feedForwardPass(net, theta, normStat, states, maxIdx);                   
                end
                [~, ~, ma, Sa] = tagi.extractStates(states);
                if net.learnSv == 1
                    [ml, mv2] = tagi.detachMeanVar(ma{end}, net.nl, net.nv2, net.batchSize, net.repBatchSize);
                    [Sl, Sv2] = tagi.detachMeanVar(Sa{end}, net.nl, net.nv2, net.batchSize, net.repBatchSize);
                     mv2a     = act.expFun(mv2, Sv2, net.gpu);
                     ml       = reshape(ml, [ny*net.batchSize*net.repBatchSize, 1]);
                     Sl       = reshape(Sl+mv2a, [ny*net.batchSize*net.repBatchSize, 1]) + net.sv.^2;
                else 
                    ml = reshape(ma{end}, [numDataPerBatch*net.nl, 1]);
                    Sl = reshape(Sa{end}+net.sv.^2, [numDataPerBatch*net.nl, 1]);
                end 
                time_loop = toc(timeval);
                time_tot  = time_tot + time_loop;
                time_rem  = double(time_tot)/(double(idxBatch(end)))*(numObs-double(idxBatch(end)))/60;
                if net.errorRateEval == 1
                    P = dp.obs2class(ml, Sl, classObs, classIdx);
                    P = reshape(P, [net.numClasses, numDataPerBatch])';
                    P = gather(P);
                    Pn(idxBatch, :) = P;
                    er(idxBatch, :) = mt.errorRate(net.labels(idxBatch, :)', P');   
%                     % Display error rate  
                    if mod(idxBatch(end), net.obsShow) == 0 && i > 1 && net.trainMode == 1 && net.displayMode == 1
                        disp(['     Error Rate : ' sprintf('%0.2f', 100*mean(er(max(1,i-200):i))) '%']);
                        disp(['     Time left  : ' sprintf('%0.2f',time_rem) ' mins'])
                    end
                else
                    if mod(idxBatch(end), net.obsShow) == 0 && i > 1 && net.trainMode == 1 && net.displayMode == 1
                        disp(['     Time left : ' sprintf('%0.2f',time_rem) ' mins'])
                    end
                end                               
            end
        end 
        function [theta, normStat, Pn, er] = classificationMultiGPUs(net, theta, normStat, states, maxIdx, x, y)
            % Initialization
            numObs = size(x, 4);
            numDataPerBatch = net.batchSize*net.repBatchSize*net.numDevices;           
            if net.learnSv == 1
                ny = net.ny-net.nv2;
            else
                ny = net.ny;
            end
            if net.errorRateEval == 1
                Pn = zeros(numObs, net.numClasses, net.dtype);
                er = zeros(numObs, 1, net.dtype);
                [classObs, classIdx] = dp.class_encoding(net.numClasses);
                numUdObs = size(classObs, 2);
                addIdx   = reshape(repmat(colon(0, ny, (net.batchSize*net.repBatchSize-1)*ny), [net.numClasses, 1]), [net.numClasses*net.batchSize*net.repBatchSize, 1]);
                classObs = repmat(classObs, [net.batchSize*net.repBatchSize, 1]);
                classIdx = repmat(classIdx, [net.batchSize*net.repBatchSize, 1]) + cast(addIdx, class(classIdx));
            else
                er = nan;
                Pn = nan;
            end
            % Loop
            loop     = 0;
            time_tot = 0;
            udIdxM   = zeros(numUdObs*net.batchSize*net.repBatchSize, net.numDevices, net.dtype, 'gpuArray');
            for i = 1:numDataPerBatch:numObs
                timeval   = tic;
                loop      = loop + 1;            
                idxBatch  = i:i+numDataPerBatch-1;
                idxBatchM = reshape(idxBatch, [net.batchSize*net.repBatchSize, net.numDevices]);
                xloop     = dp.dataLoader(x(:,:,:,idxBatch), net.da, net.trainMode);
                if net.trainMode == 1   
                    for d = 1:net.numDevices
                        udIdxM(:, d)  = dp.selectIndices(net.encoderIdx(idxBatchM(:, d), :), net.batchSize*net.repBatchSize, ny, net.dtype);
                    end
                    yloop  = reshape(y(idxBatch, :)', [net.batchSize*net.repBatchSize*net.numDevices* ny, 1]);
                    xloopM = reshape(xloop, [net.batchSize*net.nx, net.repBatchSize, net.numDevices]);
                    yloopM = reshape(yloop, [net.batchSize*net.repBatchSize*ny, net.numDevices]);
                    spmd
                        netDist   = net;
                        thetaDist = theta;
                        xdist     = gpuArray(xloopM(:, :, labindex));
                        states    = tagi.initializeInputs(states, xdist, [], [], [], [], [], [], [], [], net.xsc);
                        [states, normStat, maxIdx] = tagi.feedForwardPass(netDist, thetaDist, normStat, states, maxIdx);
                    end
                    spmd
                        ydist     = gpuArray(yloopM(:, labindex));
                        udIdxDist = gpuArray(udIdxM(:, labindex));
                        [deltaM, deltaS, deltaMx, deltaSx] = tagi.hiddenStateBackwardPass(netDist, thetaDist, normStat, states, ydist, [], udIdxDist, maxIdx);
                        deltaTheta = tagi.parameterBackwardPass(netDist, thetaDist, normStat, states, deltaM, deltaS, deltaMx, deltaSx);
                    end
%                     spmd
%                         netDist   = net;
%                         thetaDist = theta;
%                         xdist     = gpuArray(xloopM(:, :, labindex));
%                         ydist     = gpuArray(yloopM(:, labindex));
%                         udIdxDist = gpuArray(udIdxM(:, labindex));
%                         states    = tagi.initializeInputs(states, xdist, [], [], [], [], [], [], [], [], net.xsc);
%                         [states, normStat, maxIdx] = tagi.feedForwardPass_V3(netDist, thetaDist, normStat, states, maxIdx);
%                         [deltaM, deltaS, deltaMx, deltaSx] = tagi.hiddenStateBackwardPass(netDist, thetaDist, normStat, states, ydist, [], udIdxDist, maxIdx);
%                         deltaTheta = tagi.parameterBackwardPass(netDist, thetaDist, normStat, states, deltaM, deltaS, deltaMx, deltaSx);
%                     end
                    theta = tagi.globalParameterUpdateMultiGPUs(theta, deltaTheta, net.numParamsPerlayer, net.numDevices);  
                else 
                    xloopM = reshape(xloop, [net.batchSize*net.nx, net.repBatchSize, net.numDevices]);
                    labelM = reshape(net.labels(idxBatch, :), [net.batchSize*net.repBatchSize, net.numDevices]);
                    spmd
                        states = tagi.initializeInputs(states, xloopM(:, :, labindex), [], [], [], [], [], [], [], [], net.xsc);
                        states = tagi.feedForwardPass(net, theta, normStat, states, maxIdx);
                        [~, ~, mlloop, Slloop] = tagi.extractStates(states);
                        ml = reshape(mlloop{end}, [net.ny*net.batchSize*net.repBatchSize, 1]);
                        Sl = reshape(Slloop{end}, [net.ny*net.batchSize*net.repBatchSize, 1]);
                        P  = dp.obs2class(ml, Sl, classObs, classIdx);
                        P  = reshape(P, [net.numClasses, net.batchSize*net.repBatchSize])';                        
                        erloop = mt.errorRate(labelM(:, labindex)', P');
                    end  
                    Pn(idxBatch, :) = [gather(P{1}); gather(P{2})];
                    er(idxBatch, :) = [gather(erloop{2}), gather(erloop{2})]';
                end                              
                time_loop = toc(timeval);
                time_tot  = time_tot + time_loop;
                time_rem  = double(time_tot)/(double(idxBatch(end)))*(numObs-double(idxBatch(end)))/60;
                if mod(idxBatch(end), net.obsShow) == 0 && i > 1 && net.trainMode == 1 && net.displayMode == 1
                    disp(['     Time left : ' sprintf('%0.2f',time_rem) ' mins'])
                end                                             
            end
        end   
        
        % Regression 
        function [theta, normStat, zl, Szl, sv] = regression(net, theta, normStat, states, maxIdx, x, y)    % added K_Gain --BD
            % Initialization
            numObs = size(x, 1);
            numDataPerBatch = net.repBatchSize*net.batchSize;
            if net.gpu == 1
                zl  = zeros(numObs, net.ny, 'gpuArray');                                               %BD
                Szl = zeros(numObs, net.ny, 'gpuArray');
            else
                zl  = zeros(numObs, net.ny, net.dtype);                                               %BD
                Szl = zeros(numObs, net.ny, net.dtype);                                               %BD
            end
            sv  = net.sv;
            % Loop
            loop = 0;
            for i = 1:numDataPerBatch:numObs
                loop     = loop + 1;
                if numDataPerBatch==1
                    idxBatch = i:i+net.batchSize-1;
                else
                    if numObs-i>=numDataPerBatch
                        idxBatch = i:i+numDataPerBatch-1;
                    else
                        idxBatch = [i:numObs, randperm(i-1, numDataPerBatch-numObs+i-1)];
                    end
                end
                % Covariate
                xloop  = reshape(x(idxBatch, :)', [net.batchSize*net.nx, net.repBatchSize]);
                states = tagi.initializeInputs(states, xloop, [], [], [], [], [], [], [], [], net.xsc);
                
                % Training
                if net.trainMode == 1
                    % Observation
                    yloop = reshape(y(idxBatch, :)', [net.batchSize*net.nl, net.repBatchSize]);                  
                    [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta, normStat, states, maxIdx);
                    [deltaM, deltaS,deltaMx, deltaSx, ~, ~, sv] = tagi.hiddenStateBackwardPass(net, theta, normStat, states, yloop, [], [], maxIdx);
                    dtheta = tagi.parameterBackwardPass(net, theta, normStat, states, deltaM, deltaS, deltaMx, deltaSx);
                    theta  = tagi.globalParameterUpdate(theta, dtheta, net.gpu);
                    
%                     [~, ~, ma, Sa]    = tagi.extractStates(states);                                                         %BD
%                     zl(idxBatch, :)   = gather(reshape(ma{end}, [net.ny, numDataPerBatch])');                               %BD
%                     Szl(idxBatch, :)  = gather(reshape(Sa{end}, [net.ny, numDataPerBatch])');                               %BD
%                     if size(ma{end},1) > net.batchSize
%                         
%                         Szl(idxBatch, 1) = Szl(idxBatch, 1) + varW;     %BD
%                     end
                    
                    
                %% Testing    
                else
                    [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta, normStat, states, maxIdx); 
                    [~, ~, ma, Sa]   = tagi.extractStates(states);
                    
                    % Noise parameters     BD
                    if size(ma{end},1) > net.batchSize
                        B           = cast(net.batchSize, net.dtype);
                        rB          = cast(net.repBatchSize, net.dtype);
                        [~, mv2a]   = tagi.detachMeanVar(ma{end}, net.nl, net.nv2, B, rB);           %mla = mZ,  mv2a   = mV2hat
                        [~, Sv2a]   = tagi.detachMeanVar(Sa{end}, net.nl, net.nv2, B, rB);
                        % Activate log(\sigma_v2)
                        [mv2a, ~,~] = act.NoiseActFun(mv2a, Sv2a, net.NoiseActFunIdx, net.gpu);
                        E_v2hat                     = mv2a;
                        varW                        = E_v2hat;
                    end
                    % Output
                    zl(idxBatch, :)  = gather(reshape(ma{end}, [net.ny, numDataPerBatch])');                               %BD
                    Szl(idxBatch, :) = gather(reshape(Sa{end}, [net.ny, numDataPerBatch])');                               %BD
                    if size(ma{end},1) > net.batchSize
                        Szl(idxBatch, 1) = Szl(idxBatch, 1) + varW;     %BD
                    end
                    sv = net.sv;
                    
                end 
%                 
                
            end
            
        end
        function [thetaN, normStatN, zl, Szl, Prior_act_v2hat, Pos_act_v2hat]    = regressionWithNI(netD, thetaD, normStatD, statesD, maxIdxD, netN, thetaN, normStatN, statesN, maxIdxN, x, y, layerConct)
            % Initialization
            numObs = size(x, 1);
            numDataPerBatch = netN.repBatchSize*netN.batchSize;
            zl  = zeros(numObs, netN.ny, netN.dtype);
            Szl = zeros(numObs, netN.ny, netN.dtype);
            Prior_act_v2hat = zeros(numObs, 2, netN.dtype);                                        %BD
            Pos_act_v2hat   = zeros(numObs, 2, netN.dtype);                                        %BD
            % Loop
            loop = 0;
            for i = 1:numDataPerBatch:numObs
                loop     = loop + 1; 
%                 idxBatch = i:i+net.batchSize-1;
                if numDataPerBatch==1
                    idxBatch = i:i+netN.batchSize-1;
                else
                    if numObs-i>=numDataPerBatch
                        idxBatch = i:i+numDataPerBatch-1;
                    else
                        idxBatch = [i:numObs, randperm(i-1, numDataPerBatch-numObs+i-1)];
                    end
                end
                % Covariate
                xloop  = reshape(x(idxBatch, :)', [netD.batchSize*netD.nx, netD.repBatchSize]);
                statesD = tagi.initializeInputs(statesD, xloop, [], [], [], [], [], [], [], [], netD.xsc);
                % Training
                if netN.trainMode == 1                  
                    % Feed forward
                    yloop = reshape(y(idxBatch, :)', [netD.batchSize*netD.nl, netD.repBatchSize]);                  
                    [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD); 
                    [mzD, SzD, maD, SaD, JD] = tagi.extractStates(statesD);
%                     netN.statesD  = statesD;     %BD
                    statesN = tagi.initializeInputs(statesN, mzD{layerConct}, SzD{layerConct}, maD{layerConct}, SaD{layerConct}, JD{layerConct}, [], [], [], [], netN.xsc);
                    [statesN, normStatN, maxIdxN] = tagi.feedForwardPass(netN, thetaN, normStatN, statesN, maxIdxN); 
                     [~, ~, maN, SaN, JN] = tagi.extractStates(statesN);
%                     
%                     % Observation
                    [mv2a, Sv2a, Cv2a] = act.NoiseActFun(maN{end}, SaN{end}, netN.NoiseActFunIdx, netN.gpu);
                    % additional noise                 %BD
%                     if netN.epoch == 1
%                         add_noise = 0.1;
%                     else
%                        add_noise  = 0.1./(1+0.5.*netN.epoch);
%                     end
%                     Sv2a = Sv2a + add_noise;             %BD   random guess
%                     % Update noise
                    [~, ~, deltaMv2z, deltaSv2z, deltaMv2a, deltaSv2a] = tagi.noiseUpdate4regression(SzD{end}, maD{end}, SaD{end}, JD{end}, JN{end}, mv2a, Sv2a, Cv2a, yloop, netN.sv, netN.gpu);

                    [deltaMzN, deltaSzN, deltaMxN, deltaSxN] = tagi.hiddenStateBackwardPass(netN, thetaN, normStatN, statesN, deltaMv2z, deltaSv2z, [], maxIdxN);
                    dthetaN           = tagi.parameterBackwardPass(netN, thetaN, normStatN, statesN, deltaMzN, deltaSzN, deltaMxN, deltaSxN);
                    thetaN            = tagi.globalParameterUpdate(thetaN, dthetaN, netN.gpu);
                    Pos_act_mv2hat    = mv2a + deltaMv2a;                                    %BD
                    Pos_act_sv2hat    = Sv2a + deltaSv2a;                                    %BD
                    % Convert from activation scale to original scale
                    Prior_act_mv2hat     = mv2a;
                    Prior_act_sv2hat     = Sv2a;
                    Prior_act_v2hat(i,:) = [Prior_act_mv2hat Prior_act_sv2hat];               %BD
                    Pos_act_v2hat(i,:)   = [Pos_act_mv2hat Pos_act_sv2hat];                   %BD
                % Testing    
                else 
                    [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD); 
                    [mzD, SzD, maD, SaD, JD] = tagi.extractStates(statesD);
                    
                    statesN = tagi.initializeInputs(statesN, mzD{layerConct}, SzD{layerConct}, maD{layerConct}, SaD{layerConct}, JD{layerConct}, [], [], [], [], netN.xsc);
                    [statesN, normStatN, maxIdxN] = tagi.feedForwardPass(netN, thetaN, normStatN, statesN, maxIdxN); 
                    [~, ~, maN, SaN] = tagi.extractStates(statesN);
                    [mv2a,~] = act.NoiseActFun(maN{end}, SaN{end}, netN.NoiseActFunIdx, netN.gpu);
                    
                    zl(idxBatch, :)  = gather(reshape(maD{end}, [netD.ny, numDataPerBatch])');
                    Szl(idxBatch, :) = gather(reshape(SaD{end}+mv2a, [netD.ny, numDataPerBatch])');
                end 
            end
        end
        function [theta,normStat,zl,Szl, Prior_act_v2hat, Pos_act_v2hat, sv_res, svresult_LL, v2hat_out, Prior_act_Vnet,var_V]  = regressionOnlyNI(net,theta,normStat,states,maxIdx,x,y)                           %BD
            % Initialization
            numObs          = size(x, 1);
            numDataPerBatch = net.repBatchSize*net.batchSize;
            zl              = zeros(numObs, net.ny, net.dtype);
            Szl             = zeros(numObs, net.ny, net.dtype);
            Prior_act_v2hat = zeros(numObs, 2, net.dtype);                                        %BD
            Pos_act_v2hat   = zeros(numObs, 2, net.dtype);                                        %BD
            sv_res          = zeros(numObs, 2, net.dtype); 
            svresult_LL     = zeros(numObs, 2, net.dtype);
            v2hat_out       = zeros(numObs, 2, net.dtype);
            Prior_act_Vnet  = zeros(numObs, 2, net.dtype);
            var_V           = zeros(numObs, 1, net.dtype);
             % Loop
            loop = 0;
            for i = 1:numDataPerBatch:numObs
                loop     = loop + 1; 
%                 idxBatch = i:i+net.batchSize-1;
                if numDataPerBatch==1
                    idxBatch = i:i+net.batchSize-1;
                else
                    if numObs-i>=numDataPerBatch
                        idxBatch = i:i+numDataPerBatch-1;
                    else
                        idxBatch = [i:numObs, randperm(i-1, numDataPerBatch-numObs+i-1)];
                    end
                end
                % Covariate
                xloop  = reshape(x(idxBatch, :)', [net.batchSize*net.nx, net.repBatchSize]);
                states = tagi.initializeInputs(states, xloop, [], [], [], [], [], [], [], [], net.xsc);
                % Training
                if net.trainMode == 1                  
                    % Feed forward
                    yloop = reshape(y(idxBatch, :)', [net.batchSize*net.nl, net.repBatchSize]);                  
                    [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta, normStat, states, maxIdx); 
                    [~, ~, ma, Sa, J] = tagi.extractStates(states);
%                     netN.statesD  = statesD;     %BD
                    % Observation
                    if strcmp(net.noiseType,'hete')
                        % No activation for v2hat_he
                        if net.idx_LL_M ~= 1
                        [mv2a, Sv2a, Cv2a] = act.NoiseActFun(ma{end}, Sa{end}, net.NoiseActFunIdx, net.gpu);
                        else
                            mv2a = ma{end};
                            Sv2a = Sa{end};
                            Cv2a = [];
                        end
                        v2hat_LL = net.v2hat_LL;
                        if any(v2hat_LL<0)
                            check;
                        end
%                        [v2hat_LL(1), v2hat_LL(2)] = act.NoiseActFun(v2hat_LL(1), v2hat_LL(2), net.NoiseActFun_LL_Idx, net.gpu);
                        if net.idx_LL_M == 1
                            E_v2hat            =  mv2a + v2hat_LL(1);
                            var_v2hat          =  Sv2a + v2hat_LL(2);
                            [E_v2hat, var_v2hat, C_v2hat] = act.NoiseActFun(E_v2hat, var_v2hat, net.NoiseActFunIdx, net.gpu);
                        elseif net.idx_LL_M == 2
                            E_v2hat            = mv2a .* v2hat_LL(1);
                            var_v2hat          = Sv2a*v2hat_LL(2) + Sv2a*v2hat_LL(1)^2 + v2hat_LL(2)*mv2a^2;
                        else
                            E_v2hat            = mv2a;
                            var_v2hat          = Sv2a;
                        end
                        if net.var_mode == 1
                            varW = var_mode(net.EEW2g,net.SEW2g,net.aOg,E_v2hat,var_v2hat);
                        else
                            varW = E_v2hat;
                        end
                        var_V(i,:) = varW;
                        
                        [~, ~, deltaMv2z, deltaSv2z, deltaMv2a, deltaSv2a, mv2hat_LL_pos, Sv2hat_LL_pos] = tagi.noiseOnlyUpdate4regression(J{end}, mv2a, Sv2a, Cv2a, E_v2hat, var_v2hat, C_v2hat, varW, v2hat_LL, yloop, net.gpu,net.epoch,net.idx_LL_M);
                        [deltaMz, deltaSz, deltaMx, deltaSx] = tagi.hiddenStateBackwardPass(net, theta, normStat, states, deltaMv2z, deltaSv2z, [], maxIdx);
                    else
                        E_v2hat   = net.sv(1);
                        var_v2hat = net.sv(2);
                        if net.var_mode == 1
                            varW = var_mode(net.EEW2g,net.SEW2g,net.aOg,E_v2hat,var_v2hat);
                        else
                            varW = E_v2hat;
                        end
                        var_V(i,:) = varW;
                        [~, ~, deltaMv2z, deltaSv2z] = tagi.noiseOnlyUpdate4regression(J{end}, net.sv(1), net.sv(2), [], E_v2hat, var_v2hat, [], varW, [],yloop, net.gpu,[],[]);
                        [deltaMz, deltaSz, deltaMx, deltaSx] = tagi.hiddenStateBackwardPass(net, theta, normStat, states, deltaMv2z, deltaSv2z, [], maxIdx);
                    end
                    
                    dtheta           = tagi.parameterBackwardPass(net, theta, normStat, states, deltaMz, deltaSz, deltaMx, deltaSx);
                    theta            = tagi.globalParameterUpdate(theta, dtheta, net.gpu);
                    if strcmp(net.noiseType,'hete')
                        % Prior of v2hat
                        Prior_act_mv2hat     = mv2a + v2hat_LL(1);
                        Prior_act_sv2hat     = Sv2a + v2hat_LL(2);
                        Pos_act_mv2hat       = Prior_act_mv2hat   + deltaMv2a;                                  %BD
                        Pos_act_sv2hat       = Prior_act_sv2hat   + deltaSv2a;                                  %BD
                        
                        Prior_act_v2hat(i,:) = [Prior_act_mv2hat Prior_act_sv2hat];               %BD
                        Pos_act_v2hat(i,:)   = [Pos_act_mv2hat Pos_act_sv2hat];                   %BD
                        % Storing the pos v2hat_LL in network
                        net.v2hat_LL(1)      = mv2hat_LL_pos;
                        net.v2hat_LL(2)      = Sv2hat_LL_pos;
                        svresult_LL(i,:)     = net.v2hat_LL;
                    
                    else
                        % Saving sv to the net after each observation
                        net.sv(1)   = net.sv(1) + sum(deltaMv2z, 1);
                        net.sv(2)   = net.sv(2) + sum(deltaSv2z, 1);
                        sv_res(i,:) = net.sv;
                        
                    end
                %% Testing    
                else 
                    [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta, normStat, states, maxIdx);  
                    [~, ~, ma, Sa, ~] = tagi.extractStates(states);
                    if strcmp(net.noiseType,'hete')
                        if net.idx_LL_M ~= 1
                        [mv2a, Sv2a] = act.NoiseActFun(ma{end}, Sa{end}, net.NoiseActFunIdx, net.gpu);
                        else
                            mv2a = ma{end};
                            Sv2a = Sa{end};
                        end
                        v2hat_out(i,:) = [ma{end} Sa{end}];
                        v2hat_LL = net.v2hat_LL;
                        [v2hat_LL(1), v2hat_LL(2)] = act.NoiseActFun(v2hat_LL(1), v2hat_LL(2), net.NoiseActFun_LL_Idx, net.gpu);
                        if net.idx_LL_M == 1
                            E_v2hat              =  mv2a + v2hat_LL(1);
                            var_v2hat            =  Sv2a + v2hat_LL(2);
                            [E_v2hat,var_v2hat]  =  act.NoiseActFun(E_v2hat, var_v2hat, net.NoiseActFunIdx, net.gpu);
                        elseif net.idx_LL_M == 2
                            E_v2hat            = mv2a.* v2hat_LL(1);
                            var_v2hat          = Sv2a*v2hat_LL(2) + Sv2a*v2hat_LL(1)^2 + v2hat_LL(2)*mv2a^2;
                        else
                            E_v2hat            = mv2a;
                            var_v2hat          = Sv2a;
                        end
                        if net.var_mode == 1
                            varW = var_mode(net.EEW2g,net.SEW2g,net.aOg,E_v2hat,var_v2hat);
                        else
                            varW = E_v2hat;
                        end
                        Prior_act_mVnet         = mv2a;
                        Prior_act_sVnet         = Sv2a;
                        Prior_act_mv2hat        = E_v2hat;
                        Prior_act_sv2hat        = var_v2hat;
                        Prior_act_v2hat(i,:)    = [Prior_act_mv2hat Prior_act_sv2hat];
                        Prior_act_Vnet(i,:)     = [Prior_act_mVnet Prior_act_sVnet];
                    else
                        if net.var_mode == 1
                            varW = var_mode(net.EEW2g,net.SEW2g,net.aOg,net.sv(1),net.sv(2));
                        else
                            varW = net.sv(1);
                        end
                    end
                    
                    zl(idxBatch, :)  = gather(reshape(0, [net.ny, numDataPerBatch])');
                    Szl(idxBatch, :) = gather(reshape(varW, [net.ny, numDataPerBatch])');
                end 
            end
%             sv = net.sv;  % for each epoch
        end
        function [thetaN, normStatN, thetaD, normStatD, zl, Szl] = regressionWithNI_test(netD, thetaD, normStatD, statesD, maxIdxD, netN, thetaN, normStatN, statesN, maxIdxN, x, y, layerConct)
            % Initialization
            numObs = size(x, 1);
            numDataPerBatch = netN.repBatchSize*netN.batchSize;
            zl  = zeros(numObs, netN.ny, netN.dtype);
            Szl = zeros(numObs, netN.ny, netN.dtype);
            % Loop
            loop = 0;
            for i = 1:numDataPerBatch:numObs
                loop     = loop + 1; 
                if numDataPerBatch==1
                    idxBatch = i:i+netN.batchSize-1;
                else
                    if numObs-i>=numDataPerBatch
                        idxBatch = i:i+numDataPerBatch-1;
                    else
                        idxBatch = [i:numObs, randperm(i-1, numDataPerBatch-numObs+i-1)];
                    end
                end
                % Covariate
                xloop  = reshape(x(idxBatch, :)', [netD.batchSize*netD.nx, netD.repBatchSize]);
                statesD = tagi.initializeInputs(statesD, xloop, [], [], [], [], [], [], [], [], netD.xsc);
                % Training
                if netN.trainMode == 1                  
                    % Feed forward
                    yloop = reshape(y(idxBatch, :)', [netD.batchSize*netD.nl, netD.repBatchSize]);                  
                    [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD); 
                    [mzD, SzD, maD, SaD, JD] = tagi.extractStates(statesD);
                    
                    statesN = tagi.initializeInputs(statesN, mzD{layerConct}, SzD{layerConct}, maD{layerConct}, SaD{layerConct}, JD{layerConct}, [], [], [], [], netN.xsc);
                    [statesN, normStatN, maxIdxN] = tagi.feedForwardPass(netN, thetaN, normStatN, statesN, maxIdxN); 
                    [~, ~, maN, SaN, JN] = tagi.extractStates(statesN);
                    
                    % Observation
                    [mv2a, Sv2a, Cv2a] = act.expFun(maN{end}, SaN{end}, netN.gpu);
                    
                    % Update noise
                    [deltaMzlD, deltaSzlD, deltaMv2z, deltaSv2z] = tagi.noiseUpdate4regression(SzD{end}, maD{end}, SaD{end}, JD{end}, JN{end}, mv2a, Sv2a, Cv2a, yloop, netN.sv, netN.gpu);
                                    
                    [deltaMzN, deltaSzN, deltaMxN, deltaSxN] = tagi.hiddenStateBackwardPass(netN, thetaN, normStatN, statesN, deltaMv2z, deltaSv2z, [], maxIdxN);
                    dthetaN = tagi.parameterBackwardPass(netN, thetaN, normStatN, statesN, deltaMzN, deltaSzN, deltaMxN, deltaSxN);
                    thetaN  = tagi.globalParameterUpdate(thetaN, dthetaN, netN.gpu);  
                    
                    [deltaMzD, deltaSzD, deltaMxD, deltaSxD] = tagi.hiddenStateBackwardPass(netD, thetaD, normStatD, statesD, deltaMzlD, deltaSzlD, [], maxIdxD);
                    dthetaD = tagi.parameterBackwardPass(netD, thetaD, normStatD, statesD, deltaMzD, deltaSzD, deltaMxD, deltaSxD);
                    thetaD  = tagi.globalParameterUpdate(thetaD, dthetaD, netD.gpu);
                    
                % Testing    
                else 
                    [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD); 
                    [mzD, SzD, maD, SaD, JD] = tagi.extractStates(statesD);
                    
                    statesN = tagi.initializeInputs(statesN, mzD{layerConct}, SzD{layerConct}, maD{layerConct}, SaD{layerConct}, JD{layerConct}, [], [], [], [], netN.xsc);
                    [statesN, normStatN, maxIdxN] = tagi.feedForwardPass(netN, thetaN, normStatN, statesN, maxIdxN); 
                    [~, ~, maN, SaN] = tagi.extractStates(statesD);
                    [mv2a, Sv2a] = act.expFun(maN{end}, SaN{end}, netN.gpu);
                    
                    zl(idxBatch, :)  = gather(reshape(maD{end}, [netD.ny, numDataPerBatch])');
                    Szl(idxBatch, :) = gather(reshape(SaD{end}+mv2a, [netD.ny, numDataPerBatch])');
                end 
            end
        end
        % Regression with separate networks for mean and v2hat
        function [thetaD, thetaW, normStatD, normStatW, zl, Szl, svresult_LL]    = regressionWithNI2(net, netD, netW, thetaD, normStatD, statesD, maxIdxD, thetaW, normStatW, statesW, maxIdxW, x, y)
            % Initialization
            numObs          = size(x, 1);
            numDataPerBatch = net.repBatchSize*net.batchSize;
            zl              = zeros(numObs, net.ny, net.dtype);                                               %BD
            Szl             = zeros(numObs, net.ny, net.dtype);                                               %BD
            Prior_act_v2hat = zeros(numObs, 2, net.dtype);                                                    %BD
            svresult_LL     = zeros(1,2,net.dtype);
            % Loop
            loop = 0;
            for i = 1:numDataPerBatch:numObs
                loop     = loop + 1; 
%                 idxBatch = i:i+net.batchSize-1;
                if numDataPerBatch==1
                    idxBatch = i:i+net.batchSize-1;
                else
                    if numObs-i>=numDataPerBatch
                        idxBatch = i:i+numDataPerBatch-1;
                    else
                        idxBatch = [i:numObs, randperm(i-1, numDataPerBatch-numObs+i-1)];
                    end
                end
                % Covariate
                xloop  = reshape(x(idxBatch, :)', [net.batchSize*net.nx, net.repBatchSize]);
                statesD = tagi.initializeInputs(statesD, xloop, [], [], [], [], [], [], [], [], netD.xsc);
                statesW = tagi.initializeInputs(statesW, xloop, [], [], [], [], [], [], [], [], netW.xsc);
                if net.trainMode == 1                  
                    % Observation
                    yloop = reshape(y(idxBatch, :)', [net.batchSize*net.nl, net.repBatchSize]);
                    if isnan(yloop)
                        check;
                    end
                    % Feed ForwardPass for mean
                    [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD);
                     [~, ~, maD, SaD, JD] = tagi.extractStates(statesD);
                    % % Feed ForwardPass for v2hat
                    [statesW, normStatW, maxIdxW] = tagi.feedForwardPass(netW, thetaW, normStatW, statesW, maxIdxW);
                     [~, ~, maW, SaW, JW] = tagi.extractStates(statesW);
                    % Noise parameters     BD
                    mla         = maD{end};
                    Slz         = SaD{end};
                    Sla         = SaD{end};
                    mv2a        = maW{end};
                    Sv2a        = SaW{end};
                    Jl          = JD{end};
                    Jv2         = JW{end};
                    
                    % Activate v2hat
                    [mv2a, Sv2a, Cv2a] = act.NoiseActFun(mv2a, Sv2a, netW.NoiseActFunIdx, net.gpu);
                    % Add baseline variance to heteroscedastic variance
                    v2hat_LL = net.v2hat_LL;
                    [v2hat_LL(1), v2hat_LL(2)] = act.NoiseActFun(v2hat_LL(1), v2hat_LL(2), net.NoiseActFun_LL_Idx, net.gpu);
                    if net.idx_LL_M == 1
                        E_v2hat            =  mv2a + v2hat_LL(1);
                        var_v2hat          =  Sv2a + v2hat_LL(2);
                    elseif net.idx_LL_M == 2
                        E_v2hat            = mv2a .* v2hat_LL(1);
                        var_v2hat          = Sv2a*v2hat_LL(2) + Sv2a*v2hat_LL(1)^2 + v2hat_LL(2)*mv2a^2;
                    else
                        E_v2hat            = mv2a;
                        var_v2hat          = Sv2a;
                    end
                    % Mode method
                    if net.var_mode == 1
                        varW = var_mode(net.EEW2g,net.SEW2g,net.aOg,E_v2hat,var_v2hat);
                    else
                        varW = E_v2hat;
                    end
                    SzF = Sla + varW + net.sv^2;
                    if isnan(1./SzF)
                        check;
                    end
                    % Update noise
                    [deltaMlz, deltaSlz, deltaMv2z, deltaSv2z, deltaMv2a, deltaSv2a, Kz, Kw, mv2hat_LL_pos, Sv2hat_LL_pos] = tagi.noiseUpdate4regression(Slz, mla, Sla, Jl, Jv2, mv2a, Sv2a, Cv2a, E_v2hat, var_v2hat, varW, v2hat_LL, yloop, net.sv, net.gpu, net.epoch, net.idx_LL_M, net.epochCV);
                    % Prior and Post for act_v2hat
                    Pos_act_mv2hat       = E_v2hat     + deltaMv2a;                                    %BD
                    Pos_act_sv2hat       = var_v2hat   + deltaSv2a;                                    %BD
                    % Convert from activation scale to original scale
                    Prior_act_mv2hat     = E_v2hat;
                    Prior_act_sv2hat     = var_v2hat;
                    Prior_act_v2hat(i,:) = [Prior_act_mv2hat Prior_act_sv2hat];                 %BD
                    Pos_act_v2hat(i,:)   = [Pos_act_mv2hat Pos_act_sv2hat];                     %BD
                    % Storing Kalman Gain for Z and W
                    K_Gain(i,1) = Kz;
                    K_Gain(i,2) = Kw;
                    % Updating layer-wise for Mean 
                    [deltaM_D, deltaS_D,deltaMx_D, deltaSx_D] = tagi.hiddenStateBackwardPass(netD, thetaD, normStatD, statesD, deltaMlz, deltaSlz, [], maxIdxD);
                    dthetaD = tagi.parameterBackwardPass(netD, thetaD, normStatD, statesD, deltaM_D, deltaS_D, deltaMx_D, deltaSx_D);
                    thetaD  = tagi.globalParameterUpdate(thetaD, dthetaD, net.gpu);
                    % Updating layer-wise for v2hat
                    [deltaM_W, deltaS_W,deltaMx_W, deltaSx_W] = tagi.hiddenStateBackwardPass(netW, thetaW, normStatW, statesW, deltaMv2z, deltaSv2z, [], maxIdxW);
                    dthetaW = tagi.parameterBackwardPass(netW, thetaW, normStatW, statesW, deltaM_W, deltaS_W, deltaMx_W, deltaSx_W);
                    thetaW  = tagi.globalParameterUpdate(thetaW, dthetaW, net.gpu);
                    % Storing output mean and variance                                                    
                    zl(idxBatch, :)      = mla;                                     %BD
                    Szl(idxBatch, :)     = Sla + varW;                              %BD
                    % Storing the pos v2hat_LL in network
                    net.v2hat_LL(1) = mv2hat_LL_pos;
                    net.v2hat_LL(2) = Sv2hat_LL_pos;
                %% Testing    
                else 
                    [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD);
                     [~, ~, maD, SaD] = tagi.extractStates(statesD);
                    % % Feed ForwardPass for v2hat
                    [statesW, normStatW, maxIdxW] = tagi.feedForwardPass(netW, thetaW, normStatW, statesW, maxIdxW);
                     [~, ~, maW, SaW] = tagi.extractStates(statesW);
                    % Noise parameters     BD
                    % Noise parameters     BD
                    mla         = maD{end};
                    Sla         = SaD{end};
                    mv2a        = maW{end};
                    Sv2a        = SaW{end};
                    
                    % Activate v2hat
                    [mv2a, Sv2a] = act.NoiseActFun(mv2a, Sv2a, netW.NoiseActFunIdx, net.gpu);
                    % Add baseline variance to heteroscedastic variance
                    v2hat_LL = net.v2hat_LL;
                    [v2hat_LL(1), v2hat_LL(2)] = act.NoiseActFun(v2hat_LL(1), v2hat_LL(2), net.NoiseActFun_LL_Idx, net.gpu);
                    if net.idx_LL_M == 1
                        E_v2hat            =  mv2a + v2hat_LL(1);
                        var_v2hat          =  Sv2a + v2hat_LL(2);
                    elseif net.idx_LL_M == 2
                        E_v2hat            = mv2a .* v2hat_LL(1);
                        var_v2hat          = Sv2a*v2hat_LL(2) + Sv2a*v2hat_LL(1)^2 + v2hat_LL(2)*mv2a^2;
                    else
                        E_v2hat            = mv2a;
                        var_v2hat          = Sv2a;
                    end
                    % Mode method
                    if net.var_mode == 1
                        varW = var_mode(net.EEW2g,net.SEW2g,net.aOg,E_v2hat,var_v2hat);
                    else
                        varW = E_v2hat;
                    end
                    % Output
                    zl(idxBatch, :)  = mla;                                      %BD
                    Szl(idxBatch, :) = Sla + varW;                               %BD
                end 
                svresult_LL = net.v2hat_LL;
            end
            
        
        end
        % Autoencoder
        function [thetaE, thetaD, normStatE, normStatD] = AE(netE, thetaE, normStatE, statesE, maxIdxE, netD, thetaD, normStatD, statesD, maxIdxD, x)
            % Initialization
            numObs = size(x, 4);
            numDataPerBatch = netE.batchSize*netE.repBatchSize;
            % Loop
            loop = 0;
            for i = 1:numDataPerBatch:numObs
                loop     = loop + 1;
                idxBatch = i:i+numDataPerBatch-1;
                xloop    = dp.dataLoader(x(:,:,:,idxBatch), netE.da, netE.trainMode);
                xloop    = reshape(xloop, [netE.batchSize*netE.nodes(1), netE.repBatchSize]);                                          
                yloop    = xloop;
                if netE.gpu == 1
                    yloop = gpuArray(yloop);
                    xloop = yloop;
                end               
                % Forward
                statesE                       = tagi.initializeInputs(statesE, xloop, [], [], [], [], [], [], [], [], netE.xsc);  
                [statesE, normStatE, maxIdxE] = tagi.feedForwardPass(netE, thetaE, normStatE, statesE, maxIdxE);
                [mzE, SzE, maE, SaE, JE,...
                    mdxsE, SdxsE, mxsE, SxsE] = tagi.extractStates(statesE);
                
                statesD                       = tagi.initializeInputs(statesD, mzE{end}, SzE{end}, maE{end}, SaE{end}, JE{end}, mdxsE{end}, SdxsE{end}, mxsE{end}, SxsE{end}, netE.xsc); 
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD);
                [~, ~, maD, SaD]              = tagi.extractStates(statesD);
                
                % Backward
                [deltaMD, deltaSD, deltaMxD,...
                    deltaSxD, deltaMz0D, deltaSz0D] = tagi.hiddenStateBackwardPass(netD, thetaD, normStatD, statesD, yloop, [], [], maxIdxD);
                deltaThetaD                         = tagi.parameterBackwardPass(netD, thetaD, normStatD, statesD, deltaMD, deltaSD, deltaMxD, deltaSxD);
                thetaD                              = tagi.globalParameterUpdate(thetaD, deltaThetaD, netD.gpu);   
                
                [deltaME, deltaSE,...
                    deltaMxE, deltaSxE] = tagi.hiddenStateBackwardPass(netE, thetaE, normStatE, statesE, deltaMz0D, deltaSz0D, [], maxIdxE);
                deltaThetaE             = tagi.parameterBackwardPass(netE, thetaE, normStatE, statesE, deltaME, deltaSE, deltaMxE, deltaSxE);
                thetaE                  = tagi.globalParameterUpdate(thetaE, deltaThetaE. netE.gpu);
                if netD.learnSv == 1
                    ml = tagi.detachMeanVar(maD{end}, netD.nl, netD.nv2, netD.batchSize);
                    Z  = gather(reshape(ml, [netD.nl, netD.batchSize])');
                else
                    Z = gather(reshape(maD{end}, [netD.ny, netD.batchSize*netD.repBatchSize])');
                end
                if any(isnan(SaD{end}(:,1)))||any(SaD{end}(:,1)<0)
                    check=1;
                end                           
            end
        end
        function [thetaE, thetaEM, thetaES, thetaD, normStatE, normStatEM, normStatES, normStatD] = AE_V2(netE, EMnet, ESnet, netD, thetaE, thetaEM, thetaES, thetaD, normStatE, normStatEM, normStatES, normStatD, x, Sx)
            % Initialization
            numObs = size(x, 4);
            % Loop
            loop     = 0;
            time_tot = 0;
            for i = 1:netE.batchSize:numObs
                timeval  = tic;
                loop     = loop + 1;
                idxBatch = i:i+netE.batchSize-1;
                xloop    = dp.dataLoader(x(:,:,:,idxBatch), netE.da, netE.trainMode);                
                yloop    = xloop;
                if netE.gpu == 1
                    yloop = gpuArray(yloop);
                    xloop = yloop;
                end
                % Training
                [mzE, SzE, maE, SalE, JE, mdxsE, SdxsE, mxsE, SxsE, maxIdxE, normStatE] = tagi.feedForwardPass(netE, xloop, Sx, thetaE, normStatE);
                [mzEM, SzEM, maEM, SalEM, JEM, mdxsEM, SdxsEM, mxsEM, SxsEM, maxIdxEM, normStatEM] = tagi.feedForwardPass(EMnet, mzE{end}, SzE{end}, thetaEM, normStatEM);
                [mzES, SzES, maES, SalES, JES, mdxsES, SdxsES, mxsES, SxsES, maxIdxES, normStatES] = tagi.feedForwardPass(ESnet, mzE{end}, SzE{end}, thetaES, normStatES);
                mlE   = maEM{end};
                SlE   = SalEM;
                mlzE  = mzEM{end};
                SlzE  = SzEM{end};
                JlE   = JEM{end};
                mv2zE = mzES{end};
                Sv2zE = SzES{end};
                mv2aE = maES{end};
                Sv2aE = SalES;
                Jv2E  = JES{end};
                % Activate log(\sigma_v2)
%                 Cv2aE = Jv2E.*Sv2zE;
                [mv2aE, Sv2aE, Cv2aE] = act.expFun(mv2aE, Sv2aE, ESnet.gpu);               
                [mzD, SzD, maD, SalD, JD, mdxsD, SdxsD, mxsD, SxsD, maxIdxD, normStatD] = tagi.feedForwardPass(netD, mzEM{end}, SzEM{end}+mv2aE+EMnet.sv^2, thetaD, normStatD);                                          
                [thetaD, mzfD, SzfD] = tagi.feedBackward(netD, thetaD, mzD, SzD, maD, SalD, JD, mdxsD, SdxsD, mxsD, SxsD, yloop, [], [], maxIdxD, normStatD);
               
                [mlzE, SlzE, mv2zE, Sv2zE] = tagi.noiseUpdate4encoding(mlzE, SlzE, mlE, SlE, JlE, mv2zE, Sv2zE, Jv2E, mv2aE, Sv2aE, Cv2aE, mzfD, SzfD, netE.gpu);
                [thetaEM, mzfEM, SzfEM] = tagi.feedBackward(EMnet, thetaEM, mzEM, SzEM, maEM, SalEM, JEM, mdxsEM, SdxsEM, mxsEM, SxsEM, mlzE, SlzE, [], maxIdxEM, normStatEM);
                [thetaES, mzfES, SzfES] = tagi.feedBackward(ESnet, thetaES, mzES, SzES, maES, SalES, JES, mdxsES, SdxsES, mxsES, SxsES, mv2zE, Sv2zE, [], maxIdxES, normStatES);
                mzlE = mzfEM + mzfES - mzE{end};
                SzlE = SzfEM + SzfES - SzE{end};  
                thetaE = tagi.feedBackward(netE, thetaE, mzE, SzE, maE, SalE, JE, mdxsE, SdxsE, mxsE, SxsE, mzlE, SzlE, [], maxIdxE, normStatE);
                Z = gather(reshape(maD{end}, [netD.ny, netD.batchSize])');
                if any(isnan(SzD{end}))||any(SzD{end}<0)
                    check=1;
                end
                time_loop = toc(timeval);
                time_tot  = time_tot + time_loop;
                time_rem  = double(time_tot)/(double(idxBatch(end)))*(numObs-double(idxBatch(end)))/60;
                if mod(idxBatch(end), netD.obsShow) == 0 && i > 1 && netD.trainMode == 1 && netD.displayMode == 1
                    disp(['     Time left : ' sprintf('%0.2f',time_rem) ' mins'])
                end                             
            end
        end       
        function [thetaE, thetaD, thetaC, normStatE, normStatD, normStatC, Pn, er] = SSAE(netE, netD, Cnet, thetaE, thetaD, thetaC, normStatE, normStatD, normStatC, x, Sx, xlabelled, Sxlabelled, ylabelled, updateIdx)
            % Initialization
            numObs = size(x, 4);
            if Cnet.errorRateEval == 1
                Pn = zeros(numObs, Cnet.numClasses, Cnet.dtype);
                er = zeros(numObs, 1, Cnet.dtype);
                [classObs, classIdx] = dp.class_encoding(Cnet.numClasses);
                addIdx   = reshape(repmat(colon(0, Cnet.ny, (Cnet.batchSize-1)*Cnet.ny), [Cnet.numClasses, 1]), [Cnet.numClasses*Cnet.batchSize, 1]);
                classObs = repmat(classObs, [Cnet.batchSize, 1]);
                classIdx = repmat(classIdx, [Cnet.batchSize, 1]) + cast(addIdx, class(classIdx));
            else
                er = nan;
                Pn = nan;
            end
            % Loop
            loop     = 0;
            time_tot = 0;
            for i = 1:netE.batchSize:numObs
                timeval  = tic;
                loop     = loop + 1;
                idxBatch = i:i+netE.batchSize-1;
                xloop    = dp.dataLoader(x(:,:,:,idxBatch), netE.da, netE.trainMode);
                if netE.gpu == 1
                    xloop = gpuArray(xloop);
                end
                yxloop    = xloop;
                % Training
                if netE.trainMode == 1
                    [xloopS, SxloopS, yS, updateIdxS] = network.generateRealSamples(xlabelled, Sxlabelled, ylabelled, updateIdx, Cnet.batchSize);
                    xloopS = dp.dataLoader(xloopS, netE.da, netE.trainMode);
                    yloop = reshape(yS', [length(idxBatch)*Cnet.ny, 1]);
                    updateIdxLoop = dp.selectIndices(updateIdxS, Cnet.batchSize, Cnet.ny, Cnet.dtype);
                    % Supervised samples
                    [mzE, SzE, maE, SalE, JE, mdxsE, SdxsE, mxsE, SxsE, maxIdxE, normStatE] = tagi.feedForwardPass(netE, xloopS, SxloopS, thetaE, normStatE);
                    [mzC, SzC, maC, SalC, JC, mdxsC, SdxsC, mxsC, SxsC, maxIdxC, normStatC] = tagi.feedForwardPass(Cnet, mzE{end}, SzE{end}, thetaC, normStatC);
                    [thetaC, zfC, SzfC] = tagi.feedBackward(Cnet, thetaC, mzC, SzC, maC, SalC, JC, mdxsC, SdxsC, mxsC, SxsC, yloop, SxloopS, updateIdxLoop, maxIdxC, normStatC);
                    [thetaE, ~, ~] = tagi.feedBackward(netE, thetaE, mzE, SzE, maE, SalE, JE, mdxsE, SdxsE, mxsE, SxsE, zfC, SzfC, [], maxIdxE, normStatE);
                    % Unsupervised samples
                    [mzE, SzE, maE, SalE, JE, mdxsE, SdxsE, mxsE, SxsE, maxIdxE, normStatE] = tagi.feedForwardPass(netE, xloop, Sx, thetaE, normStatE);
                    [~, ~, maC, SalC] = tagi.feedForwardPass(Cnet, mzE{end}, SzE{end}, thetaC, normStatC);
                    [mzD, SzD, maD, SalD, JD, mdxsD, SdxsD, mxsD, SxsD, maxIdxD, normStatD] = tagi.feedForwardPass(netD, mzE{end}, SzE{end}, thetaD, normStatD);
                    
                    [thetaD, zfD, SzfD] = tagi.feedBackward(netD, thetaD, mzD, SzD, maD, SalD, JD, mdxsD, SdxsD, mxsD, SxsD, yxloop, Sx, [], maxIdxD, normStatD);
                    [thetaE, ~, ~] = tagi.feedBackward(netE, thetaE, mzE, SzE, maE, SalE, JE, mdxsE, SdxsE, mxsE, SxsE, zfD, SzfD, [], maxIdxE, normStatE);
                    Z    = gather(reshape(maD{end}, [netD.ny, length(idxBatch)])');
                else
                    [mzE, SzE] = tagi.feedForwardPass(netE, xloop, Sx, thetaE, normStatE);
                    [~, ~, maC, SalC] = tagi.feedForwardPass(Cnet, mzE{end}, SzE{end}, thetaC, normStatC);
                end
                if i >1000
                    check=1;
                end
                if any(isnan(SalC))||any(SalC<0)
                    check=1;
                end
                time_loop = toc(timeval);
                time_tot  = time_tot + time_loop;
                time_rem  = double(time_tot)/(double(idxBatch(end)))*(numObs-double(idxBatch(end)))/60;
                if Cnet.errorRateEval == 1
                    P = dp.obs2class(maC{end}, SalC + Cnet.sv.^2, classObs, classIdx);
                    P = reshape(P, [Cnet.numClasses, Cnet.batchSize])';
                    P = gather(P);
                    if any(any(isnan(P)))
                        check=1;
                    end
                    Pn(idxBatch, :) = P;
                    er(idxBatch, :) = mt.errorRate(Cnet.labels(idxBatch, :)', P');
                    % Display error rate
                    if mod(idxBatch(end), Cnet.obsShow) == 0 && i > 1 && Cnet.trainMode == 1 && Cnet.displayMode == 1
                        disp(['     Error Rate : ' sprintf('%0.2f', 100*mean(er(max(1,i-200):i))) '%']);
                        disp(['     Time left  : ' sprintf('%0.2f',time_rem) ' mins'])
                    end
                else
                    if mod(idxBatch(end), Cnet.obsShow) == 0 && i > 1 && Cnet.trainMode == 1 && Cnet.displayMode == 1
                        disp(['     Time left : ' sprintf('%0.2f',time_rem) ' mins'])
                    end
                end
            end
        end
        
        % GAN
        function [thetaD, thetaG, normStatD, normStatG] = GAN(netD, thetaD, normStatD, statesD, maxIdxD, netG, thetaG, normStatG, statesG, maxIdxG, xD)
            % Initialization
            numObs = size(xD, 4);
            numDataPerBatch = netD.batchSize*netD.repBatchSize;
            if netD.gpu
                zeroPad = zeros(1, 1, netD.dtype, 'gpuArray');
            else
                zeroPad = zeros(1, 1, netD.dtype);
            end
            % Loop
            loop     = 0;
            for i = 1:numDataPerBatch:numObs
                loop     = loop + 1;
                idxBatch = i:i+numDataPerBatch-1;
                xloopG   = reshape(randn(netG.nx, netG.batchSize*netG.repBatchSize, 'like', zeroPad), [netG.nx*netG.batchSize, netG.repBatchSize]);
                xloop    = dp.dataLoader(xD(:,:,:,idxBatch), netD.da, netD.batchSize, netD.repBatchSize, netD.trainMode); 
                xloop    = reshape(xloop, [netD.batchSize*netD.nodes(1), netD.repBatchSize]);  
                
                yfake    = ones(netD.batchSize, netD.repBatchSize, 'like', zeroPad);
                yreal    = -ones(netD.batchSize, netD.repBatchSize, 'like', zeroPad);
                
                % Update dicriminator (netD)
                    % Real example
                statesD                       = tagi.initializeInputs(statesD, xloop, [], [], [], [], [], [], [], [], netD.xsc);  
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD); 
                
                [deltaMD, deltaSD,...
                    deltaMxD, deltaSxD] = tagi.hiddenStateBackwardPass(netD, thetaD, normStatD, statesD, yreal, [], [], maxIdxD);
                deltaThetaD             = tagi.parameterBackwardPass(netD, thetaD, normStatD, statesD, deltaMD, deltaSD, deltaMxD, deltaSxD);
                thetaD                  = tagi.globalParameterUpdate(thetaD, deltaThetaD);  
                    % Fake examples
                statesG                       = tagi.initializeInputs(statesG, xloopG, [], [], [], [], [], [], [], [], netG.xsc);  
                [statesG, normStatG, maxIdxG] = tagi.feedForwardPass(netG, thetaG, normStatG, statesG, maxIdxG);
                [mzG, SzG, maG, SaG, JG,...
                    mdxsG, SdxsG, mxsG, SxsG] = tagi.extractStates(statesG);
                statesD                       = tagi.initializeInputs(statesD, mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end}, mdxsG{end}, SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc); 
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD);
                
                [deltaMD, deltaSD,...
                    deltaMxD, deltaSxD] = tagi.hiddenStateBackwardPass(netD, thetaD, normStatD, statesD, yfake, [], [], maxIdxD);
                deltaThetaD             = tagi.parameterBackwardPass(netD, thetaD, normStatD, statesD, deltaMD, deltaSD, deltaMxD, deltaSxD);
                thetaD                  = tagi.globalParameterUpdate(thetaD, deltaThetaD); 
                
                % Update generator (netG)
                statesD                       = tagi.initializeInputs(statesD, mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end}, mdxsG{end}, SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc); 
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD); 
                [~, ~, ~, ~,...
                    deltaMz0D, deltaSz0D]     = tagi.hiddenStateBackwardPass(netD, thetaD, normStatD, statesD, yreal, [], [], maxIdxD);
                
                [deltaMG, deltaSG,...
                    deltaMxG, deltaSxG] = tagi.hiddenStateBackwardPass(netG, thetaG, normStatG, statesG, deltaMz0D, deltaSz0D, [], maxIdxG);
                deltaThetaG             = tagi.parameterBackwardPass(netG, thetaG, normStatG, statesG, deltaMG, deltaSG, deltaMxG, deltaSxG);
                thetaG                  = tagi.globalParameterUpdate(thetaG, deltaThetaG); 
                
                Z  = gather(reshape(maG{end}, [netG.ny, length(idxBatch)])');
                if any(isnan(SzG{end}(:, 1)))||any(SzG{end}(:, 1)<0)
                    check=1;
                end                          
            end
        end        
        function [thetaD, thetaG, thetaC, thetaP, normStatD, normStatG, normStatC, normStatP, Pn, er] = SSGAN(netD, netG, Cnet, netP, thetaD, thetaG, thetaC, thetaP, normStatD, normStatG, normStatC, normStatP, x, Sx, xlabelled, Sxlabelled, ylabelled, updateIdx)
            % Initialization
            numObs = size(x, 4);
            if Cnet.errorRateEval == 1
                Pn = zeros(numObs, Cnet.numClasses, Cnet.dtype);
                er = zeros(numObs, 1, Cnet.dtype);
                [classObs, classIdx] = dp.class_encoding(Cnet.numClasses);
                addIdx   = reshape(repmat(colon(0, Cnet.ny, (Cnet.batchSize-1)*Cnet.ny), [Cnet.numClasses, 1]), [Cnet.numClasses*Cnet.batchSize, 1]);
                classObs = repmat(classObs, [Cnet.batchSize, 1]);
                classIdx = repmat(classIdx, [Cnet.batchSize, 1]) + cast(addIdx, class(classIdx));
            else
                er = nan;
                Pn = nan;
            end
            % Loop
            loop     = 0;
            time_tot = 0;
            for i = 1:netD.batchSize:numObs
                timeval  = tic;
                loop     = loop + 1;
                idxBatch = i:i+netD.batchSize-1;
                xloopG   = reshape(randn(netG.batchSize, netG.nx, 'like', x)', [netG.nx*netG.batchSize, 1]);
                xloop    = dp.dataLoader(x(:,:,:,idxBatch), netD.da, netD.trainMode);
                if netD.gpu == 1
                    xloop  = gpuArray(xloop);
                    xloopG = gpuArray(xloopG);
                end
                % Training
                if netD.trainMode == 1
                    % Create observation for unsupervised discriminator 
                    yfake = ones(netD.batchSize, 1, 'like', xloop);
                    yreal = -ones(netD.batchSize, 1, 'like', xloop);
                    % Load the real data for supervised discriminator
                    [xloopS, SxloopS, yS, updateIdxS] = network.generateRealSamples(xlabelled, Sxlabelled, ylabelled, updateIdx, Cnet.batchSize);
                    xloopS  = dp.dataLoader(xloopS, netD.da, netD.trainMode);
                    yloop   = reshape(yS', [length(idxBatch)*Cnet.ny, 1]);
                    updateIdxLoop = dp.selectIndices(updateIdxS, Cnet.batchSize, Cnet.ny, Cnet.dtype);
                    % Update supervised discriminator (netD and Cnent)
                    [mzD, SzD, maD, SalD, JD, mdxsD, SdxsD, mxsD, SxsD, maxIdxD, normStatD] = tagi.feedForwardPass(netD, xloopS, SxloopS, thetaD, normStatD);
                    [mzC, SzC, maC, SalC, JC, mdxsC, SdxsC, mxsC, SxsC, maxIdxC, normStatC] = tagi.feedForwardPass(Cnet, mzD{end}, SzD{end}, thetaC, normStatC);
                    [thetaC, zfC, SzfC] = tagi.feedBackward(Cnet, thetaC, mzC, SzC, maC, SalC, JC, mdxsC, SdxsC, mxsC, SxsC, yloop, SxloopS, updateIdxLoop, maxIdxC, normStatC);
                    [thetaD, ~, ~] = tagi.feedBackward(netD, thetaD, mzD, SzD, maD, SalD, JD, mdxsD, SdxsD, mxsD, SxsD, zfC, SzfC, [], maxIdxD, normStatD);
                    % Update unsupervised discriminator (netD and Pnent)
                        % Fake samples
                    [mzG, SzG, maG, SaG, JG, mdxsG, SdxsG, mxsG, SxsG, maxIdxG, normStatG] = tagi.feedForwardPass(netG, xloopG, [], thetaG, normStatG);    
                    [mzD, SzD, maD, SaD, JD, mdxsD, SdxsD, mxsD, SxsD, maxIdxD, normStatD]  = tagi.feedForwardPass(netD, mzG{end}, SzG{end}, thetaD, normStatD);  
                    [mzP, SzP, maP, SalP, JP, mdxsP, SdxsP, mxsP, SxsP, maxIdxP, normStatP] = tagi.feedForwardPass(netP, mzD{end}, SzD{end}, thetaP, normStatP);
                    [thetaP, zfP, SzfP] = tagi.feedBackward(netP, thetaP, mzP, SzP, maP, SalP, JP, mdxsP, SdxsP, mxsP, SxsP, yfake, [], [], maxIdxP, normStatP);
                    [thetaD, ~, ~]      = tagi.feedBackward(netD, thetaD, mzD, SzD, maD, SaD, JD, mdxsD, SdxsD, mxsD, SxsD, zfP, SzfP, [], maxIdxD, normStatD); 
                        % Real Samples
                    [mzD, SzD, maD, SaD, JD, mdxsD, SdxsD, mxsD, SxsD, maxIdxD, normStatD]  = tagi.feedForwardPass(netD, xloop, Sx, thetaD, normStatD);
                    [~, ~, maC, SalC]   = tagi.feedForwardPass(Cnet, mzD{end}, SzD{end}, thetaC, normStatC);
                    [mzP, SzP, maP, SalP, JP, mdxsP, SdxsP, mxsP, SxsP, maxIdxP, normStatP] = tagi.feedForwardPass(netP, mzD{end}, SzD{end}, thetaP, normStatP);
                    [thetaP, zfP, SzfP] = tagi.feedBackward(netP, thetaP, mzP, SzP, maP, SalP, JP, mdxsP, SdxsP, mxsP, SxsP, yreal, [], [], maxIdxP, normStatP);
                    [thetaD, ~, ~]      = tagi.feedBackward(netD, thetaD, mzD, SzD, maD, SaD, JD, mdxsD, SdxsD, mxsD, SxsD, zfP, SzfP, [], maxIdxD, normStatD);                    
                    % Update generator (Gnent)
                    [mzD, SzD, maD, SaD, JD, mdxsD, SdxsD, mxsD, SxsD, maxIdxD, normStatD]  = tagi.feedForwardPass(netD, mzG{end}, SzG{end}, thetaD, normStatD); 
                    [mzP, SzP, maP, SalP, JP, mdxsP, SdxsP, mxsP, SxsP, maxIdxP, normStatP] = tagi.feedForwardPass(netP, mzD{end}, SzD{end}, thetaP, normStatP);                    
                    [~, zfP, SzfP] = tagi.feedBackward(netP, thetaP, mzP, SzP, maP, SalP, JP, mdxsP, SdxsP, mxsP, SxsP, yreal, [], [], maxIdxP, normStatP);
                    [~, zfD, SzfD] = tagi.feedBackward(netD, thetaD, mzD, SzD, maD, SaD, JD, mdxsD, SdxsD, mxsD, SxsD, zfP, SzfP, [], maxIdxD, normStatD); 
                    [thetaG, ~, ~] = tagi.feedBackward(netG, thetaG, mzG, SzG, maG, SaG, JG, mdxsG, SdxsG, mxsG, SxsG, zfD, SzfD, [], maxIdxG, normStatG); 
                    Z = gather(reshape(maG{end}, [netG.ny, length(idxBatch)])');
                else
                    [mzD, SzD] = tagi.feedForwardPass(netD, xloop, Sx, thetaD, normStatD);
                    [~, ~, maC, SalC] = tagi.feedForwardPass(Cnet, mzD{end}, SzD{end}, thetaC, normStatC);
                end
                if i >1000
                    check=1;
                end
                if any(isnan(SalC))||any(SalC<0)
                    check=1;
                end
                time_loop = toc(timeval);
                time_tot  = time_tot + time_loop;
                time_rem  = double(time_tot)/(double(idxBatch(end)))*(numObs-double(idxBatch(end)))/60;
                if Cnet.errorRateEval == 1
                    P = dp.obs2class(maC{end}, SalC + Cnet.sv.^2, classObs, classIdx);
                    P = reshape(P, [Cnet.numClasses, Cnet.batchSize])';
                    P = gather(P);
                    if any(any(isnan(P)))
                        check=1;
                    end
                    Pn(idxBatch, :) = P;
                    er(idxBatch, :) = mt.errorRate(Cnet.labels(idxBatch, :)', P');
                    % Display error rate
                    if mod(idxBatch(end), Cnet.obsShow) == 0 && i > 1 && Cnet.trainMode == 1 && Cnet.displayMode == 1
                        disp(['     Error Rate : ' sprintf('%0.2f', 100*mean(er(max(1,i-200):i))) '%']);
                        disp(['     Time left  : ' sprintf('%0.2f',time_rem) ' mins'])
                    end
                else
                    if mod(idxBatch(end), Cnet.obsShow) == 0 && i > 1 && Cnet.trainMode == 1 && Cnet.displayMode == 1
                        disp(['     Time left : ' sprintf('%0.2f',time_rem) ' mins'])
                    end
                end
            end
        end
         
        % infoGAN
        function [thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG, normStatQ, normStatP] = infoGAN(netD, thetaD, normStatD, statesD, maxIdxD, ...
                netG, thetaG, normStatG, statesG, maxIdxG, netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netP, thetaP, normStatP, statesP, maxIdxP, xD)
            % Initialization
            numObs = size(xD, 4);
            numDataPerBatch = netD.batchSize*netD.repBatchSize;
            % Loop
            loop     = 0;
            for i = 1:numDataPerBatch:numObs
                loop     = loop + 1;
                idxBatch = i:i+numDataPerBatch-1;
                
                % Form a vector including random noise + categorical
                % variables + continuous variables
                [xcc, udIdxQ, xloopG] = network.generateLatentVar(netQ.numClasses, netQ.numCatVar, netQ.numContVar, numDataPerBatch, netG.nx, netQ.dtype, netQ.gpu); 
                xcc    = reshape(xcc, [numel(xcc)/(netQ.repBatchSize), netQ.repBatchSize]);
                udIdxQ = reshape(udIdxQ, [numel(udIdxQ)/(netQ.repBatchSize), netQ.repBatchSize]);
                xloopG = reshape(xloopG, [numel(xloopG)/(netQ.repBatchSize), netQ.repBatchSize]);
                
                % Real image pixels
                xloop  = dp.dataLoader(xD(:,:,:,idxBatch), netD.da, netD.batchSize, netD.repBatchSize, netD.trainMode); 
                xloop  = reshape(xloop, [netD.batchSize*netD.nodes(1), netD.repBatchSize]);  
                if netD.gpu==1||netG.gpu==1||netQ.gpu==1||netP.gpu==1
                    xloop  = gpuArray(xloop);
                end
                yfake = ones(netD.batchSize, netD.repBatchSize, 'like', xloop);
                yreal = -ones(netD.batchSize, netD.repBatchSize, 'like', xloop);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Update discriminator (netD, netP, and netQ)
                statesD                   = tagi.initializeInputs(statesD, xloop, [], [], [], [], [], [], [], [], netD.xsc);
                [thetaD, thetaP,...
                    normStatD, normStatP] = network.updateDPinfoGAN(netD, thetaD, normStatD, statesD, maxIdxD, ...
                netP, thetaP, normStatP, statesP, maxIdxP, yreal);
            
                % Generate fake examples using netG
                statesG                       = tagi.initializeInputs(statesG, xloopG, [], [], [], [], [], [], [], [], netG.xsc);  
                [statesG, normStatG, maxIdxG] = tagi.feedForwardPass(netG, thetaG, normStatG, statesG, maxIdxG);
                [mzG, SzG, maG, SaG, JG,...
                    mdxsG, SdxsG, mxsG, SxsG] = tagi.extractStates(statesG);
                
                % Feed fake examples to netD
                statesD                              = tagi.initializeInputs(statesD, mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end}, mdxsG{end}, SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc);  
                [thetaD, thetaP, thetaQ,...
                    normStatD, normStatP, normStatQ] = network.updateDPQacGAN(netD, thetaD, normStatD, statesD, maxIdxD, ...
                netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netP, thetaP, normStatP, statesP, maxIdxP, yfake, xcc, udIdxQ);         
                                   
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%% Update generator (netG)
                % Feed fake examples to netD
                statesD                       = tagi.initializeInputs(statesD, mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end}, mdxsG{end}, SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc);  
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD);               
                [mzD, SzD, maD, SaD, JD,...
                    mdxsD, SdxsD, mxsD, SxsD] = tagi.extractStates(statesD);
                
                % Feed netD's outputs to netP
                statesP                       = tagi.initializeInputs(statesP, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netP.xsc);  
                [statesP, normStatP, maxIdxP] = tagi.feedForwardPass(netP, thetaP, normStatP, statesP, maxIdxP);
                
                % Feed netD's output to the 2nd head (netQ) that infers
                % latent variables
                statesQ                       = tagi.initializeInputs(statesQ, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netQ.xsc);  
                [statesQ, normStatQ, maxIdxQ] = tagi.feedForwardPass(netQ, thetaQ, normStatQ, statesQ, maxIdxQ);       
                
                % Update hidden states for netP 
                [~, ~, ~, ~, deltaMz0P, deltaSz0P] = tagi.hiddenStateBackwardPass(netP, thetaP, normStatP, statesP, yreal, [], [], maxIdxP);
                % Update parameters & hidden states for netQ
                [~, ~, ~, ~, deltaMz0Q, deltaSz0Q] = tagi.hiddenStateBackwardPass(netQ, thetaQ, normStatQ, statesQ, xcc, [], udIdxQ, maxIdxQ);
                
                deltaMzLD = deltaMz0P + deltaMz0Q;
                deltaSzLD = deltaSz0P + deltaSz0Q;
                [~, ~, ~, ~, deltaMz0D, deltaSz0D] = tagi.hiddenStateBackwardPass(netD, thetaD, normStatD, statesD, deltaMzLD, deltaSzLD, [], maxIdxD);   
                
                % Update parameters & hidden states for netG
                [deltaMG, deltaSG,...
                    deltaMxG, deltaSxG] = tagi.hiddenStateBackwardPass(netG, thetaG, normStatG, statesG, deltaMz0D, deltaSz0D, [], maxIdxG);
                deltaThetaG             = tagi.parameterBackwardPass(netG, thetaG, normStatG, statesG, deltaMG, deltaSG, deltaMxG, deltaSxG);
                thetaG                  = tagi.globalParameterUpdate(thetaG, deltaThetaG); 

                Z  = gather(reshape(maG{end}, [netG.ny, netG.batchSize*netG.repBatchSize])');               
                if any(isnan(SzG{end}(:, 1)))||any(SzG{end}(:, 1)<0)
                    check=1;
                end
            end
        end
        function [thetaD, thetaG, thetaQ, thetaQc, thetaP, normStatD, normStatG, normStatQ, normStatQc, normStatP] = infoGAN_V2(netD, thetaD, normStatD, statesD, maxIdxD,...
                netG, thetaG, normStatG, statesG, maxIdxG, netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netQc, thetaQc, normStatQc, statesQc, maxIdxQc, netP, thetaP, normStatP, statesP, maxIdxP, xD)
            % Initialization
            numObs          = size(xD, 4);
            numDataPerBatch = netD.batchSize*netD.repBatchSize;
            % Loop
            loop     = 0;
            for i = 1:numDataPerBatch:numObs
                loop     = loop + 1;
                idxBatch = i:i+numDataPerBatch-1;
                
                % Form a vector including random noise + categorical
                % variables + continuous variables
                [xd, xc, xloopG, udIdxQ] = network.generateLatentVar_V2(netQ.numClasses, netQ.numCatVar, netQc.numContVar, numDataPerBatch, netG.nx, netQ.dtype, netQ.gpu); 
                xd     = reshape(xd, [numel(xd)/(netQ.repBatchSize), netQ.repBatchSize]);
                xc     = reshape(xc, [numel(xc)/(netQc.repBatchSize), netQc.repBatchSize]);                
                udIdxQ = reshape(udIdxQ, [numel(udIdxQ)/(netQ.repBatchSize), netQ.repBatchSize]);
                xloopG = reshape(xloopG, [numel(xloopG)/(netQ.repBatchSize), netQ.repBatchSize]);
                
                % Real image pixels
                xloop  = dp.dataLoader(xD(:,:,:,idxBatch), netD.da, netD.batchSize, netD.repBatchSize, netD.trainMode); 
                xloop  = reshape(xloop, [netD.batchSize*netD.nodes(1), netD.repBatchSize]);  
                if netD.gpu==1||netG.gpu==1||netQ.gpu==1||netP.gpu==1
                    xloop  = gpuArray(xloop);
                end
                yfake = ones(netD.batchSize, netD.repBatchSize, 'like', xloop);
                yreal = -ones(netD.batchSize, netD.repBatchSize, 'like', xloop);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Update discriminator (netD, netP, and netQ)
                statesD                               = tagi.initializeInputs(statesD, xloop, [], [], [], [], [], [], [], [], netD.xsc);
                [thetaD, thetaP,normStatD, normStatP] = network.updateDPinfoGAN(netD, thetaD, normStatD, statesD, maxIdxD, netP, thetaP, normStatP, statesP, maxIdxP, yreal);
            
                % Generate fake examples using netG
                statesG                       = tagi.initializeInputs(statesG, xloopG, [], [], [], [], [], [], [], [], netG.xsc);  
                [statesG, normStatG, maxIdxG] = tagi.feedForwardPass(netG, thetaG, normStatG, statesG, maxIdxG);
                [mzG, SzG, maG, SaG, JG,...
                    mdxsG, SdxsG, mxsG, SxsG] = tagi.extractStates(statesG);
                
                % Feed fake examples to netD
                statesD = tagi.initializeInputs(statesD, mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end}, mdxsG{end}, SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc);  
                [thetaD, thetaQ, thetaQc, thetaP, normStatD, normStatQ, normStatQc, normStatP] = network.updateDPQinfoGAN(netD, thetaD, normStatD, statesD, maxIdxD, ...
                    netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netQc, thetaQc, normStatQc, statesQc, maxIdxQc, netP, thetaP, normStatP, statesP, maxIdxP, xd, xc, yfake, udIdxQ);
                                   
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%% Update generator (netG)
                % Feed fake examples to netD
                statesD                       = tagi.initializeInputs(statesD, mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end}, mdxsG{end}, SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc);  
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD);               
                [mzD, SzD, maD, SaD, JD,...
                    mdxsD, SdxsD, mxsD, SxsD] = tagi.extractStates(statesD);
                
                % Feed netD's outputs to netP
                statesP                       = tagi.initializeInputs(statesP, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netP.xsc);  
                [statesP, normStatP, maxIdxP] = tagi.feedForwardPass(netP, thetaP, normStatP, statesP, maxIdxP);
                
                % Feed netD's output to the 2nd head (netQ) that infers
                % latent variables
                statesQ                       = tagi.initializeInputs(statesQ, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netQ.xsc);  
                [statesQ, normStatQ, maxIdxQ] = tagi.feedForwardPass(netQ, thetaQ, normStatQ, statesQ, maxIdxQ);   
                
                % Feed netD's output to the 3rd head (netQc) that infers
                % continuous latent variables
                statesQc                         = tagi.initializeInputs(statesQc, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netQc.xsc);
                [statesQc, normStatQc, maxIdxQc] = tagi.feedForwardPass(netQc, thetaQc, normStatQc, statesQc, maxIdxQc);
            
                % Update hidden states for netP 
                [~, ~, ~, ~, deltaMz0P, deltaSz0P] = tagi.hiddenStateBackwardPass(netP, thetaP, normStatP, statesP, yreal, [], [], maxIdxP);
                % Update parameters & hidden states for netQ
                [~, ~, ~, ~, deltaMz0Q, deltaSz0Q] = tagi.hiddenStateBackwardPass(netQ, thetaQ, normStatQ, statesQ, xd, [], udIdxQ, maxIdxQ);
                % Update parameters & hidden states for netQ
                [~, ~, ~, ~, deltaMz0Qc, deltaSz0Qc] = tagi.hiddenStateBackwardPass(netQc, thetaQc, normStatQc, statesQc, xc, [], [], maxIdxQ);
                
                deltaMzLD = deltaMz0P + deltaMz0Q + deltaMz0Qc;
                deltaSzLD = deltaSz0P + deltaSz0Q + deltaSz0Qc;
                [~, ~, ~, ~, deltaMz0D, deltaSz0D] = tagi.hiddenStateBackwardPass(netD, thetaD, normStatD, statesD, deltaMzLD, deltaSzLD, [], maxIdxD);                
                % Update parameters & hidden states for netG
                [deltaMG, deltaSG,...
                    deltaMxG, deltaSxG] = tagi.hiddenStateBackwardPass(netG, thetaG, normStatG, statesG, deltaMz0D, deltaSz0D, [], maxIdxG);
                deltaThetaG             = tagi.parameterBackwardPass(netG, thetaG, normStatG, statesG, deltaMG, deltaSG, deltaMxG, deltaSxG);
                thetaG                  = tagi.globalParameterUpdate(thetaG, deltaThetaG); 

                Z  = gather(reshape(maG{end}, [netG.ny, netG.batchSize*netG.repBatchSize])');               
                if any(isnan(SzG{end}(:, 1)))||any(SzG{end}(:, 1)<0)
                    check=1;
                end
            end
        end
        
        % ACGAN
        function [thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG, normStatQ, normStatP] = ACGAN(netD, thetaD, normStatD, statesD, maxIdxD, ...
                netG, thetaG, normStatG, statesG, maxIdxG, netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netP, thetaP, normStatP, statesP, maxIdxP, x, y, udIdx)
            % Initialization
            numObs = size(x, 4);
            numDataPerBatch = netD.batchSize*netD.repBatchSize;
            % Loop
            loop = 0;
            if netD.gpu==1||netG.gpu==1||netQ.gpu==1||netP.gpu==1
                yPfake = ones(netP.batchSize, netP.repBatchSize, netP.dtype, 'gpuArray');
                yPreal = -ones(netP.batchSize, netP.repBatchSize, netP.dtype, 'gpuArray');
            else
                yPfake = ones(netP.batchSize, netP.repBatchSize, 'like', x);
                yPreal = -ones(netP.batchSize, netP.repBatchSize, 'like', x);
            end
            for i = 1:numDataPerBatch:numObs
                loop     = loop + 1;
                idxBatch = i:i+numDataPerBatch-1;
                % Noise and fake labels
                [yQfake, udIdxQfake] = network.generateLabels(netQ.numClasses, netQ.numCatVar, numDataPerBatch, netQ.dtype);               
                udIdxQfake = dp.selectIndices(udIdxQfake, numDataPerBatch, netQ.ny, netQ.dtype);
                xG         = [randn(numDataPerBatch, netG.nx-netQ.ny), yQfake];
                xG         = reshape(xG', [netG.nx*netQ.batchSize, netQ.repBatchSize]);
                yQfake     = reshape(yQfake', [netQ.batchSize*netQ.ny, netQ.repBatchSize]);
                % Real images and labels
                xD         = dp.dataLoader(x(:,:,:,idxBatch), netD.da, netD.batchSize, netD.repBatchSize, netD.trainMode); 
                xD         = reshape(xD, [netD.batchSize*netD.nodes(1), netD.repBatchSize]);  
                yQreal     = reshape(y(idxBatch, :)', [netQ.batchSize*netQ.ny, netQ.repBatchSize]);
                udIdxQreal = dp.selectIndices(udIdx(idxBatch, :), numDataPerBatch, netQ.ny, netQ.dtype);
                if netD.gpu==1||netG.gpu==1||netQ.gpu==1||netP.gpu==1
                    xG     = gpuArray(xG);
                    xD     = gpuArray(xD);
                    yQfake = gpuArray(yQfake);
                    yQreal = gpuArray(yQreal);
                    udIdxQfake = gpuArray(udIdxQfake);
                    udIdxQreal = gpuArray(udIdxQreal);  
                end  
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Update discriminator (netD, netP, and netQ)
                % Feed real examples to netD
                statesD = tagi.initializeInputs(statesD, xD, [], [], [], [], [], [], [], [], netD.xsc); 
                [thetaD, thetaP, thetaQ, normStatD, normStatP, normStatQ] = network.updateDPQacGAN(netD, thetaD, normStatD, statesD, maxIdxD, ...
                netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netP, thetaP, normStatP, statesP, maxIdxP, yPreal, yQreal, udIdxQreal);               
                                
                % Generate fake examples using netG
                statesG                       = tagi.initializeInputs(statesG, xG, [], [], [], [], [], [], [], [], netG.xsc);  
                [statesG, normStatG, maxIdxG] = tagi.feedForwardPass(netG, thetaG, normStatG, statesG, maxIdxG);
                [mzG, SzG, maG, SaG, JG,...
                    mdxsG, SdxsG, mxsG, SxsG] = tagi.extractStates(statesG);
                
                % Feed fake examples to netD
                statesD = tagi.initializeInputs(statesD, mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end}, mdxsG{end}, SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc);  
                [thetaD, thetaP, thetaQ, normStatD, normStatP, normStatQ] = network.updateDPQacGAN(netD, thetaD, normStatD, statesD, maxIdxD, ...
                netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netP, thetaP, normStatP, statesP, maxIdxP, yPfake, yQfake, udIdxQfake); 
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%% Update generator (netG)
                % Feed fake examples to netD
                statesD                       = tagi.initializeInputs(statesD, mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end}, mdxsG{end}, SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc);  
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD);               
                [mzD, SzD, maD, SaD, JD,...
                    mdxsD, SdxsD, mxsD, SxsD] = tagi.extractStates(statesD);
                
                % Feed netD's outputs to netP
                statesP                       = tagi.initializeInputs(statesP, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netP.xsc);  
                [statesP, normStatP, maxIdxP] = tagi.feedForwardPass(netP, thetaP, normStatP, statesP, maxIdxP);
                
                % Feed netD's outputs to netQ
                statesQ                       = tagi.initializeInputs(statesQ, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netQ.xsc);
                [statesQ, normStatQ, maxIdxQ] = tagi.feedForwardPass(netQ, thetaQ, normStatQ, statesQ, maxIdxQ);
                
                % Update hidden states for netQ, netP, & netD
                [~, ~, ~, ~, deltaMz0P, deltaSz0P] = tagi.hiddenStateBackwardPass(netP, thetaP, normStatP, statesP, yPreal, [], [], maxIdxP);
                [~, ~, ~, ~, deltaMz0Q, deltaSz0Q] = tagi.hiddenStateBackwardPass(netQ, thetaQ, normStatQ, statesQ, yQfake, [], udIdxQfake, maxIdxQ);
                deltaMzLD = deltaMz0P + deltaMz0Q;
                deltaSzLD = deltaSz0P + deltaSz0Q;
                [~, ~, ~, ~, deltaMz0D, deltaSz0D] = tagi.hiddenStateBackwardPass(netD, thetaD, normStatD, statesD, deltaMzLD , deltaSzLD , [], maxIdxD);
                
                % Update parameters & hidden states for netG
                [deltaMG, deltaSG,...
                    deltaMxG, deltaSxG] = tagi.hiddenStateBackwardPass(netG, thetaG, normStatG, statesG, deltaMz0D, deltaSz0D, [], maxIdxG);
                deltaThetaG             = tagi.parameterBackwardPass(netG, thetaG, normStatG, statesG, deltaMG, deltaSG, deltaMxG, deltaSxG);
                thetaG                  = tagi.globalParameterUpdate(thetaG, deltaThetaG); 
                
                Z = gather(reshape(maG{end}, [netG.ny, length(idxBatch)])');                
                if any(isnan(SzG{end}(:, 1)))||any(SzG{end}(:, 1)<0)
                    check=1;
                end                                          
            end
        end
        function [thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG, normStatQ, normStatP] = SSinfoGAN(netD, netG, netQ, netP, thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG, normStatQ, normStatP, xD, SxD)
            % Initialization
            numObs = size(xD, 4);
            if netQ.errorRateEval == 1
                [classObs, classIdx] = dp.class_encoding(netQ.numClasses);
                addIdx   = reshape(repmat(colon(0, netQ.ny-netQ.numContVar, (netQ.batchSize-1)*(netQ.ny-netQ.numContVar)), [netQ.numClasses, 1]), [netQ.numClasses*netQ.batchSize, 1]);
                classObs = repmat(classObs, [netQ.batchSize, 1]);
                classIdx = repmat(classIdx, [netQ.batchSize, 1]) + cast(addIdx, class(classIdx));
            end
            % Loop
            loop     = 0;
            time_tot = 0;
            for i = 1:netD.batchSize:numObs
                timeval  = tic;
                loop     = loop + 1;
                idxBatch = i:i+netD.batchSize-1;
                % Form a vector including random noise + categorical
                % variables + continuous variables
                [xcc, encoderIdx] = network.generateCatContVar(netQ.numClasses, netQ.numContVar, netQ.batchSize, netQ.dtype);
                xloopG = [randn(netG.batchSize, netG.nx-netQ.ny), xcc];
                xloopG = reshape(xloopG', [numel(xloopG), 1]);
                xcc    = reshape(xcc', [numel(xcc), 1]);
                % Real images pixels
                xloop = dp.dataLoader(xD(:,:,:,idxBatch), netD.da, netD.trainMode); 
                if netD.gpu==1||netG.gpu==1||netQ.gpu==1||netP.gpu==1
                    xloopG = gpuArray(xloopG);
                    xloop  = gpuArray(xloop);
                    xcc    = gpuArray(xcc);
                    encoderIdx = gpuArray(encoderIdx);
                end
                % Training
                if netD.trainMode == 1
                    yfake      = ones(netD.batchSize, 1, 'like', xD);
                    yreal      = -ones(netD.batchSize, 1, 'like', xD);
                    updateIdxQ = dp.selectIndices(encoderIdx, netQ.batchSize, netQ.ny, netQ.dtype);
                    % Update discriminator (Dnent)
                    [thetaD, thetaP, normStatD, normStatP] = network.updatenetDInfoGAN(netD, netP, thetaD, thetaP, normStatD, normStatP, xloop, SxD, yreal);
                    [mzG, SzG, maG, SaG, JG, mdxsG, SdxsG, mxsG, SxsG, maxIdxG, normStatG] = tagi.feedForwardPass(netG, xloopG, [], thetaG, normStatG); 
                    [thetaD, thetaP, normStatD, normStatP] = network.updatenetDInfoGAN(netD, netP, thetaD, thetaP, normStatD, normStatP, mzG{end}, SzG{end}, yfake);                                     
                    % Update generator (netG) and auxilaire net (netQ)
                    [thetaQ, normStatQ, maQ, SalQ, zfD, SzfD] = network.updatenetD4netGInfoGAN(netD, netQ, netP, thetaD, thetaQ, thetaP, normStatD, normStatQ, normStatP, mzG{end}, SzG{end}, yreal, xcc, updateIdxQ);                  
                    [thetaG, ~, ~] = tagi.feedBackward(netG, thetaG, mzG, SzG, maG, SaG, JG, mdxsG, SdxsG, mxsG, SxsG, zfD, SzfD, [], maxIdxG, normStatG); 
                    Z  = gather(reshape(maG{end}, [netG.ny, length(idxBatch)])');
                else 
                    [mzD, SzD] = tagi.feedForwardPass(netD, xloop, SxD, thetaD, normStatD);
                    [~, ~, maQ, SalQ] = tagi.feedForwardPass(netQ, mzD{end}, SzD{end}, thetaQ, normStatQ);
                end
                if i >1000
                    check=1;
                end
                if any(isnan(SzG{end}))||any(SzG{end}<0)
                    check=1;
                end
                time_loop = toc(timeval);
                time_tot  = time_tot + time_loop;
                time_rem  = double(time_tot)/(double(idxBatch(end)))*(numObs-double(idxBatch(end)))/60;
                if netQ.errorRateEval == 1
                    maCat = reshape(maQ{end}, [netQ.ny, length(idxBatch)])';
                    maCat = maCat(:, 1:netQ.ny-netQ.numContVar);
                    SaCat = reshape(SalQ, [netQ.ny, length(idxBatch)])';
                    SaCat = SaCat(:, 1:netQ.ny-netQ.numContVar);
                    maCat = reshape(maCat', [numel(maCat), 1]);
                    SaCat = reshape(SaCat', [numel(SaCat), 1]);
                    P     = dp.obs2class(maCat, SaCat + netQ.sv.^2, classObs, classIdx);
                    P     = reshape(P, [netQ.numClasses, netQ.batchSize])';
                    P     = gather(P);
                    [~,idx_pred] = max(P);
                    % Display error rate
                    if mod(idxBatch(end), netQ.obsShow) == 0 && i > 1 && netQ.trainMode == 1 && netQ.displayMode == 1
%                         disp(['     Error Rate : ' sprintf('%0.2f', 100*mean(er(max(1,i-200):i))) '%']);
                        disp(['     Time left  : ' sprintf('%0.2f',time_rem) ' mins'])
                    end
                else
                    if mod(idxBatch(end), netQ.obsShow) == 0 && i > 1 && netQ.trainMode == 1 && netQ.displayMode == 1
                        disp(['     Time left : ' sprintf('%0.2f',time_rem) ' mins'])
                    end
                end                             
            end
        end
        
        % Sharing funtions
        function [thetaD, thetaQ, thetaQc, thetaP, normStatD, normStatQ, normStatQc, normStatP] = updateDPQinfoGAN(netD, thetaD, normStatD, statesD, maxIdxD, ...
                netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netQc, thetaQc, normStatQc, statesQc, maxIdxQc, netP, thetaP, normStatP, statesP, maxIdxP, yQ, yQc, yP, udIdxQ)
            % Feed real examples to netD
            [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD);
            [mzD, SzD, maD, SaD, JD,...
                mdxsD, SdxsD, mxsD, SxsD] = tagi.extractStates(statesD);
            
            % Feed netD's output to the 1st head (netP) that
            % discriminates fake/real images
            statesP                       = tagi.initializeInputs(statesP, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netP.xsc);
            [statesP, normStatP, maxIdxP] = tagi.feedForwardPass(netP, thetaP, normStatP, statesP, maxIdxP);
            
            % Feed netD's output to the 2nd head (netQ) that infers
            % discrete slatent variables            
            statesQ                       = tagi.initializeInputs(statesQ, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netQ.xsc);
            [statesQ, normStatQ, maxIdxQ] = tagi.feedForwardPass(netQ, thetaQ, normStatQ, statesQ, maxIdxQ);
            
            % Feed netD's output to the 3rd head (netQc) that infers
            % continuous latent variables
            statesQc                         = tagi.initializeInputs(statesQc, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netQc.xsc);
            [statesQc, normStatQc, maxIdxQc] = tagi.feedForwardPass(netQc, thetaQc, normStatQc, statesQc, maxIdxQc);
            
            % Update parameters & hidden states for netP
            [deltaMP, deltaSP, deltaMxP,...
                deltaSxP, deltaMz0P, deltaSz0P] = tagi.hiddenStateBackwardPass(netP, thetaP, normStatP, statesP, yP, [], [], maxIdxP);
            deltaThetaP                         = tagi.parameterBackwardPass(netP, thetaP, normStatP, statesP, deltaMP, deltaSP, deltaMxP, deltaSxP);           
            thetaP                              = tagi.globalParameterUpdate(thetaP, deltaThetaP);
            
            % Update parameters & hidden states for netQ
            [deltaMQ, deltaSQ, deltaMxQ,...
                deltaSxQ, deltaMz0Q, deltaSz0Q] = tagi.hiddenStateBackwardPass(netQ, thetaQ, normStatQ, statesQ, yQ, [], udIdxQ, maxIdxQ);
            deltaThetaQ                         = tagi.parameterBackwardPass(netQ, thetaQ, normStatQ, statesQ, deltaMQ, deltaSQ, deltaMxQ, deltaSxQ);
            thetaQ                              = tagi.globalParameterUpdate(thetaQ, deltaThetaQ);
            
            % Update parameters & hidden states for netQc
            [deltaMQc, deltaSQc, deltaMxQc,...
                deltaSxQc, deltaMz0Qc, deltaSz0Qc] = tagi.hiddenStateBackwardPass(netQc, thetaQc, normStatQc, statesQc, yQc, [], [], maxIdxQc);
            deltaThetaQc                           = tagi.parameterBackwardPass(netQc, thetaQc, normStatQc, statesQc, deltaMQc, deltaSQc, deltaMxQc, deltaSxQc);
            thetaQc                                = tagi.globalParameterUpdate(thetaQc, deltaThetaQc);
            
            % Update parameters & hidden states for Dnent from netQ and netP
            deltaMzLD = deltaMz0P + deltaMz0Q + deltaMz0Qc;
            deltaSzLD = deltaSz0P + deltaSz0Q + deltaSz0Qc;
            [deltaMD, deltaSD,...
                deltaMxD, deltaSxD] = tagi.hiddenStateBackwardPass(netD, thetaD, normStatD, statesD,deltaMzLD, deltaSzLD, [], maxIdxD);
            deltaThetaD             = tagi.parameterBackwardPass(netD, thetaD, normStatD, statesD, deltaMD, deltaSD, deltaMxD, deltaSxD);
            thetaD                  = tagi.globalParameterUpdate(thetaD, deltaThetaD);
        end 
        function [thetaD, thetaP, thetaQ, normStatD, normStatP, normStatQ] = updateDPQacGAN(netD, thetaD, normStatD, statesD, maxIdxD, ...
                netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netP, thetaP, normStatP, statesP, maxIdxP, yP, yQ, udIdxQ)
            % Feed real examples to netD
            [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD);
            [mzD, SzD, maD, SaD, JD,...
                mdxsD, SdxsD, mxsD, SxsD] = tagi.extractStates(statesD);
            
            % Feed netD's output to the 1st head (netP) that
            % discriminates fake/real images
            statesP                       = tagi.initializeInputs(statesP, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netP.xsc);
            [statesP, normStatP, maxIdxP] = tagi.feedForwardPass(netP, thetaP, normStatP, statesP, maxIdxP);
            
            % Feed netD's output to the 2nd head (netQ) that infers
            % latent variables
            statesQ                       = tagi.initializeInputs(statesQ, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netQ.xsc);
            [statesQ, normStatQ, maxIdxQ] = tagi.feedForwardPass(netQ, thetaQ, normStatQ, statesQ, maxIdxQ);
            
            % Update parameters & hidden states for netP
            [deltaMP, deltaSP, deltaMxP,...
                deltaSxP, deltaMz0P, deltaSz0P] = tagi.hiddenStateBackwardPass(netP, thetaP, normStatP, statesP, yP, [], [], maxIdxP);
            deltaThetaP                         = tagi.parameterBackwardPass(netP, thetaP, normStatP, statesP, deltaMP, deltaSP, deltaMxP, deltaSxP);
            thetaP                              = tagi.globalParameterUpdate(thetaP, deltaThetaP);
            
            % Update parameters & hidden states for netQ
            [deltaMQ, deltaSQ, deltaMxQ,...
                deltaSxQ, deltaMz0Q, deltaSz0Q] = tagi.hiddenStateBackwardPass(netQ, thetaQ, normStatQ, statesQ, yQ, [], udIdxQ, maxIdxQ);
            deltaThetaQ                         = tagi.parameterBackwardPass(netQ, thetaQ, normStatQ, statesQ, deltaMQ, deltaSQ, deltaMxQ, deltaSxQ);
            thetaQ                              = tagi.globalParameterUpdate(thetaQ, deltaThetaQ);
            
            % Update parameters & hidden states for Dnent from netQ and netP
            deltaMzLD = deltaMz0P + deltaMz0Q;
            deltaSzLD = deltaSz0P + deltaSz0Q;
            [deltaMD, deltaSD,...
                deltaMxD, deltaSxD] = tagi.hiddenStateBackwardPass(netD, thetaD, normStatD, statesD,deltaMzLD, deltaSzLD, [], maxIdxD);
            deltaThetaD             = tagi.parameterBackwardPass(netD, thetaD, normStatD, statesD, deltaMD, deltaSD, deltaMxD, deltaSxD);
            thetaD                  = tagi.globalParameterUpdate(thetaD, deltaThetaD);
        end            
        function [thetaD, thetaP, normStatD, normStatP] = updateDPinfoGAN(netD, thetaD, normStatD, statesD, maxIdxD, netP, thetaP, normStatP, statesP, maxIdxP, yP)            
            [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, thetaD, normStatD, statesD, maxIdxD);
            [mzD, SzD, maD, SaD, JD,...
                mdxsD, SdxsD, mxsD, SxsD] = tagi.extractStates(statesD);
            
            % Feed netD's output to the 1st head (netP) that
            % discriminates fake/real images
            statesP                       = tagi.initializeInputs(statesP, mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netP.xsc);
            [statesP, normStatP, maxIdxP] = tagi.feedForwardPass(netP, thetaP, normStatP, statesP, maxIdxP);
            
            % Update parameters & hidden states for netP
            [deltaMP, deltaSP, deltaMxP,...
                deltaSxP, deltaMz0P, deltaSz0P] = tagi.hiddenStateBackwardPass(netP, thetaP, normStatP, statesP, yP, [], [], maxIdxP);
            deltaThetaP                         = tagi.parameterBackwardPass(netP, thetaP, normStatP, statesP, deltaMP, deltaSP, deltaMxP, deltaSxP);
            thetaP                              = tagi.globalParameterUpdate(thetaP, deltaThetaP);
            
            % Update parameters & hidden states for Dnent from netP
            [deltaMD, deltaSD,...
                deltaMxD, deltaSxD] = tagi.hiddenStateBackwardPass(netD, thetaD, normStatD, statesD, deltaMz0P, deltaSz0P, [], maxIdxD);
            deltaThetaD             = tagi.parameterBackwardPass(netD, thetaD, normStatD, statesD, deltaMD, deltaSD, deltaMxD, deltaSxD);
            thetaD                  = tagi.globalParameterUpdate(thetaD, deltaThetaD);
        end
        
        function [x, idx, xG]      = generateLatentVar(numClasses, numCatVar, numContVar, B, latentDim, dtype, gpu)
            [xcat, idx1] = network.generateLabels(numClasses, numCatVar, B, dtype);
            numH  = cast(size(xcat, 2), 'like', idx1);
            numUd = cast(size(idx1, 2), 'like', idx1);
            xcat  = reshape(xcat', [numH*numCatVar, B])';
            idx1  = reshape(idx1', [size(idx1,2)*numCatVar, B])';
            addIdx= reshape(repmat(colon(0,numH,(numCatVar-1)*numH), [numUd, 1]), [1, numCatVar*numUd]);
            idx1  = idx1+addIdx;
            if numContVar>0
                xcont = (rand(B, numContVar)*2-1);
                idx2  = repmat(colon(numH*numCatVar + 1, size(xcat, 2) + numContVar), [B, 1]);
                x     = [xcat, xcont];
                idx   = [idx1, idx2];
            else
                x   = xcat;
                idx = idx1;
            end
            xG    = [randn(B, latentDim-numH*numCatVar-numContVar), x];
            xG    = reshape(xG', [numel(xG), 1]);
            x     = reshape(x', [numel(x), 1]);
            idx   = dp.selectIndices(idx, B, numH*numCatVar+numContVar, dtype);
            if gpu == 1
                x   = gpuArray(x);
                idx = gpuArray(idx);
                xG  = gpuArray(xG);
            end
        end
        function [xd, xc, xG, idx] = generateLatentVar_V2(numClasses, numCatVar, numContVar, B, latentDim, dtype, gpu)
            [xd, idx] = network.generateLabels(numClasses, numCatVar, B, dtype);
            numH  = cast(size(xd, 2), 'like', idx);
            xd  = reshape(xd', [numH*numCatVar, B])';
            if numContVar > 0
                xc = (rand(B, numContVar)*2-1);
                xdc = [xd, xc];
            else
                xdc = xd;
                xc  = nan;
            end
            xG  = [randn(B, latentDim-numH*numCatVar-numContVar), xdc];
            xG  = reshape(xG', [numel(xG), 1]);
            xc  = reshape(xc', [numel(xc), 1]);
            xd  = reshape(xd', [numel(xd), 1]);
            idx = dp.selectIndices(idx, B, numH*numCatVar, dtype);
            if gpu == 1
                if ~isnan(xc)
                    xc  = gpuArray(xc);
                end
                xd  = gpuArray(xd);
                idx = gpuArray(idx);
                xG  = gpuArray(xG);
            end
        end
        function [x, idx, xG]      = generateLatentVar_plot(numClasses, numCatVar, numContVar, B, latentDim, dtype, gpu)
            [xcat, idx1] = network.generateLabels_plot(numClasses, numCatVar, B, dtype);
            numH  = cast(size(xcat, 2), 'like', idx1);
            numUd = cast(size(idx1, 2), 'like', idx1);
            xcat  = reshape(xcat', [numH*numCatVar, B])';
            idx1  = reshape(idx1', [size(idx1,2)*numCatVar, B])';
            addIdx= reshape(repmat(colon(0,numH,(numCatVar-1)*numH), [numUd, 1]), [1, numCatVar*numUd]);
            idx1  = idx1+addIdx;
            if numContVar>0
                xcont = (0*rand(B, numContVar)*2-1);
%                 xcont(:,2)=repmat(linspace(-2,2,10)', [10, 1]);
%                 xcont(:,1)=repmat(linspace(-2,2,10)', [10, 1]);
                idx2  = repmat(colon(numH*numCatVar + 1, size(xcat, 2) + numContVar), [B, 1]);
                x     = [xcat, xcont];
                idx   = [idx1, idx2];
            else
                x   = xcat;
                idx = idx1;
            end
            xG    = [randn(B, latentDim-numH*numCatVar-numContVar), x];
            xG    = reshape(xG', [numel(xG), 1]);
            x     = reshape(x', [numel(x), 1]);
            idx   = dp.selectIndices(idx, B, numH*numCatVar+numContVar, dtype);
            if gpu == 1
                x   = gpuArray(x);
                idx = gpuArray(idx);
                xG  = gpuArray(xG);
            end
        end
        function [x, idx]          = generateLabels(numClasses, numCat, B, dtype)
            x = randi(numClasses, [B, numCat]) - 1; 
            x = reshape(x', [numel(x), 1]);
            [x, idx] = dp.encoder(x, numClasses, dtype);
        end
        function [x, idx]          = generateLabels_plot(numClasses, numCat, B, dtype)
             x = randi(numClasses, [B, numCat]) - 1; 
%              x = [5     3     0     1     5     9     1     5     0     2
%                   5     3     8     6     9     5     1     0     3     7
%                   0     1     8     3     3     0     2     0     3     6
%                   6     6     9     4     4     0     6     1     9     5
%                   6     3     4     1     9     8     4     0     4     6];
%              x = repmat(x', [10, 1]);
%              x = reshape(x, [10, 10*5])';
%             x = repmat([5     7     1     7     2     1     4     3     8     0], [B, 1]); 
%             x = repmat([5     3     0     1     5     9     1     5     0     2], [B, 1]); 
%             %Hair color
%             x(:,8) = repmat([0:9]', [5, 1]);
%             % Hair color
%             x(:,9) = repmat([0:9]', [5, 1]);
%             % Gender
%             x(:,6) = repmat([0:9]', [5, 1]);
%             % Skin color
%              x(:,5) = repmat([0:9]', [5, 1]);
%             % Long hair
%             x(:,3) = repmat([0:9]', [5, 1]);
%             % Smile + rotation
%             x(:,1) = repmat([0:9]', [5, 1]);
%             % Shirt
%             x(:,4) = [0:9]';

%             x = repmat([0:9], [B, 1]);
%             x = repmat([5     3     0     1     5     9     1     5     0     2], [B, 1]);            
%             x = x*0+2;
            x = reshape(x', [numel(x), 1]);
            x = repmat([0:9], [B/10, numCat]);
            x = reshape(x, [numel(x), 1]);
%             x = 0*ones(B, 1);
            [x, idx] = dp.encoder(x, numClasses, dtype);
        end
        function [x, Sx, y, updateIdx] = generateRealSamples(x, Sx, y, updateIdx, numSamples)
            idx = randperm(size(x, 4), numSamples)';
            x = x(:, :, :, idx);
            if ~isempty(Sx)
                Sx = Sx(idx);
            end
            y = y(idx, :);
            updateIdx = updateIdx(idx, :);
        end  
        
        % Initialization
        function [net, states, maxIdx, netInfo] = initialization(net)
            % Build indices
            net = indices.initialization(net);
            net = indices.layerEncoder(net);
            net = indices.parameters(net);
            net = indices.covariance(net);
            netInfo = indices.savedInfo(net);
            % States
            states = tagi.initializeStates(net.nodes, net.batchSize, net.repBatchSize, net.xsc, net.dtype, net.gpu);
            maxIdx = tagi.initializeMaxPoolingIndices(net.nodes, net.layer, net.layerEncoder, net.batchSize, net.repBatchSize, net.dtype, net.gpu);                        
        end
        
        % Check points
        function checkParameters(theta, thetaT, numParamsPerlayer_2)
%             [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
%             [mwT, SwT, mbT, SbT, mwxT, SwxT, mbxT, SbxT] = tagi.extractParameters(thetaT);
            L = size(theta, 1);
            tol = 1E-5;
            for j = 2:L
                idxw = (numParamsPerlayer_2(1, j-1)+1):numParamsPerlayer_2(1, j);
                idxb = (numParamsPerlayer_2(2, j-1)+1):numParamsPerlayer_2(2, j);  
                A = gather(theta{1}(idxw))-gather(thetaT{1}(idxw));
                if any(abs(A)>tol)
                    check=1;
                end
                A = gather(theta{2}(idxw))-gather(thetaT{2}(idxw));
                if any(abs(A)>tol)
                    check=1;
                end
                A = gather(theta{3}(idxb))-gather(thetaT{3}(idxb));
                if any(abs(A)>tol)
                    check=1;
                end
                A = gather(theta{4}(idxb))-gather(thetaT{4}(idxb));
                if any(abs(A)>tol)
                    check=1;
                end
%                 A = mwx{j}-mwxT{j};
%                 if any(abs(A)>tol)&&~any(isnan(gather(A)))
%                     check=1;
%                 end
%                 A = Swx{j}-SwxT{j};
%                 if any(abs(A)>tol)&&~any(isnan(gather(A)))
%                     check=1;
%                 end
%                 A = mbx{j}-mbxT{j};
%                 if any(abs(A)>tol)&&~any(isnan(gather(A)))
%                     check=1;
%                 end
%                 A = Sbx{j}-SbxT{j};
%                 if any(abs(A)>tol)&&~any(isnan(gather(A)))
%                     check=1;
%                 end
            end
        end
        function checkHiddenStates(s1, s2)
            [mz1, Sz1, ma1, Sa1, ~, mdxs1, Sdxs1, mxs1, Sxs1] = tagi.extractStates(s1);
            [mz2, Sz2, ma2, Sa2, ~, mdxs2, Sdxs2, mxs2, Sxs2] = tagi.extractStates(s2);
            L = size(mz1, 1);
            tol = 1E-5;
            for j = 1:L
                A = gather(mz1{j})-gather(mz2{j});
                if any(abs(A)>tol)
                    check=1;
                end
                A = gather(Sz1{j})-gather(Sz2{j});
                if any(abs(A)>tol)
                    check=1;
                end
                A = gather(ma1{j})-gather(ma2{j});
                if any(abs(A)>tol)
                    check=1;
                end
                A = gather(Sa1{j})-gather(Sa2{j});
                if any(abs(A)>tol)
                    check=1;
                end
                % Skip connection
                A = gather(mdxs1{j})-gather(mdxs2{j});
                if any(abs(A)>tol)
                    check=1;
                end
                A = gather(Sdxs1{j})-gather(Sdxs2{j});
                if any(abs(A)>tol)
                    check=1;
                end
                A = gather(mxs1{j})-gather(mxs2{j});
                if any(abs(A)>tol)
                    check=1;
                end
                A = gather(Sxs1{j})-gather(Sxs2{j});
                if any(abs(A)>tol)
                    check=1;
                end
            end
        end
        function checkDeltaHiddenStates(m1, m2, S1, S2)
            L = size(m1, 1);
            tol = 1E-5;
            for j = 1:L
                A = gather(m1{j})-gather(m2{j});
                if any(abs(A)>tol)
                    check=1;
                end
                A = gather(S1{j})-gather(S2{j});
                if any(abs(A)>tol)
                    check=1;
                end
            end
        end
               
    end
end