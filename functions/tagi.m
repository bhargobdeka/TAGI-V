%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         tagi
% Description:  Tractable Approximate Gaussian Inference (TAGI) 
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      November 03, 2019
% Updated:      December 02, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef tagi
    methods(Static) 
        % Feedforward
        function [states, normStat, maxIdx] = feedForwardPass(net, theta, normStat, states, maxIdx)
            % Initialization
%             net.gpu=1;
             [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
            [mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs] = tagi.extractStates(states);
            [mra, Sra] = tagi.extractNormStat(normStat);
            numLayers  = length(net.nodes);
            actFunIdx  = net.actFunIdx; 
            layer      = net.layer;
            imgW       = cast(net.imgW, net.dtype);
            imgH       = cast(net.imgH, net.dtype);
            filter     = cast(net.filter, net.dtype);
            kernelSize = cast(net.kernelSize, net.dtype);
            B          = cast(net.batchSize, net.dtype);
            rB         = cast(net.repBatchSize, net.dtype);
            nodes      = cast(net.nodes, net.dtype);
            epsilon    = net.epsilon;   
            mhat       = cell(numLayers, 1);
            Shat       = cell(numLayers, 1);
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            % Hidden Layers
            for j = 2:numLayers
                idxw = (numParamsPerlayer_2(1, j-1)+1):numParamsPerlayer_2(1, j);
                idxb = (numParamsPerlayer_2(2, j-1)+1):numParamsPerlayer_2(2, j);     
                % Max pooling
                if layer(j) == net.layerEncoder.mp 
                    maPool = normrnd(ma{j-1}, sqrt(abs(Sa{j-1})));
                    if net.padding(j-1) ~= 0
                        maPool = vertcat(maPool, -Inf*ones(1, size(maPool, 2), 'like', maPool));
                    end
                    maPool(Sa{j-1}<=0) = -Inf;
                    [mz{j}, Sz{j}, maxIdx{j}] = tagi.mpMeanVar(mz{j}, Sz{j}, maPool, ma{j-1}, Sa{j-1}, net.idxPooling{j-1}, maxIdx{j}, rB, net.gpu);
                    
                % Average pooling     
                elseif layer(j) == net.layerEncoder.ap 
                    [mz{j}, Sz{j}] = tagi.apMeanVar(mz{j}, Sz{j}, ma{j-1}, Sa{j-1}, net.idxPooling{j-1}, net.padding(j-1), rB);
                    
                % Normalization     
                elseif layer(j) == net.layerEncoder.ln || layer(j) == net.layerEncoder.bn 
                    if net.trainMode == 1
                        [mhat{j-1}, Shat{j-1}] = tagi.pMeanVar(ma{j-1}, Sa{j-1}, nodes(j-1), imgW(j-1), imgH(j-1), filter(j-1), B, rB, layer(j-1), layer(j), net.layerEncoder);
                        % Running average for mean and variance
                        mra{j-1} = net.normMomentum*mra{j-1} + (1-net.normMomentum)*mhat{j-1};
                        Sra{j-1} = net.normMomentum*Sra{j-1} + (1-net.normMomentum)*Shat{j-1};
                    end    
                    mhatD = tagi.distributeNormMeanVar(mra{j-1}, nodes(j-1), imgW(j-1), imgH(j-1), filter(j-1), B, rB, layer(j-1), layer(j), net.layerEncoder);
                    ShatD = tagi.distributeNormMeanVar(Sra{j-1}, nodes(j-1), imgW(j-1), imgH(j-1), filter(j-1), B, rB, layer(j-1), layer(j), net.layerEncoder);
                    if layer(j-1) == net.layerEncoder.fc
                        [mz{j}, Sz{j}] = tagi.fcNormMeanVar(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, mhatD, ShatD, epsilon, B, rB, net.gpu);
                    elseif layer(j-1) == net.layerEncoder.conv||layer(j-1) == net.layerEncoder.tconv                       
                        [mz{j}, Sz{j}] = tagi.convNormMeanVar(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, mhatD, ShatD, epsilon, imgH(j-1), imgH(j-1), filter(j-1), B, rB, net.gpu);
                    end  
                    
                % Convolutional
                elseif layer(j) == net.layerEncoder.conv 
                    if B==1&&rB==1
                        [mz{j}, Sz{j}] = tagi.convMeanVarB1(mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, net.idxFmwa(j-1, :),...
                            kernelSize(j-1), filter(j-1), imgW(j), imgH(j), filter(j), net.padding(j-1), net.gpu);
                    else
                        [mz{j}, Sz{j}] = tagi.convMeanVar(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, net.idxFmwa(j-1, :),...
                            kernelSize(j-1), filter(j-1), imgW(j), imgH(j), filter(j), B, rB, net.padding(j-1), net.gpu);
                    end                                        
                % Transposed convolutional    
                elseif layer(j) == net.layerEncoder.tconv  
                    [mz{j}, Sz{j}] = tagi.tconvMeanVar(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, net.idxFmwa(j-1, :),...
                        imgW(j), imgH(j), filter(j), B, rB, net.gpu); 
                    
                % Full-connected
                elseif layer(j) == net.layerEncoder.fc
                    if B==1&&rB==1
                        [mz{j}, Sz{j}] = tagi.fcMeanVarB1(mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, nodes(j-1), nodes(j), net.gpu);
                    else
                        [mz{j}, Sz{j}] = tagi.fcMeanVar(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, nodes(j-1), nodes(j), B, rB, net.gpu);
                    end
                end     
                
                % Shortcut connection for residual networks 
                if net.xsc(j)~=0&&(net.filter(net.xsc(j))~=net.filter(j)||net.imgW(net.xsc(j))~=net.imgW(j)) 
                    idxXsc = net.xsc(j);
                    idxwx = (numParamsPerlayer_2(3, idxXsc)+1):numParamsPerlayer_2(3, idxXsc+1);
                    idxbx = (numParamsPerlayer_2(4, idxXsc)+1):numParamsPerlayer_2(4, idxXsc+1);                   
                    [mxs{j}, Sxs{j}] = tagi.convMeanVar(mxs{j}, Sxs{j}, mwx(idxwx), Swx(idxwx), mbx(idxbx), Sbx(idxbx), ma{idxXsc}, Sa{idxXsc}, net.idxFmwaXsc(idxXsc, :),...
                        1, filter(idxXsc), imgW(j), imgH(j), filter(j), B, rB, net.paddingXsc(idxXsc), net.gpu);
                    % Save convolutional hidden state before adding x
                    % shortcut
                    mdxs{j} = mz{j};
                    Sdxs{j} = Sz{j};
                    [mz{j}, Sz{j}] = arrayfun(@twoPlus, mz{j}, Sz{j}, mxs{j}, Sxs{j});
                elseif net.xsc(j)~=0&&(net.filter(net.xsc(j))==net.filter(j)||net.imgW(net.xsc(j))~=net.imgW(j))
                    mxs{j}  = mz{net.xsc(j)};
                    Sxs{j}  = Sz{net.xsc(j)};
                    mdxs{j} = mz{j};
                    Sdxs{j} = Sz{j};
                    [mz{j}, Sz{j}] = arrayfun(@twoPlus, mz{j}, Sz{j}, mxs{j}, Sxs{j});
                end
                % Activation
                if actFunIdx(j)~=0 
                    [ma{j}, Sa{j}, J{j}] = act.meanVar(mz{j}, mz{j}, Sz{j}, actFunIdx(j), B, rB, net.gpu);
                else
                    ma{j} = mz{j};
                    Sa{j} = Sz{j};
                    J{j}  = ones(size(mz{j}), 'like', mz{j});                    
                end
            end 
            normStat = tagi.compressNormStat(mra, Sra);
            states   = tagi.compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs);
        end
        
        % Inference 
        function [deltaM, deltaS, deltaMx, deltaSx, deltaMz0, deltaSz0, sv] = hiddenStateBackwardPass(net, theta, normStat, states, y, Sy, udIdx, maxIdx)
            % Initialization
            
            [mw, ~, ~, ~, mwx, ~, ~, ~] = tagi.extractParameters(theta);
            [mz, Sz, ma, Sa, J, mdxs, Sdxs, ~, Sxs] = tagi.extractStates(states);
            [~, Sra] = tagi.extractNormStat(normStat);
            numLayers  = length(net.nodes);
            imgW       = cast(net.imgW, net.dtype);
            imgH       = cast(net.imgH, net.dtype);
            filter     = cast(net.filter, net.dtype);
            kernelSize = cast(net.kernelSize, net.dtype);
            stride     = cast(net.stride, net.dtype);
            B          = cast(net.batchSize, net.dtype);
            rB         = cast(net.repBatchSize, net.dtype);
            nodes      = cast(net.nodes, net.dtype);
            epsilon    = net.epsilon;
            layer      = net.layer;
            lHL        = numLayers-1;
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            
            deltaM     = cell(numLayers, 1);
            deltaS     = cell(numLayers, 1);
            deltaMx    = cell(numLayers, 1);
            deltaSx    = cell(numLayers, 1);
            deltaMxs   = cell(numLayers, 1);
            deltaMdxs  = cell(numLayers, 1);
            deltaSxs   = cell(numLayers, 1);
            deltaSdxs  = cell(numLayers, 1);                      
            if net.lastLayerUpdate == 1
                if net.learnSv == 0
                    if net.ny == length(net.sv)
                        sv = net.sv';
                    else
                        sv = repmat(net.sv, [net.ny, 1]);
                    end
                    % Update hidden states for the last hidden layer
                    if isempty(Sy)
                        R = repmat(sv.^2, [net.batchSize, 1]);
                    else
                        R = repmat(sv.^2, [net.batchSize, 1]) + Sy;
                    end
                    Szv = Sa{end} + R;
                    if isempty(udIdx)
                        [deltaMz, deltaSz] = tagi.fowardHiddenStateUpdate(ma{lHL+1}, Szv, J{lHL+1}.*Sz{lHL+1}, y, net.gpu);
                    else
                        mzf = ma{end}(udIdx);
                        Szf = J{lHL+1}(udIdx).*Sz{lHL+1}(udIdx);
                        ys  = y(udIdx);
                        Szv = Szv(udIdx);
                        deltaMz = zeros(size(mz{lHL+1}), 'like', mz{lHL+1});
                        deltaSz = zeros(size(Sz{lHL+1}), 'like', Sz{lHL+1});
                        [deltaMz(udIdx), deltaSz(udIdx)] = tagi.fowardHiddenStateUpdate(mzf, Szv, Szf, ys, net.gpu);
                    end
                elseif net.learnSv==1                   
                     if strcmp(net.task, 'regression')&&strcmp(net.noiseType, 'hete')
                        [mla, mv2a] = tagi.detachMeanVar(ma{end}, net.nl, net.nv2, B, rB);           %mla = mZ,  mv2a   = mV2hat
                        [Sla, Sv2a] = tagi.detachMeanVar(Sa{end}, net.nl, net.nv2, B, rB);
                        [Slz, ~]  = tagi.detachMeanVar(Sz{end}, net.nl, net.nv2, B, rB);
                        [Jl, Jv2] = tagi.detachMeanVar(J{end}, net.nl, net.nv2, B, rB);
                        % Activate log(\sigma_v2)
                        [mv2a, Sv2a, Cv2a] = act.NoiseActFun(mv2a, Sv2a, net.NoiseActFunIdx, net.gpu);                                                                            %BD
                        [deltaMlz, deltaSlz, deltaMv2z, deltaSv2z] = tagi.noiseUpdate4regression(Slz, mla, Sla, Jl, Jv2, mv2a, Sv2a, Cv2a, y, net.gpu);
                        deltaMz = tagi.attachMeanVar(deltaMlz, deltaMv2z, net.nl, net.nv2, B, rB);       %BD
                        deltaSz = tagi.attachMeanVar(deltaSlz, deltaSv2z, net.nl, net.nv2, B, rB);       %BD
                        
                        %[Pos_z_mv2hat,Pos_z_sv2hat] = act.Act2Z(Prior_z_mv2hat, Prior_z_sv2hat,Prior_act_mv2hat, Prior_act_sv2hat, Cv2a, Pos_act_mv2hat, Pos_act_Pv2hat); 
                    elseif strcmp(net.task, 'regression')&&strcmp(net.noiseType, 'homo')     % to be completed similar to 'hete'
                        mv2a = net.sv(1);
                        Sv2a = net.sv(2);
                        % Activating the variance node
%                         [mv2a, Sv2a, Cv2a] = act.NoiseActFun(mv2a, Sv2a, net.NoiseActFunIdx, net.gpu);   
                        mla  = ma{end};
                        Slz  = Sz{end};
                        Sla  = Sa{end};
                        Jl   = J{end};  
                        [deltaMz, deltaSz, ~, ~, mv2a, Sv2a] = tagi.homoNoiseUpdate4regression(Slz, mla, Sla, Jl, mv2a, Sv2a, y, net.gpu);               %BD
%                         net.sv(1) = net.sv(1) + sum(deltaMv2z, 1);
%                         net.sv(2) = net.sv(2) + sum(deltaSv2z, 1);
                          net.sv(1) = mv2a;
                          net.sv(2) = Sv2a;
                    elseif strcmp(net.task, 'classification')
                        [mla, mv2a] = tagi.detachMeanVar(ma{end}, net.nl, net.nv2, B, rB);
                        [Sla, Sv2a] = tagi.detachMeanVar(Sa{end}, net.nl, net.nv2, B, rB);
                        [Slz, ~]  = tagi.detachMeanVar(Sz{end}, net.nl, net.nv2, B, rB);
                        [Jl, Jv2] = tagi.detachMeanVar(J{end}, net.nl, net.nv2, B, rB);
                        % Activate log(\sigma_v2)
                        [mv2a, Sv2a, Cv2a] = act.expFun(mv2a, Sv2a, net.gpu);
                        
                        deltaMlz  = zeros(size(mla), 'like', mla);
                        deltaSlz  = zeros(size(mla), 'like', mla);
                        deltaMv2z = zeros(size(mla), 'like', mla);
                        deltaSv2z = zeros(size(mla), 'like', mla);
                        [deltaMlz(udIdx), deltaSlz(udIdx), deltaMv2z(udIdx), deltaSv2z(udIdx)] = tagi.noiseUpdate4classification_V2(Slz, mla, Sla, Jl, Jv2, mv2a, Sv2a, Cv2a, y, net.sv, udIdx, net.gpu);
                        deltaMz = tagi.attachMeanVar(deltaMlz, deltaMv2z, net.nl, net.nv2, B, rB);
                        deltaSz = tagi.attachMeanVar(deltaSlz, deltaSv2z, net.nl, net.nv2, B, rB);
                    end                    
                end
            else
                deltaMz = y;
                deltaSz = Sy;
            end
            sv = net.sv;  %BD
            for k = (numLayers-1):-1:1
                if kernelSize(k)==stride(k)||(kernelSize(k)==imgW(k)&&stride(k)==1); overlap = 0; else; overlap = 1; end
                if isempty(mdxs{k+1}); nSz = Sz{k+1}; else; nSz = Sdxs{k+1}; end
                if isempty(mdxs{k}); cSz = Sz{k}; else; cSz = Sdxs{k}; end
                
                cSxs = Sxs{k};
                idxw = (numParamsPerlayer_2(1, k)+1):numParamsPerlayer_2(1, k+1);
                %Shortcut connection for residual network
                if net.xsc(k+1)~=0 && (net.filter(net.xsc(k+1))~=net.filter(k+1)||net.imgW(net.xsc(k+1))~=net.imgH(k+1))
                    [deltaMx{k+1}, deltaSx{k+1}] = tagi.inovationVector(Sxs{k+1}, deltaMzx, deltaSzx, net.gpu);
                    idxXsc = net.xsc(k+1);  
                    idxwx = (numParamsPerlayer_2(3, idxXsc)+1):numParamsPerlayer_2(3, idxXsc+1);
                    if idxXsc>1                                 
                        [deltaMxs{idxXsc}, deltaSxs{idxXsc}, deltaMdxs{idxXsc}, deltaSdxs{idxXsc}] = tagi.xshortDelta(deltaMx{k+1}, deltaSx{k+1}, Sxs{idxXsc}, Sdxs{idxXsc}, J{idxXsc}, mwx(idxwx),...
                            net.idxSzzUdXsc{idxXsc}, net.idxFCzwaXsc(idxXsc, :), filter(idxXsc), B, rB, size(net.idxFCzwaXsc{idxXsc, 2}, 1), net.gpu); 
                    end                   
                elseif net.xsc(k+1)~=0 && (net.filter(net.xsc(k+1))==net.filter(k+1)||net.imgW(net.xsc(k+1))==net.imgH(k+1))
                    [deltaMx{k+1}, deltaSx{k+1}] = tagi.inovationVector(Sxs{k+1}, deltaMzx, deltaSzx, net.gpu);
                    idxXsc = net.xsc(k+1);
                    if idxXsc>1&&~isempty(Sxs{idxXsc})                      
                        [deltaMxs{idxXsc}, deltaSxs{idxXsc}, deltaMdxs{idxXsc}, deltaSdxs{idxXsc}] = tagi.xshortDelta(deltaMx{k+1}, deltaSx{k+1}, Sxs{idxXsc}, Sdxs{idxXsc}, J{idxXsc}, [],...
                            [], [], [], [], rB, [], net.gpu);
                    elseif idxXsc>1&&isempty(Sdxs{idxXsc})&&isempty(Sxs{idxXsc}) % First shortcut
                        [~, ~, deltaMdxs{idxXsc}, deltaSdxs{idxXsc}] = tagi.xshortDelta(deltaMx{k+1}, deltaSx{k+1}, [], Sz{idxXsc}, J{idxXsc}, [], [], [], [], [], rB, [], net.gpu);
                    end
                end   
                
                % Innovation vector
                [deltaM{k+1}, deltaS{k+1}] = tagi.inovationVector(nSz, deltaMz, deltaSz, net.gpu);
                
                % Max pooling 
                if layer(k+1) == net.layerEncoder.mp       
                    [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.mpHiddenStateBackwardPass(cSz, cSxs, J{k}, deltaM{k+1}, deltaS{k+1}, maxIdx{k+1}, rB, overlap, net.gpu);
                    
                % Average pooling     
                elseif layer(k+1) == net.layerEncoder.ap 
                    [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.agHiddenStateBackwardPass(cSz, cSxs, J{k}, size(net.idxPooling{k}, 2), deltaM{k+1}, deltaS{k+1},...
                        net.idxSzzUd{k}, imgW(k+1), imgH(k+1), filter(k+1), kernelSize(k), B, rB, overlap, net.gpu);
                    
                % Convolutional     
                elseif layer(k+1) == net.layerEncoder.conv 
                    if k > 1||net.convariateEstm
                        if B==1&&rB==1
                            [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.convHiddenStateBackwardPassB1(cSz, cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                            net.idxSzzUd{k}, net.idxFCzwa(k, :), imgW(k), imgH(k), filter(k), net.gpu);
                        else
                            [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.convHiddenStateBackwardPass(cSz, cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                                net.idxSzzUd{k}, net.idxFCzwa(k, :), imgW(k), imgH(k), filter(k), B, rB, net.gpu);
                        end                       
                    end
                    
                % Transposed convolutional
                elseif layer(k+1) == net.layerEncoder.tconv 
                    if k > 1||net.convariateEstm
                        [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.tconvHiddenStateBackwardPass(cSz, cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                            net.idxSzzUd{k}, net.idxFCzwa(k, :), imgW(k), imgH(k), filter(k), B, rB, net.gpu);                       
                    end
                    
                % Normalization     
                elseif layer(k+1) == net.layerEncoder.ln || layer(k+1) == net.layerEncoder.bn                     
                    if k > 1||net.convariateEstm
                        Shat = tagi.distributeNormMeanVar(Sra{k}, nodes(k), imgW(k), imgH(k), filter(k), B, rB, layer(k), layer(k+1), net.layerEncoder);
                        [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.normHiddenStateBackwardPass(cSz, cSxs, J{k}, mw(idxw), Shat, epsilon, deltaM{k+1}, deltaS{k+1},...
                            imgW(k), imgH(k), filter(k), B, rB, layer(k), net.layerEncoder, net.gpu);
                    end 
                    
                % Full-connected     
                elseif  layer(k+1) == net.layerEncoder.fc
                    if k > 1||net.convariateEstm
                        if B==1&&rB==1
                            [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.fcHiddenStateBackwardPassB1(cSz, cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1}, nodes(k), nodes(k+1), net.gpu);
                        else
                            [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.fcHiddenStateBackwardPass(cSz, cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1}, nodes(k), nodes(k+1), B, rB, net.gpu);
                        end                                               
                    end                  
                end
                
                % Update hidden states from shortcut
                if ~isempty(deltaMxs{k})&&~isempty(deltaMdxs{k})
                    [deltaMzx, deltaSzx, deltaMz, deltaSz] = arrayfun(@fourPlus, deltaMzx, deltaSzx, deltaMz, deltaSz, deltaMxs{k}, deltaSxs{k}, deltaMdxs{k}, deltaSdxs{k});
                elseif ~isempty(deltaMdxs{k})&&isempty(deltaMxs{k})
                    [deltaMz, deltaSz] = arrayfun(@twoPlus, deltaMz, deltaSz, deltaMdxs{k}, deltaSdxs{k});
                end
            end
            deltaMz0 = deltaMz;
            deltaSz0 = deltaSz;        
        end
        function deltaTheta = parameterBackwardPass(net, theta, normStat, states, deltaM, deltaS, deltaMx, deltaSx)
            % Initialization
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
            [~, ~, ma, ~, ~, ~, ~, ~, ~] = tagi.extractStates(states);
            [mra, Sra] = tagi.extractNormStat(normStat);
            numLayers  = length(net.nodes);
            imgW       = cast(net.imgW, net.dtype);
            imgH       = cast(net.imgH, net.dtype);
            filter     = cast(net.filter, net.dtype);
            kernelSize = cast(net.kernelSize, net.dtype);
            B          = cast(net.batchSize, net.dtype);
            rB         = cast(net.repBatchSize, net.dtype);
            nodes      = cast(net.nodes, net.dtype);
            epsilon    = net.epsilon;
            layer      = net.layer;
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            
            deltaMw    = mw;
            deltaSw    = Sw;
            deltaMb    = mb;
            deltaSb    = Sb;
            deltaMwx   = mwx;
            deltaSwx   = Swx;
            deltaMbx   = mbx;
            deltaSbx   = Sbx;
            for k = (numLayers-1):-1:1
                idxw = (numParamsPerlayer_2(1, k)+1):numParamsPerlayer_2(1, k+1);
                idxb = (numParamsPerlayer_2(2, k)+1):numParamsPerlayer_2(2, k+1);
                %Shortcut connection for residual network
                if net.xsc(k+1)~=0 && (net.filter(net.xsc(k+1))~=net.filter(k+1)||net.imgW(net.xsc(k+1))~=net.imgH(k+1))
                    idxXsc = net.xsc(k+1); 
                    idxwx = (numParamsPerlayer_2(3, idxXsc)+1):numParamsPerlayer_2(3, idxXsc+1);
                    idxbx = (numParamsPerlayer_2(4, idxXsc)+1):numParamsPerlayer_2(4, idxXsc+1);
                    [deltaMwx(idxwx), deltaSwx(idxwx), deltaMbx(idxbx), deltaSbx(idxbx)] = tagi.convParameterBackwardPass(deltaMwx(idxwx), deltaSwx(idxwx), deltaMbx(idxbx), deltaSbx(idxbx),...
                        Swx(idxwx), Sbx(idxbx), ma{idxXsc}, deltaMx{k+1}, deltaSx{k+1},...
                        net.idxFmwaXsc(idxXsc, :), net.paddingXsc(idxXsc), 1, filter(idxXsc), imgW(k+1), imgH(k+1), filter(k+1), B, rB, net.gpu);
                end 
                
                % Convolutional     
                if layer(k+1) == net.layerEncoder.conv  
                    if B==1&&rB==1
                        [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.convParameterBackwardPassB1(Sw(idxw), Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1},...
                            net.idxFmwa(k, :), net.padding(k), kernelSize(k), filter(k), imgW(k+1), imgH(k+1), filter(k+1), net.gpu);
                    else
                        [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.convParameterBackwardPass(deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb), Sw(idxw), Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1},...
                            net.idxFmwa(k, :), net.padding(k), kernelSize(k), filter(k), imgW(k+1), imgH(k+1), filter(k+1), B, rB, net.gpu);
                    end                   
                % Transposed convolutional
                elseif layer(k+1) == net.layerEncoder.tconv 
                    [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.tconvParameterBackwardPass(deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb), Sw(idxw), Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1},...
                        net.idxSwzUd{k}, net.idxFCwz(k, :), kernelSize(k), filter(k), imgW(k+1), imgH(k+1), filter(k+1), B, rB, net.gpu); 
                    
                % Normalization     
                elseif layer(k+1) == net.layerEncoder.ln || layer(k+1) == net.layerEncoder.bn  
                    mhat = tagi.distributeNormMeanVar(mra{k}, nodes(k), imgW(k), imgH(k), filter(k), B, rB, layer(k), layer(k+1), net.layerEncoder);
                    Shat = tagi.distributeNormMeanVar(Sra{k}, nodes(k), imgW(k), imgH(k), filter(k), B, rB, layer(k), layer(k+1), net.layerEncoder);
                    [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.normParameterBackwardPass(deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb), Sw(idxw), Sb(idxb), ma{k}, mhat, Shat, epsilon, deltaM{k+1}, deltaS{k+1},...
                        nodes(k), imgW(k), imgH(k), filter(k), B, rB, layer(k), net.layerEncoder, net.gpu);        
                    
                % Full-connected     
                elseif  layer(k+1) == net.layerEncoder.fc 
                    if B==1&&rB==1
                        [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.fcParameterBackwardPassB1(Sw(idxw), Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1}, nodes(k), nodes(k+1), net.gpu);
                    else
                        [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.fcParameterBackwardPass(deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb), Sw(idxw), Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1}, nodes(k), nodes(k+1), B, rB, net.gpu);
                    end
                   
                end
            end
%             if net.epoch < 10
%                 constr = 0.01;
%                 deltaMw(deltaMw>constr) = constr;
%                 deltaMw(deltaMw < -constr) = -constr;
%                 deltaSw(deltaSw>constr) = constr;
%                 deltaSw(deltaSw < -constr) = -constr;
%                 deltaMb(deltaMb>constr) = constr;
%                 deltaMb(deltaMb < -constr) = -constr;
%                 deltaSb(deltaSb>constr) = constr;
%                 deltaSb(deltaSb < -constr) = -constr;
%             elseif net.epoch < 5
%                 constr = 0.005;
%                 deltaMw(deltaMw>constr) = constr;
%                 deltaMw(deltaMw < -constr) = -constr;
%                 deltaSw(deltaSw>constr) = constr;
%                 deltaSw(deltaSw < -constr) = -constr;
%                 deltaMb(deltaMb>constr) = constr;
%                 deltaMb(deltaMb < -constr) = -constr;
%                 deltaSb(deltaSb>constr) = constr;
%                 deltaSb(deltaSb < -constr) = -constr;
%             end
            deltaTheta = tagi.compressParameters(deltaMw, deltaSw, deltaMb, deltaSb, deltaMwx, deltaSwx, deltaMbx, deltaSbx); 
            
        end
        
        % Pooling layer
        function [mz, Sz, maxIdx] = mpMeanVar(mz, Sz, maS, ma, Sa, idxpooling, maxIdx, rB, gpu)
            n = size(idxpooling, 1);   
            for t = 1:rB
                maSloop = maS(:, t);
                [~, idx] = max(maSloop(idxpooling), [], 2);
                if gpu == 1
                    n   = gpuArray(cast(n, 'int32'));
                    col = gpuArray.colon(1, n);
                    col = col(:);
                    fun = @(x, y, z) (x-1).*y + z;
                    idx = arrayfun(fun, idx, n, col);
                else
                    col = colon(1,n)';
                    idx = (idx-1)*n + col;
                end
                maxIdx(:, t) = idxpooling(idx);
                mz(:, t) = ma(maxIdx(:, t), t);
                Sz(:, t) = Sa(maxIdx(:, t), t);
            end
        end
        function [mz, Sz] = apMeanVar(mz, Sz, ma, Sa, idxPooling, padding, rB)
            n   = size(idxPooling, 2);
            if padding ~= 0
                zeroPad = zeros(1, size(ma, 2), 'like', ma);
                ma = [ma; zeroPad];
                Sa = [Sa; zeroPad];
            end
            for t = 1:rB  
                maloop = ma(:, t);
                Saloop = Sa(:, t);
                mz(:, t) = mean(maloop(idxPooling), 2);
                Sz(:, t) = sum(Saloop(idxPooling), 2)./(n^2);
            end           
        end             
        function [deltaMz, deltaSz, deltaMxs, deltaSxs] = mpHiddenStateBackwardPass(Sz, Sxs, J, deltaM, deltaS, maxIdx, rB, overlap, gpu)
            deltaMz  = Sz;
            deltaSz  = Sz;
            deltaMxs = Sxs;
            deltaSxs = Sxs;
            n = single(size(Sz, 1));
            if gpu == 1
                if isempty(Sxs)
                    for t = 1:rB
                        Czz = bsxfun(@times, J(:, t), Sz(:, t));
                        Czz = Czz(maxIdx(:, t));
                        if overlap == 1
                            [deltaMzloop, deltaSzloop] = arrayfun(@vectorizedDelta, Czz, deltaM(:, t), deltaS(:, t));
                            deltaMz(:, t) = accumarray(maxIdx(:, t), deltaMzloop, [n, 1], @sum);
                            deltaSz(:, t) = accumarray(maxIdx(:, t), deltaSzloop , [n, 1], @sum);
                        else
                            [deltaMz(maxIdx(:, t), t), deltaSz(maxIdx(:, t), t)] = arrayfun(@vectorizedDelta, Czz, deltaM(:, t), deltaS(:, t));
                        end
                    end
                else
                    for t = 1:rB
                        if overlap == 1
                            [deltaMzloop, deltaSzloop, deltaMxsloop, deltaSxsloop] = arrayfun(@vectorized4delta, J(maxIdx(:, t), t), Sz(maxIdx(:, t), t), Sxs(maxIdx(:, t), t), deltaM(:, t), deltaS(:, t));
                            deltaMz(:, t)  = accumarray(maxIdx(:, t), deltaMzloop, [n, 1], @sum);
                            deltaSz(:, t)  = accumarray(maxIdx(:, t), deltaSzloop , [n, 1], @sum);
                            deltaMxs(:, t) = accumarray(maxIdx(:, t), deltaMxsloop, [n, 1], @sum);
                            deltaSxs(:, t) = accumarray(maxIdx(:, t), deltaSxsloop , [n, 1], @sum);
                        else
                            [deltaMz(maxIdx(:, t), t), deltaSz(maxIdx(:, t), t), deltaMxs(maxIdx(:, t), t), deltaSxs(maxIdx(:, t), t)] = arrayfun(@vectorized4delta, J(maxIdx(:, t), t), Sz(maxIdx(:, t), t), Sxs(maxIdx(:, t), t), deltaM(:, t), deltaS(:, t));
                        end
                    end
                end
            else
                if isempty(Sxs)
                    for t = 1:rB
                        Czz = J(:, t).*Sz(:, t);
                        Czz = Czz(maxIdx(:, t));
                        if overlap == 1
                            deltaMzloop   = Czz.*deltaM(:, t);
                            deltaSzloop   = Czz.*deltaS(:, t).*Czz;
                            deltaMz(:, t) = accumarray(maxIdx(:, t), deltaMzloop, [n, 1], @sum);
                            deltaSz(:, t) = accumarray(maxIdx(:, t), deltaSzloop , [n, 1], @sum);
                        else
                            deltaMz(maxIdx(:, t), t) = Czz.*deltaM(:, t);
                            deltaSz(maxIdx(:, t), t) = Czz.*deltaS(:, t).*Czz;
                        end
                    end
                else
                    for t = 1:rB
                        Czz = J(:, t).*Sz(:, t);
                        Czz = Czz(maxIdx(:, t));
                        Czx = J(:, t).*Sxs(:, t);
                        Czx = Czx(maxIdx(:, t));
                        if overlap == 1
                            deltaMzloop    = Czz.*deltaM(:, t);
                            deltaSzloop    = Czz.*deltaS(:, t).*Czz;
                            deltaMxsloop   = Czx.*deltaM(:, t);
                            deltaSxsloop   = Czx.*deltaS(:, t).*Czx;
                            deltaMz(:, t)  = accumarray(maxIdx(:, t), deltaMzloop, [n, 1], @sum);
                            deltaSz(:, t)  = accumarray(maxIdx(:, t), deltaSzloop, [n, 1], @sum);
                            deltaMxs(:, t) = accumarray(maxIdx(:, t), deltaMxsloop, [n, 1], @sum);
                            deltaSxs(:, t) = accumarray(maxIdx(:, t), deltaSxsloop, [n, 1], @sum);
                        else
                            deltaMz(maxIdx(:, t), t) = Czz.*deltaM(:, t);
                            deltaSz(maxIdx(:, t), t) = Czz.*deltaS(:, t).*Czz;
                        end
                    end
                end
            end
        end              
        function [deltaMz, deltaSz, deltaMxs, deltaSxs] = agHiddenStateBackwardPass(Sz, Sxs, J, n, deltaM, deltaS, idx,  wo, ho, fo, ki, B, rB, overlap, gpu)    
            deltaMz  = Sz;
            deltaSz  = Sz;
            deltaMxs = Sxs;
            deltaSxs = Sxs;
            n = cast(n, 'like', Sz);
            if gpu == 1
                if isempty(Sxs)
                    for t = 1:rB
                        if overlap == 0
                            deltaMzloop = reshape(repmat(reshape(repmat(transpose(deltaM(:, t)), [ki, 1]), [ki*ho, wo*fo*B]), [ki, 1]), [ho*wo*fo*ki*ki*B, 1]);
                            deltaSzloop = reshape(repmat(reshape(repmat(transpose(deltaS(:, t)), [ki, 1]), [ki*ho, wo*fo*B]), [ki, 1]), [ho*wo*fo*ki*ki*B, 1]);
                        else
                            zeroPadding = zeros(1,1,'like',deltaM);
                            deltaMzloop = [deltaM(:, t); zeroPadding];
                            deltaSzloop = [deltaS(:, t); zeroPadding];
                            deltaMzloop = deltaMzloop(idx);
                            deltaSzloop = deltaSzloop(idx);
                        end
                        [deltaMzloop, deltaSzloop] = arrayfun(@vectorizedDelta_V2, J(:, t), Sz(:, t)/n, deltaMzloop, deltaSzloop);
                        deltaMz(:, t) = sum(deltaMzloop, 2);
                        deltaSz(:, t) = sum(deltaSzloop, 2);
                    end
                else
                    Czz = bsxfun(@times, J, Sz);
                    Czx = bsxfun(@times, J, Sxs);
                    for t = 1:rB
                        if overlap == 0
                            deltaMloop = reshape(repmat(reshape(repmat(transpose(deltaM(:, t)), [ki, 1]), [ki*ho, wo*fo*B]), [ki, 1]), [ho*wo*fo*ki*ki*B, 1]);
                            deltaSloop = reshape(repmat(reshape(repmat(transpose(deltaS(:, t)), [ki, 1]), [ki*ho, wo*fo*B]), [ki, 1]), [ho*wo*fo*ki*ki*B, 1]);
                        else
                            zeroPadding = zeros(1,1,'like',deltaM);
                            deltaMloop = [deltaM(:, t); zeroPadding];
                            deltaSloop = [deltaS(:, t); zeroPadding];
                            deltaMloop = deltaMloop(idx);
                            deltaSloop = deltaSloop(idx);
                        end
                        [deltaMzloop, deltaSzloop, deltaMxsloop, deltaSxsloop] = arrayfun(@vectorized4delta, 1/n, Czz(:, t), Czx(:, t), deltaMloop, deltaSloop);
                        deltaMz(:, t) = sum(deltaMzloop, 2);
                        deltaSz(:, t) = sum(deltaSzloop, 2);
                        deltaMxs(:, t) = sum(deltaMxsloop, 2);
                        deltaSxs(:, t) = sum(deltaSxsloop, 2);
                    end
                end
            else
                if isempty(Sxs)
                    for t = 1:rB
                        Czz = (J(:, t).*Sz(:, t))/n;
                        if overlap == 0
                            deltaMzloop = reshape(repmat(reshape(repmat(transpose(deltaM(:, t)), [ki, 1]), [ki*ho, wo*fo*B]), [ki, 1]), [ho*wo*fo*ki*ki*B, 1]);
                            deltaSzloop = reshape(repmat(reshape(repmat(transpose(deltaS(:, t)), [ki, 1]), [ki*ho, wo*fo*B]), [ki, 1]), [ho*wo*fo*ki*ki*B, 1]);
                        else
                            zeroPadding = zeros(1,1,'like',deltaM);
                            deltaMzloop = [deltaM(:, t); zeroPadding];
                            deltaSzloop = [deltaS(:, t); zeroPadding];
                            deltaMzloop = deltaMzloop(idx);
                            deltaSzloop = deltaSzloop(idx);
                        end
                        deltaMzloop   = Czz.*deltaMzloop;
                        deltaSzloop   = Czz.*deltaSzloop.*Czz;
                        deltaMz(:, t) = sum(deltaMzloop, 2);
                        deltaSz(:, t) = sum(deltaSzloop, 2);
                    end
                else
                    Czz = (J.*Sz)/n;
                    Czx = (J.*Sxs)/n;
                    for t = 1:rB                       
                        if overlap == 0
                            deltaMloop = reshape(repmat(reshape(repmat(transpose(deltaM(:, t)), [ki, 1]), [ki*ho, wo*fo*B]), [ki, 1]), [ho*wo*fo*ki*ki*B, 1]);
                            deltaSloop = reshape(repmat(reshape(repmat(transpose(deltaS(:, t)), [ki, 1]), [ki*ho, wo*fo*B]), [ki, 1]), [ho*wo*fo*ki*ki*B, 1]);
                        else
                            zeroPadding = zeros(1,1,'like',deltaM);
                            deltaMloop = [deltaM(:, t); zeroPadding];
                            deltaSloop = [deltaS(:, t); zeroPadding];
                            deltaMloop = deltaMloop(idx);
                            deltaSloop = deltaSloop(idx);
                        end
                        deltaMzloop    = Czz(:, t).*deltaMloop;
                        deltaSzloop    = Czz(:, t).*deltaSloop*Czz(:, t);
                        deltaMxsloop   = Czx(:, t).*deltaMloop;
                        deltaSxsloop   = Czx(:, t).*deltaSloop*Czx(:, t);
                        deltaMz(:, t)  = sum(deltaMzloop, 2);
                        deltaSz(:, t)  = sum(deltaSzloop, 2);
                        deltaMxs(:, t) = sum(deltaMxsloop, 2);
                        deltaSxs(:, t) = sum(deltaSxsloop, 2);
                    end
                end
            end
        end
        
        % Normalization layer
        function [m, S] = pMeanVar(pm, pS, ni, wi, hi, fi, B, rB, li, lo, le)
            if li == le.fc && lo == le.ln 
                pm = reshape(pm, [ni, B*rB]);
                pS = reshape(pS, [ni, B*rB]);
                m  = mean(pm, 1);
                S  = (sum(pS, 1) + sum((pm-m).^2, 1))/(ni-1);            
            elseif li == le.fc && lo == le.bn
                pm = reshape(pm, [ni, B*rB]);
                pS = reshape(pS, [ni, B*rB]);
                m  = mean(pm, 2);
                S  = (sum(pS, 2) + sum((pm-m).^2, 2))/(B*rB-1);
            elseif li ~= le.fc && lo == le.ln
                pm = reshape(pm, [wi*hi*fi, B*rB]);
                pS = reshape(pS, [wi*hi*fi, B*rB]);
                m  = mean(pm, 1);
                S  = (sum(pS, 1) + sum((pm-m).^2, 1))/(wi*hi*fi-1);
            elseif li ~= le.fc && lo == le.bn
                pm = reshape(reshape(pm, [wi*hi*fi, B*rB])', [wi*hi*B*rB, fi]);
                pS = reshape(reshape(pS, [wi*hi*fi, B*rB])', [wi*hi*B*rB, fi]);
                m  = mean(pm, 1);
                S  = (sum(pS, 1) + sum((pm-m).^2, 1))/(wi*hi*B*rB - 1);
            end
            m = m(:);
            S = S(:);
        end
        function m = distributeNormMeanVar(m, ni, wi, hi, fi, B, rB, li, lo, le)
            if li == le.fc && lo == le.ln                 
                m  = reshape(repmat(m', [ni, 1]), [ni*B, rB]);                         
            elseif li == le.fc && lo == le.bn
                m  = repmat(m, [B, rB]);
            elseif li ~= le.fc && lo == le.ln
                m  = reshape(repmat(m', [wi*hi*fi, 1]), [wi*hi*fi*B, rB]);
            elseif li ~= le.fc && lo == le.bn
                m  = repmat(reshape(repmat(m', [wi*hi, 1]),[wi*hi*fi, 1]), [B, rB]);
            end
        end
        function [mz, Sz] = fcNormMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa, mhat, Shat, epsilon, B, rB, gpu)
            mb = repmat(mb, [B, 1]);
            Sb = repmat(Sb, [B, 1]);
            mw = repmat(mw, [B, 1]);            
            Sw = repmat(Sw, [B, 1]);                     
            if gpu == 1
                funA = @(x, y) 1./(x+y);
                A = arrayfun(funA, Shat, epsilon);
                [mz, Sz] = arrayfun(@vectorizedNormMeanVar, ma, Sa, mhat, mw, Sw, mb, Sb, A);
            else
                for t = 1:rB
                    A = 1./(Shat(:, t) + epsilon);
                    mz(:, t) = sqrt(A).*(ma(:, t) - mhat(:, t)).*mw + mb;
                    Sz(:, t) = A.*(Sa(:, t).*(mw.^2) + Sw.*(ma(:, t).^2 - mhat(:, t).^2 + Sa(:, t))) + Sb;
                end
            end
        end
        function [mz, Sz] = convNormMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa, mhat, Shat, epsilon,  wi, hi, fi, B, rB, gpu)
            mb   = repmat(reshape(repmat(mb', [wi*hi, 1]), [fi*hi*wi, 1]), [B, 1]);
            Sb   = repmat(reshape(repmat(Sb', [wi*hi, 1]), [fi*hi*wi, 1]), [B, 1]);      
            mw   = repmat(reshape(repmat(mw', [wi*hi, 1]), [fi*wi*hi, 1]), [B, 1]);
            Sw   = repmat(reshape(repmat(Sw', [wi*hi, 1]), [fi*wi*hi, 1]), [B, 1]);                    
            if gpu == 1
                funA = @(x, y) 1./(x+y);
                A = arrayfun(funA, Shat, epsilon);
                [mz, Sz] = arrayfun(@vectorizedNormMeanVar, ma, Sa, mhat, mw, Sw, mb, Sb, A);
            else
                for t = 1:rB
                    A = 1./(Shat + epsilon);
                    mz(:, t) = sqrt(A).*(ma(:, t) - mhat(:, t)).*mw + mb;
                    Sz(:, t) = A.*(Sa(:, t).*(mw.^2) + Sw.*(ma(:, t).^2 - mhat(:, t).^2 + Sa(:, t))) + Sb;
                end
            end
        end          
        function [deltaMw, deltaSw, deltaMb, deltaSb] = normParameterBackwardPass(deltaMw, deltaSw, deltaMb, deltaSb, Sw, Sb, ma, mra, Sra, epsilon, deltaM, deltaS, ni, wi, hi, fi, B, rB, li, layerEncoder, gpu)
            fun = @(x,y,z,t,q) sqrt((1./(x+q))).*(y-z).*t;
            if li == layerEncoder.fc % Previous layer is full-connected
                Sw = repmat(Sw, [B, 1]);
                Cbz = repmat(Sb, [B, 1]);                 
                if gpu == 1                   
                    Cwz = arrayfun(fun, Sra, ma, mra, Sw, epsilon);
                    % Weights
                    for t = 1:rB 
                        [deltaMwloop, deltaSwloop, deltaMbloop, deltaSbloop] = arrayfun(@vectorizedDelta4normParam, Cwz(:, t), Cbz, deltaM(:, t), deltaS(:, t));
                        deltaMw(:, t) = sum(reshape(deltaMwloop, [ni, B]), 2);
                        deltaSw(:, t) = sum(reshape(deltaSwloop, [ni, B]), 2);
                        deltaMb(:, t) = sum(reshape(deltaMbloop, [ni, B]), 2);
                        deltaSb(:, t) = sum(reshape(deltaSbloop, [ni, B]), 2);
                    end
                else
                    for t = 1:rB
                        A   = 1./(Sra(:, t) + epsilon);
                        Cwz = sqrt(A).*(ma(:, t) - mra(:, t)).*Sw;
                        deltaMwloop = Cwz.*deltaM(:, t);
                        deltaSwloop = Cwz.*deltaS(:, t).*Cwz;
                        deltaMbloop = Cbz.*deltaM(:, t);
                        deltaSbloop = Cbz.*deltaS(:, t).*Cbz;
                        deltaMw(:, t) = sum(reshape(deltaMwloop, [ni, B]), 2);
                        deltaSw(:, t) = sum(reshape(deltaSwloop, [ni, B]), 2);
                        deltaMb(:, t) = sum(reshape(deltaMbloop, [ni, B]), 2);
                        deltaSb(:, t) = sum(reshape(deltaSbloop, [ni, B]), 2);
                    end
                end
            elseif li == layerEncoder.conv||li == layerEncoder.tconv % Previous layer is convolutional
                Sw = repmat(reshape(repmat(Sw', [wi*hi, 1]), [fi*hi*wi, 1]), [B, 1]);
                Cbz = repmat(reshape(repmat(Sb', [wi*hi, 1]), [fi*hi*wi, 1]), [B, 1]);
                if gpu == 1
                    Cwz = arrayfun(fun, Sra, ma, mra, Sw, epsilon);
                    for t = 1:rB                       
                        [deltaMwloop, deltaSwloop, deltaMbloop, deltaSbloop] = arrayfun(@vectorizedDelta4normParam, Cwz(:, t), Cbz, deltaM(:, t), deltaS(:, t));
                        deltaMwloop = squeeze(permute(reshape(deltaMwloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaSwloop = squeeze(permute(reshape(deltaSwloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaMwloop = sum(sum(deltaMwloop, 1), 2);
                        deltaSwloop = sum(sum(deltaSwloop, 1), 2);
                        deltaMw(:, t) = deltaMwloop(:);
                        deltaSw(:, t) = deltaSwloop(:);
                        % Bias
                        deltaMbloop = squeeze(permute(reshape(deltaMbloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaSbloop = squeeze(permute(reshape(deltaSbloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaMbloop = sum(sum(deltaMbloop, 1), 2);
                        deltaSbloop = sum(sum(deltaSbloop, 1), 2);
                        deltaMb(:, t) = deltaMbloop(:);
                        deltaSb(:, t) = deltaSbloop(:);
                    end
                else
                    for t = 1:rB
                        A   = 1./(Sra(:, t)+epsilon);
                        Cwz = sqrt(A).*(ma(:, t) - mra(:, t)).*Sw;
                        [deltaMwloop, deltaSwloop] = vectorizedDelta(Cwz, deltaM(:, t), deltaS(:, t));
                        [deltaMbloop, deltaSbloop] = vectorizedDelta(Cbz, deltaM(:, t), deltaS(:, t));
                        deltaMwloop = squeeze(permute(reshape(deltaMwloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaSwloop = squeeze(permute(reshape(deltaSwloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaMwloop = sum(sum(deltaMwloop, 1), 2);
                        deltaSwloop = sum(sum(deltaSwloop, 1), 2);
                        deltaMw(:, t) = deltaMwloop(:);
                        deltaSw(:, t) = deltaSwloop(:);
                        % Bias
                        deltaMbloop = squeeze(permute(reshape(deltaMbloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaSbloop = squeeze(permute(reshape(deltaSbloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaMbloop = sum(sum(deltaMbloop, 1), 2);
                        deltaSbloop = sum(sum(deltaSbloop, 1), 2);
                        deltaMb(:, t) = deltaMbloop(:);
                        deltaSb(:, t) = deltaSbloop(:);
                    end
                end
            end
            deltaMw = sum(deltaMw, 2);
            deltaSw = sum(deltaSw, 2);
            deltaMb = sum(deltaMb, 2);
            deltaSb = sum(deltaSb, 2);
        end
        function [deltaMz, deltaSz, deltaMxs, deltaSxs] = normHiddenStateBackwardPass(Sz, Sxs, J, mw, Sra, epsilon, deltaM, deltaS, wi, hi, fi, B, rB, li, layerEncoder, gpu)
            deltaMz = Sz;
            deltaSz = Sz;
            deltaMxs = Sxs;
            deltaSxs = Sxs;
            if li == layerEncoder.fc
                mw = repmat(mw, [B, 1]);                
            elseif li == layerEncoder.conv||li == layerEncoder.tconv
                mw = repmat(reshape(repmat(mw', [wi*hi, 1]), [fi*wi*hi, 1]), [B, 1]);
            end
            if gpu == 1
                if isempty(Sxs)
                    fun = @(x, y, z, t, q) x.*sqrt(1./(y+q)).*z.*t;
                    Czz = arrayfun(fun, J, Sra, Sz, mw, epsilon);
                    for t = 1:rB
                        [deltaMz(:, t), deltaSz(:, t)] = arrayfun(@vectorizedDelta, Czz(:, t), deltaM(:, t), deltaS(:, t));
                    end
                else
                    fun = @(x, y, z, q) x.*sqrt(1./(y+q)).*z;
                    Czz = arrayfun(fun, J, Sra, Sz, epsilon);
                    Czx = arrayfun(fun, J, Sra, Sxs, epsilon);
                    for t = 1:rB
                        [deltaMz(:, t), deltaSz(:, t), deltaMxs(:, t), deltaSxs(:, t)] = arrayfun(@vectorized4delta, mw, Czz(:, t), Czx(:, t), deltaM(:, t), deltaS(:, t));
                    end
                end
            else
                if isempty(Sxs)
                    A   = 1./(Sra+epsilon);
                    Czz = J.*sqrt(A).*Sz.*mw;
                    for t = 1:rB                      
                        [deltaMz(:, t), deltaSz(:, t)] = vectorizedDelta(Czz, deltaM(:, t), deltaS(:, t));
                    end
                else
                    A   = 1./(Sra+epsilon);
                    Czz = J.*sqrt(A).*Sz;
                    Czx = J.*sqrt(A).*Sz;
                    for t = 1:rB
                        [deltaMz(:, t), deltaSz(:, t), deltaMxs(:, t), deltaSxs(:, t)] = arrayfun(@vectorized4delta, mw, Czz(:, t), Czx(:, t), deltaM(:, t), deltaS(:, t));
                    end
                end
            end
        end
        
        % Full connected layer 
        function [mz, Sz] = fcMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa, ni, no, B, rB, gpu)
            idxSum = 1;
            if any(isnan(mb))
                mb = zeros(1,1,'like', mw);
                Sb = zeros(1,1,'like', Sw);               
            else
                mb = repmat(mb, [B, 1]);
                Sb = repmat(Sb, [B, 1]);
            end
            mw  = repmat(reshape(mw, [ni, no]), [1, B]);                     
            Sw  = repmat(reshape(Sw, [ni, no]), [1, B]);                                  
            if gpu == 1
                for t = 1:rB
                    maloop = reshape(repmat(reshape(ma(:, t), [ni, B]), [no, 1]), [ni, no*B]);
                    Saloop = reshape(repmat(reshape(Sa(:, t), [ni, B]), [no, 1]), [ni, no*B]);
                    [mzloop, Szloop] = arrayfun(@vectorizedMeanVar, maloop, mw, Saloop, Sw);
                    mzloop = transpose(sum(mzloop, idxSum));
                    Szloop = transpose(sum(Szloop, idxSum));
                    [mz(:, t), Sz(:, t)] = arrayfun(@twoPlus, mzloop, Szloop, mb, Sb);
                end
            else
                for t = 1:rB
                    maloop = reshape(repmat(reshape(ma(:, t), [ni, B]), [no, 1]), [ni, no*B]);
                    Saloop = reshape(repmat(reshape(Sa(:, t), [ni, B]), [no, 1]), [ni, no*B]);
                    [mzloop, Szloop] = vectorizedMeanVar(maloop, mw, Saloop, Sw);
                    mzloop = transpose(sum(mzloop, idxSum));
                    Szloop = transpose(sum(Szloop, idxSum));
                    mz(:, t) = mzloop + mb;
                    Sz(:, t) = Szloop + Sb;
                end
            end            
        end
        function [mz, Sz] = fcMeanVarB1(mw, Sw, mb, Sb, ma, Sa, ni, no, gpu)
            if any(isnan(mb))
                mb = zeros(1,1,'like', mw);
                Sb = zeros(1,1,'like', Sw);               
            end
            mw = reshape(mw, [ni, no]);                     
            Sw = reshape(Sw, [ni, no]); 
            if gpu == 1
                [mzloop, Szloop] = arrayfun(@vectorizedMeanVar, ma, mw, Sa, Sw);
                mzloop = sum(mzloop, 1);
                Szloop = sum(Szloop, 1);
                mzloop = mzloop(:);
                Szloop = Szloop(:);
                [mz, Sz] = arrayfun(@twoPlus, mzloop, Szloop, mb, Sb);
            else
                [mzloop, Szloop] = vectorizedMeanVar(ma, mw, Sa, Sw);
                mzloop = transpose(sum(mzloop, 1));
                Szloop = transpose(sum(Szloop, 1));
                mz = mzloop + mb;
                Sz = Szloop + Sb;
            end            
        end 
        function [deltaMw, deltaSw, deltaMb, deltaSb] = fcParameterBackwardPass(deltaMw, deltaSw, deltaMb, deltaSb, Sw, Sb, ma, deltaMr, deltaSr, ni, no, B, rB, gpu)  
            Cbz = repmat(Sb, [1, B]);
            if gpu == 1  
                for t = 1:rB                   
                    maloop   = repmat(reshape(ma(:, t), [ni, B]), [no, 1]);               
                    deltaMrw = reshape(repmat(transpose(deltaMr(:, t)), [ni, 1]),[ni*no, B]);
                    deltaSrw = reshape(repmat(transpose(deltaSr(:, t)), [ni, 1]),[ni*no, B]);                  
                    % Weights
%                     Cwz = bsxfun(@times, Sw, maloop);  
                    [deltaMrw, deltaSrw] = arrayfun(@vectorizedDelta_V2, Sw, maloop, deltaMrw, deltaSrw);
                    deltaMw(:, t) = sum(deltaMrw, 2);
                    deltaSw(:, t) = sum(deltaSrw, 2);
                    % Bias
                    if any(~isnan(Sb))                        
                        deltaMrb = reshape(deltaMr(:, t), [no, B]);
                        deltaSrb = reshape(deltaSr(:, t), [no, B]);                      
                        [deltaMrb, deltaSrb] = arrayfun(@vectorizedDelta, Cbz, deltaMrb, deltaSrb);
                        deltaMb(:, t) = sum(deltaMrb, 2);
                        deltaSb(:, t) = sum(deltaSrb, 2);
                    end
                end
            else
                for t = 1:rB
                    if any(isnan(ma(:,t))) || any(isnan(deltaMr(:, t))) || any(isnan(deltaSr(:, t))) || any(isnan(Sw))     %BD
                        index1 = find(isnan(deltaMr(:, t)));
                        index2 = find(isnan(deltaSr(:, t)));
                        deltaMr(index1,t) = zeros(length(index1),1);
                        deltaSr(index2,t) = zeros(length(index2),1);
                    end
                    maloop   = repmat(reshape(ma(:, t), [ni, B]), [no, 1]);               
                    deltaMrw = reshape(repmat(transpose(deltaMr(:, t)), [ni, 1]),[ni*no, B]);
                    deltaSrw = reshape(repmat(transpose(deltaSr(:, t)), [ni, 1]),[ni*no, B]); 
                    Cwz      = Sw.*maloop;
                    deltaMrw = Cwz.*deltaMrw;
                    deltaSrw = Cwz.*deltaSrw.*Cwz;
                    deltaMw(:, t) = nansum(deltaMrw, 2);
                    deltaSw(:, t) = nansum(deltaSrw, 2);
                    if any(~isnan(Sb))
                        deltaMrb = reshape(deltaMr(:, t), [no, B]);
                        deltaSrb = reshape(deltaSr(:, t), [no, B]);                        
                        deltaMrb = Cbz.*deltaMrb;
                        deltaSrb = Cbz.*deltaSrb.*Cbz;
                        deltaMb(:, t) = nansum(deltaMrb, 2);
                        deltaSb(:, t) = nansum(deltaSrb, 2);
                    end
                end
            end  
            deltaMw = sum(deltaMw, 2);
            deltaSw = sum(deltaSw, 2);
            deltaMb = sum(deltaMb, 2);
            deltaSb = sum(deltaSb, 2);
        end
        function [deltaMw, deltaSw, deltaMb, deltaSb] = fcParameterBackwardPassB1(Sw, Sb, ma, deltaMr, deltaSr, ni, no, gpu)  
            Cbz      = Sb;                   
            maloop   = repmat(ma, [no, 1]);
            deltaMrw = repmat(transpose(deltaMr), [ni, 1]);
            deltaMrw = deltaMrw(:);
            deltaSrw = repmat(transpose(deltaSr), [ni, 1]);
            deltaSrw = deltaSrw(:);
            % Weights
            if gpu==1
                [deltaMrw, deltaSrw] = arrayfun(@vectorizedDelta_V2, Sw, maloop, deltaMrw, deltaSrw);
            else
                Cwa = Sw.*maloop;
                deltaMrw = Cwa.*deltaMrw;
                deltaSrw = (Cwa.^2).*deltaSrw;                
            end
            deltaMw = sum(deltaMrw, 2);
            deltaSw = sum(deltaSrw, 2);
            % Bias
            if any(~isnan(Sb))
                if gpu==1
                    [deltaMrb, deltaSrb] = arrayfun(@vectorizedDelta, Cbz, deltaMr, deltaSr);
                else
                    [deltaMrb, deltaSrb] = vectorizedDelta(Cbz, deltaMr, deltaSr);
                end
                deltaMb = sum(deltaMrb, 2);
                deltaSb = sum(deltaSrb, 2);
            else
                deltaMb = Sb;
                deltaSb = Sb;
            end
        end
        function [deltaMz, deltaSz, deltaMzx, deltaSzx] = fcHiddenStateBackwardPass(Sz, Sxs, J, mw, deltaM, deltaS, ni, no, B, rB, gpu) 
            deltaMz  = Sz;
            deltaSz  = Sz;
            deltaMzx = Sxs;
            deltaSzx = Sxs;
            mw  = repmat(reshape(mw, [ni, no]), [B, 1]);              
            if gpu == 1
                Caz = bsxfun(@times, J, Sz);
                if isempty(Sxs)
                    for t = 1:rB
                        deltaMzloop = reshape(repmat(reshape(deltaM(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        deltaSzloop = reshape(repmat(reshape(deltaS(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        [deltaMzloop, deltaSzloop] = arrayfun(@vectorizedDelta_V2, mw, Caz(:, t), deltaMzloop, deltaSzloop);
                        deltaMz(:, t) = sum(deltaMzloop, 2);
                        deltaSz(:, t) = sum(deltaSzloop, 2);
                    end
                else
                    Caxs = bsxfun(@times, J, Sxs);
                    for t = 1:rB
                        deltaMloop = reshape(repmat(reshape(deltaM(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        deltaSloop = reshape(repmat(reshape(deltaS(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        [deltaMzloop, deltaSzloop, deltaMxsloop, deltaSxsloop] = arrayfun(@vectorized4delta, mw, Caz(:, t), Caxs(:, t), deltaMloop, deltaSloop);
                        deltaMz(:, t)  = sum(deltaMzloop, 2);
                        deltaSz(:, t)  = sum(deltaSzloop, 2);
                        deltaMzx(:, t) = sum(deltaMxsloop, 2);
                        deltaSzx(:, t) = sum(deltaSxsloop, 2);
                    end
                end
            else
                if isempty(Sxs)
                    for t = 1:rB
                        Czz = J(:, t).*Sz(:, t).*mw;
                        deltaMzloop = reshape(repmat(reshape(deltaM(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        deltaSzloop = reshape(repmat(reshape(deltaS(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        deltaMzloop = Czz.*deltaMzloop;
                        deltaSzloop = Czz.*deltaSzloop.*Czz;
                        deltaMz(:, t) = nansum(deltaMzloop, 2);    %BD
                        deltaSz(:, t) = nansum(deltaSzloop, 2);
                    end
                else
                    for t = 1:rB
                        Czz = J(:, t).*Sz(:, t).*mw;
                        Czx = J(:, t).*Sz(:, t).*mw;
                        deltaMloop     = reshape(repmat(reshape(deltaM(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        deltaSloop     = reshape(repmat(reshape(deltaS(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        deltaMzloop    = Czz.*deltaMloop;
                        deltaSzloop    = Czz.*deltaSloop.*Czz;
                        deltaMxsloop   = Czx.*deltaMloop;
                        deltaSxsloop   = Czx.*deltaSloop.*Czx;
                        deltaMz(:, t)  = sum(deltaMzloop, 2);
                        deltaSz(:, t)  = sum(deltaSzloop, 2);
                        deltaMzx(:, t) = sum(deltaMxsloop, 2);
                        deltaSzx(:, t) = sum(deltaSxsloop, 2);
                    end
                end
            end
        end 
        function [deltaMz, deltaSz, deltaMzx, deltaSzx] = fcHiddenStateBackwardPassB1(Sz, Sxs, J, mw, deltaM, deltaS, ni, no, gpu) 
            mw  = reshape(mw, [ni, no]);              
            deltaMzx = Sxs;
            deltaSzx = Sxs;
            if isempty(Sxs)
                deltaMloop = repmat(deltaM', [ni, 1]);
                deltaSloop = repmat(deltaS', [ni, 1]);
                if gpu==1
                    Caz = bsxfun(@times, J, Sz);
                    [deltaMzloop, deltaSzloop] = arrayfun(@vectorizedDelta_V2, mw, Caz, deltaMloop, deltaSloop);
                else
                    Caz = J.*Sz;
                    Cwa = mw.*Caz;
                    deltaMzloop = Cwa.*deltaMloop;
                    deltaSzloop = (Cwa.^2).*deltaSloop;
                end
                deltaMz = sum(deltaMzloop, 2);
                deltaSz = sum(deltaSzloop, 2);
            else                
                deltaMloop = repmat(deltaM', [ni, 1]);
                deltaSloop = repmat(deltaS', [ni, 1]);
                if gpu==1
                    Caz = bsxfun(@times, J, Sz);
                    Caxs = bsxfun(@times, J, Sxs);
                    [deltaMzloop, deltaSzloop, deltaMxsloop, deltaSxsloop] = arrayfun(@vectorized4delta, mw, Caz, Caxs, deltaMloop, deltaSloop);
                else
                    Caz = J.*Sz;
                    Caxs = J.*Sxs;
                    [deltaMzloop, deltaSzloop, deltaMxsloop, deltaSxsloop] = vectorized4delta(mw, Caz, Caxs, deltaMloop, deltaSloop);
                end
                deltaMz  = sum(deltaMzloop, 2);
                deltaSz  = sum(deltaSzloop, 2);
                deltaMzx = sum(deltaMxsloop, 2);
                deltaSzx = sum(deltaSxsloop, 2);
            end
        end 
        
        % Convolutional layer
        function [mz, Sz] = convMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa, idxFmwa, ki, fi, wo, ho, fo, B, rB, padding, gpu)           
            if any(isnan(mb))
                mb = zeros(1,1,'like', ma(1));
                Sb = zeros(1,1,'like', Sa(1));               
            else
                mb = repmat(reshape(repmat(mb', [wo*ho, 1]), [wo*ho*fo, 1]), [B, 1]);
                Sb = repmat(reshape(repmat(Sb', [wo*ho, 1]), [wo*ho*fo, 1]), [B, 1]);
            end
            mw = repmat(reshape(mw, [ki*ki*fi, 1, fo]), [1, B*wo*ho, 1]);  
            Sw = repmat(reshape(Sw, [ki*ki*fi, 1, fo]), [1, B*wo*ho, 1]);            
            if padding ~= 0 && any(~isempty(Sa))
                zeroPad = zeros(1, size(ma, 2), 'like', ma);
                ma = [ma; zeroPad];
                Sa = [Sa; zeroPad];
            end            
            if gpu == 1
                for t = 1:rB
                    maloop = ma(:, t);
                    Saloop = Sa(:, t);
                    maloop = maloop(idxFmwa{2});
                    Saloop = Saloop(idxFmwa{2});
                    [mzloop, Szloop] = arrayfun(@vectorizedMeanVar, maloop, mw, Saloop, Sw);
                    mzloop = permute(reshape(sum(mzloop, 1), [wo*ho, B, fo]), [1 3 2]);
                    Szloop = permute(reshape(sum(Szloop, 1), [wo*ho, B, fo]), [1 3 2]);
                    mz(:, t) = mzloop(:);
                    Sz(:, t) = Szloop(:);
                end
                [mz, Sz] = arrayfun(@twoPlus, mz, Sz, mb, Sb);
            else
                for t = 1:rB
                    maloop = ma(:, t);
                    Saloop = Sa(:, t);
                    maloop = repmat(maloop(idxFmwa{2}), [1, 1, fo]);
                    Saloop = repmat(Saloop(idxFmwa{2}), [1, 1, fo]);
                    [mzloop, Szloop] = vectorizedMeanVar(maloop, mw, Saloop, Sw);
                    mzloop = sum(mzloop, 1);
                    Szloop = sum(Szloop, 1);
                    mzloop = permute(reshape(mzloop, [wo*ho, B, fo]), [1 3 2]);
                    Szloop = permute(reshape(Szloop, [wo*ho, B, fo]), [1 3 2]);
                    mz(:, t) = mzloop(:) + mb;
                    Sz(:, t) = Szloop(:) + Sb;
                end
            end
        end     
        function [mz, Sz] = convMeanVarB1(mw, Sw, mb, Sb, ma, Sa, idxFmwa, ki, fi, wo, ho, fo, padding, gpu)           
            if any(isnan(mb))
                mb = zeros(1,1,'like', mw);
                Sb = zeros(1,1,'like', Sw);               
            else
                mb = reshape(repmat(mb', [wo*ho, 1]), [wo*ho*fo, 1]);
                Sb = reshape(repmat(Sb', [wo*ho, 1]), [wo*ho*fo, 1]);
            end
            mw = reshape(mw, [ki*ki*fi, 1, fo]);  
            Sw = reshape(Sw, [ki*ki*fi, 1, fo]);            
            if padding ~= 0 && any(~isempty(Sa))
                zeroPad = zeros(1, size(ma, 2), 'like', ma);
                ma = [ma; zeroPad];
                Sa = [Sa; zeroPad];
            end 
            if gpu == 1
                ma = ma(idxFmwa{2});
                Sa = Sa(idxFmwa{2});
                [mz, Sz] = arrayfun(@vectorizedMeanVar, ma, mw, Sa, Sw);
                mz = sum(mz, 1);
                Sz = sum(Sz, 1);
                mz = mz(:);
                Sz = Sz(:);
                [mz, Sz] = arrayfun(@twoPlus, mz, Sz, mb, Sb);
            else
                maloop = repmat(ma(idxFmwa{2}), [1, 1, fo]);
                Saloop = repmat(Sa(idxFmwa{2}), [1, 1, fo]);
                [mzloop, Szloop] = vectorizedMeanVar(maloop, mw, Saloop, Sw);
                mzloop = sum(mzloop, 1);
                Szloop = sum(Szloop, 1);
                mz(:, t) = mzloop(:) + mb;
                Sz(:, t) = Szloop(:) + Sb;
            end
        end
        function [deltaMw, deltaSw, deltaMb, deltaSb] = convParameterBackwardPass(deltaMw, deltaSw, deltaMb, deltaSb, Sw, Sb, ma, deltaMr, deltaSr, idxFmwa, padding, k, fi, wo, ho, fo, B, rB, gpu)    
            Cbz = repmat(Sb', [1, B]);            
            Sw  = reshape(Sw, [k*k*fi, 1, fo]);
            if padding ~= 0
                zeroPad = zeros(1, size(ma, 2), 'like', ma);
                ma = [ma; zeroPad];
            end
            if gpu==1 
                for t = 1:rB
                    deltaMrw = repmat(reshape(permute(reshape(deltaMr(:, t), [1, wo*ho, fo, B]), [1, 2, 4, 3]), [1, wo*ho*B, fo]), [k*k*fi, 1, 1]);
                    deltaSrw = repmat(reshape(permute(reshape(deltaSr(:, t), [1, wo*ho, fo, B]), [1, 2, 4, 3]), [1, wo*ho*B, fo]), [k*k*fi, 1, 1]);
                    % Weights
                    maloop = ma(:, t);
%                     maloop = repmat(maloop(idxFmwa{2}), [1, 1, fo]);
                    maloop = maloop(idxFmwa{2});
%                     Cwz    = bsxfun(@times, Sw, maloop);  
                    [deltaMrw, deltaSrw] = arrayfun(@vectorizedDelta_V2, Sw, maloop, deltaMrw, deltaSrw);
                    deltaMrw = sum(deltaMrw, 2);
                    deltaSrw = sum(deltaSrw, 2);
                    deltaMw(:, t) = deltaMrw(:);
                    deltaSw(:, t) = deltaSrw(:);
                    % Bias
                    if any(~isnan(Sb))%||any(~isempty(Sb)) 
                        deltaMrb = reshape(deltaMr(:, t), [ho*wo, fo*B]);
                        deltaSrb = reshape(deltaSr(:, t), [ho*wo, fo*B]);
                        [deltaMrb, deltaSrb] = arrayfun(@vectorizedDelta, Cbz, deltaMrb, deltaSrb);
                        deltaMb(:, t) = sum(reshape(sum(deltaMrb, 1), [fo, B]), 2);
                        deltaSb(:, t) = sum(reshape(sum(deltaSrb, 1), [fo, B]), 2);
                    else
                        deltaMb(:, t) = nan;
                        deltaSb(:, t) = nan;
                    end
                end
            else 
                for t = 1:rB
                    deltaMrw = repmat(reshape(permute(reshape(deltaMr(:, t), [1, wo*ho, fo, B]), [1, 2, 4, 3]), [1, wo*ho*B, fo]), [k*k*fi, 1, 1]);
                    deltaSrw = repmat(reshape(permute(reshape(deltaSr(:, t), [1, wo*ho, fo, B]), [1, 2, 4, 3]), [1, wo*ho*B, fo]), [k*k*fi, 1, 1]);
                    % Weights
                    maloop = ma(:, t);
                    maloop = repmat(maloop(idxFmwa{2}), [1, 1, fo]);
                    Cwz    = bsxfun(@times, Sw, maloop); 
                    deltaMrw = Cwz.*deltaMrw;
                    deltaSrw = (Cwz.^2).*deltaSrw;
                    deltaMrw = sum(deltaMrw, 2);
                    deltaSrw = sum(deltaSrw, 2);
                    deltaMw(:, t) = deltaMrw(:);
                    deltaSw(:, t) = deltaSrw(:);
                    if any(~isnan(Sb))
                        deltaMrb = reshape(deltaMr(:, t), [ho*wo, fo*B]);
                        deltaSrb = reshape(deltaSr(:, t), [ho*wo, fo*B]);
                        deltaMrb = Cbz.*deltaMrb;
                        deltaSrb = (Cbz.^2).*deltaSrb;
                        deltaMb(:, t) = sum(reshape(sum(deltaMrb, 1), [fo, B]), 2);
                        deltaSb(:, t) = sum(reshape(sum(deltaSrb, 1), [fo, B]), 2);
                    else
                        deltaMb(:, t) = nan;
                        deltaSb(:, t) = nan;
                    end
                end
            end
            deltaMw = sum(deltaMw, 2);
            deltaSw = sum(deltaSw, 2);
            deltaMb = sum(deltaMb, 2);
            deltaSb = sum(deltaSb, 2);
        end     
        function [deltaMw, deltaSw, deltaMb, deltaSb] = convParameterBackwardPassB1(Sw, Sb, ma, deltaMr, deltaSr, idxFmwa, padding, k, fi, wo, ho, fo, gpu)    
            Cbz = Sb';            
            Sw  = reshape(Sw, [k*k*fi, 1, fo]);
            if padding ~= 0
                zeroPad = zeros(1, size(ma, 2), 'like', ma);
                ma = [ma; zeroPad];
            end
            deltaMrw = reshape(deltaMr, [1, wo*ho, fo]);
            deltaSrw = reshape(deltaSr, [1, wo*ho, fo]);
            % Weights
            ma = ma(idxFmwa{2});
            if gpu==1
                [deltaMrw, deltaSrw] = arrayfun(@vectorizedDelta_V2, Sw, ma, deltaMrw, deltaSrw);
            else
                [deltaMrw, deltaSrw] = vectorizedDelta_V2(Sw, ma, deltaMrw, deltaSrw);
            end
            deltaMrw = sum(deltaMrw, 2);
            deltaSrw = sum(deltaSrw, 2);
            deltaMw  = deltaMrw(:);
            deltaSw  = deltaSrw(:);
            % Bias
            if any(~isnan(Sb))%||any(~isempty(Sb))
                deltaMrb = reshape(deltaMr, [ho*wo, fo]);
                deltaSrb = reshape(deltaSr, [ho*wo, fo]);
                if gpu==1
                    [deltaMrb, deltaSrb] = arrayfun(@vectorizedDelta, Cbz, deltaMrb, deltaSrb);
                else
                    [deltaMrb, deltaSrb] = vectorizedDelta(Cbz, deltaMrb, deltaSrb);
                end
                deltaMb = sum(deltaMrb, 1);
                deltaMb = deltaMb(:);
                deltaSb = sum(deltaSrb, 1);
                deltaSb = deltaSb(:);
            else
                deltaMb = nan;
                deltaSb = nan;
            end
        end   
        function [deltaMz, deltaSz, deltaMzx, deltaSzx] = convHiddenStateBackwardPass(Sz, Sxs, J, mw, deltaM, deltaS, idx, idxFCzwa, wi, hi, fi, B, rB, gpu)
            deltaMz  = Sz;
            deltaSz  = Sz;
            deltaMzx = Sxs;
            deltaSzx = Sxs;
            n = size(idxFCzwa{1}, 1);
            mw = [mw;zeros(1,1, 'like', mw)];
            mw = repmat(reshape(mw(idxFCzwa{1}), [n, wi*hi, fi]), [1, B, 1]);           
            Caz = bsxfun(@times, Sz, J);
            if ~isempty(idx)
                deltaM = [deltaM; zeros(1,size(deltaM, 2), 'like', deltaM)];
                deltaS = [deltaS; zeros(1,size(deltaS, 2), 'like', deltaS)];
            end
            if gpu == 1  
                if isempty(Sxs)
                    for t = 1:rB
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        if ~isempty(idx)
                            deltaMloop = deltaMloop(idx');
                            deltaSloop = deltaSloop(idx');
                        end
                        Cazloop = reshape(permute(reshape(Caz(:, t), [1, wi*hi, fi, B]), [1, 2, 4, 3]), [1, wi*hi*B, fi]);
                        [deltaMloop, deltaSloop] = arrayfun(@vectorizedDelta_V2, Cazloop, mw, deltaMloop, deltaSloop);
                        deltaMloop = permute(reshape(sum(deltaMloop, 1), [wi*hi, B, fi]), [1, 3, 2]);
                        deltaSloop = permute(reshape(sum(deltaSloop, 1), [wi*hi, B, fi]), [1, 3, 2]);
                        deltaMz(:, t) = deltaMloop(:);
                        deltaSz(:, t) = deltaSloop(:);
                    end
                else
                    Caxs = bsxfun(@times, Sxs, J);
                    for t = 1:rB
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        if ~isempty(idx)
                            deltaMloop = deltaMloop(idx');
                            deltaSloop = deltaSloop(idx');
                        end
                        Cazloop  = reshape(permute(reshape(Caz(:, t), [1, wi*hi, fi, B]), [1, 2, 4, 3]), [1, wi*hi*B, fi]);
                        Caxsloop = reshape(permute(reshape(Caxs(:, t), [1, wi*hi, fi, B]), [1, 2, 4, 3]), [1, wi*hi*B, fi]);
                        [deltaMzloop, deltaSzloop, deltaMzxloop, deltaSzxloop] = arrayfun(@vectorized4delta, mw, Cazloop, Caxsloop, deltaMloop, deltaSloop);
                        deltaMzloop   = permute(reshape(sum(deltaMzloop, 1), [wi*hi, B, fi]), [1, 3, 2]);
                        deltaSzloop   = permute(reshape(sum(deltaSzloop, 1), [wi*hi, B, fi]), [1, 3, 2]);
                        deltaMzxloop  = permute(reshape(sum(deltaMzxloop, 1), [wi*hi, B, fi]), [1, 3, 2]);
                        deltaSzxloop  = permute(reshape(sum(deltaSzxloop, 1), [wi*hi, B, fi]), [1, 3, 2]);
                        deltaMz(:, t) = deltaMzloop(:);
                        deltaSz(:, t) = deltaSzloop(:);
                        deltaMzx(:, t) = deltaMzxloop(:);
                        deltaSzx(:, t) = deltaSzxloop(:);
                    end
                end
            else
                if isempty(Sxs)
                    for t = 1:rB
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        if ~isempty(idx)
                            deltaMloop = repmat(deltaMloop(idx'), [1, 1, fi]);
                            deltaSloop = repmat(deltaSloop(idx'), [1, 1, fi]);
                        end
                        Cazloop = reshape(permute(reshape(Caz(:, t), [1, wi*hi, fi, B]), [1, 2, 4, 3]), [1, wi*hi*B, fi]);
                        Czz = Cazloop.*mw;
                        deltaMloop = sum(Czz.*deltaMloop, 1);
                        deltaSloop = sum(Czz.*deltaSloop.*Czz, 1);
                        deltaMloop = permute(reshape(deltaMloop, [wi*hi, B, fi]), [1, 3, 2]);
                        deltaSloop = permute(reshape(deltaSloop, [wi*hi, B, fi]), [1, 3, 2]);
                        deltaMz(:, t) = deltaMloop(:);
                        deltaSz(:, t) = deltaSloop(:);
                    end
                else
                    Caxs = bsxfun(@times, Sxs, J);
                    for t = 1:rB
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        if ~isempty(idx)
                            deltaMloop = repmat(deltaMloop(idx'), [1, 1, fi]);
                            deltaSloop = repmat(deltaSloop(idx'), [1, 1, fi]);
                        end
                        Cazloop  = reshape(permute(reshape(Caz(:, t), [1, wi*hi, fi, B]), [1, 2, 4, 3]), [1, wi*hi*B, fi]);
                        Caxsloop = reshape(permute(reshape(Caxs(:, t), [1, wi*hi, fi, B]), [1, 2, 4, 3]), [1, wi*hi*B, fi]);
                        Czz      = Cazloop.*mw;
                        Czx      = Caxsloop.*mw;
                        deltaMzloop    = sum(Czz.*deltaMloop, 1);
                        deltaSzloop    = sum(Czz.*deltaSloop.*Czz, 1);
                        deltaMzxloop   = sum(Czx.*deltaMloop, 1);
                        deltaSzxloop   = sum(Czx.*deltaSloop.*Czx, 1);
                        deltaMzloop    = permute(reshape(deltaMzloop, [wi*hi, B, fi]), [1, 3, 2]);
                        deltaSzloop    = permute(reshape(deltaSzloop, [wi*hi, B, fi]), [1, 3, 2]);
                        deltaMzxloop   = permute(reshape(deltaMzxloop, [wi*hi, B, fi]), [1, 3, 2]);
                        deltaSzxloop   = permute(reshape(deltaSzxloop, [wi*hi, B, fi]), [1, 3, 2]);
                        deltaMz(:, t)  = deltaMzloop(:);
                        deltaSz(:, t)  = deltaSzloop(:);
                        deltaMzx(:, t) = deltaMzxloop(:);
                        deltaSzx(:, t) = deltaSzxloop(:);
                    end
                end
            end
        end
        function [deltaMz, deltaSz, deltaMzx, deltaSzx] = convHiddenStateBackwardPassB1(Sz, Sxs, J, mw, deltaM, deltaS, idx, idxFCzwa, wi, hi, fi, gpu)
            n = size(idxFCzwa{1}, 1);
            mw = [mw;zeros(1,1, 'like', mw)];
            mw = reshape(mw(idxFCzwa{1}), [n, wi*hi, fi]);                      
            if ~isempty(idx)
                deltaM = [deltaM; zeros(1, 1, 'like', deltaM)];
                deltaS = [deltaS; zeros(1, 1, 'like', deltaS)];
                deltaM = deltaM(idx)';
                deltaS = deltaS(idx)';
            end
            if isempty(Sxs)
                if gpu==1
                    Caz = bsxfun(@times, Sz, J);
                    Caz = reshape(Caz, [1, wi*hi, fi]);
                    [deltaMloop, deltaSloop] = arrayfun(@vectorizedDelta_V2, Caz, mw, deltaM, deltaS);
                else
                    Caz = Sz.*J;
                    Caz = reshape(Caz, [1, wi*hi, fi]);
                    [deltaMloop, deltaSloop] = vectorizedDelta_V2(Caz, mw, deltaM, deltaS);
                end
                deltaMloop = sum(deltaMloop, 1);
                deltaSloop = sum(deltaSloop, 1);
                deltaMz    = deltaMloop(:);
                deltaSz    = deltaSloop(:);
                deltaMzx   = Sxs;
                deltaSzx   = Sxs;
            else
                if gpu==1
                    Caz  = Sz.*J;
                    Caxs = Sxs.*J;
                    Caz  = reshape(Caz, [1, wi*hi, fi]);
                    Caxs = reshape(Caxs, [1, wi*hi, fi]);
                    [deltaMzloop, deltaSzloop, deltaMzxloop, deltaSzxloop] = arrayfun(@vectorized4delta, mw, Caz, Caxs, deltaM, deltaS);
                else
                    Caz = bsxfun(@times, Sz, J);
                    Caxs = bsxfun(@times, Sxs, J);
                    Caz = reshape(Caz, [1, wi*hi, fi]);
                    Caxs = reshape(Caxs, [1, wi*hi, fi]);
                    [deltaMzloop, deltaSzloop, deltaMzxloop, deltaSzxloop] = vectorized4delta(mw, Caz, Caxs, deltaM, deltaS);
                end
                deltaMzloop  = sum(deltaMzloop, 1);
                deltaSzloop  = sum(deltaSzloop, 1);
                deltaMzxloop = sum(deltaMzxloop, 1);
                deltaSzxloop = sum(deltaSzxloop, 1);
                deltaMz      = deltaMzloop(:);
                deltaSz      = deltaSzloop(:);
                deltaMzx     = deltaMzxloop(:);
                deltaSzx     = deltaSzxloop(:);
            end
        end
        
        % Transposed convolutional layer        
        function [mz, Sz] = tconvMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa, idxFmwa, wo, ho, fo, B, rB, gpu)           
            if any(~isnan(mb))
                mb = repmat(reshape(repmat(mb', [wo*ho, 1]), [wo*ho*fo, 1]), [B, 1]);
                Sb = repmat(reshape(repmat(Sb', [wo*ho, 1]), [wo*ho*fo, 1]), [B, 1]);
            else
                mb = zeros(1,1,'like', ma(1));
                Sb = zeros(1,1,'like', Sa(1));
            end
            n  = size(idxFmwa{1});
            mw = [mw; zeros(1, 1, 'like', mw)];
            Sw = [Sw; zeros(1, 1, 'like', Sw)];
            mw = repmat(mw(idxFmwa{1}), [1, 1, B]);  
            Sw = repmat(Sw(idxFmwa{1}), [1, 1, B]);
            ma = [ma; zeros(1, size(ma, 2), 'like', ma)];
            Sa = [Sa; zeros(1, size(ma, 2), 'like', Sa)];      
            if gpu == 1
                for t = 1:rB
                    maloop = ma(:, t);
                    Saloop = Sa(:, t);
                    maloop = repmat(reshape(maloop(idxFmwa{2}), [n(1), wo*ho, B]), [1, fo, 1]);
                    Saloop = repmat(reshape(Saloop(idxFmwa{2}), [n(1), wo*ho, B]), [1, fo, 1]);
                    [mzloop, Szloop] = arrayfun(@vectorizedMeanVar, maloop, mw, Saloop, Sw);
                    mzloop = sum(mzloop, 1);
                    Szloop = sum(Szloop, 1);
                    mz(:, t) = mzloop(:);
                    Sz(:, t) = Szloop(:);
                end
                if any(~isnan(mb))
                    [mz, Sz] = arrayfun(@twoPlus, mz, Sz, mb, Sb);
                end
            else
                for t = 1:rB
                    maloop = ma(:, t);
                    Saloop = Sa(:, t);
                    maloop = repmat(reshape(maloop(idxFmwa{2}), [n(1), wo*ho, B]), [1, fo, 1]);
                    Saloop = repmat(reshape(Saloop(idxFmwa{2}), [n(1), wo*ho, B]), [1, fo, 1]);
                    [mzloop, Szloop] = vectorizedMeanVar(maloop, mw, Saloop, Sw);
                    mzloop = sum(mzloop, 1);
                    Szloop = sum(Szloop, 1);
                    mz(:, t) = mzloop(:) + mb;
                    Sz(:, t) = Szloop(:) + Sb;
                end
            end
        end
        function [deltaMw, deltaSw, deltaMb, deltaSb] = tconvParameterBackwardPass(deltaMw, deltaSw, deltaMb, deltaSb, Sw, Sb, ma, deltaM, deltaS, idx, idxFCwz, ki, fi, wo, ho, fo, B, rB, gpu)  
            ma  = [ma; zeros(1, size(ma, 2), 'like', ma)]; 
            n   = size(idxFCwz{2});         
            Sw  = reshape(Sw, [1, ki*ki*fo, fi]);
            Cbz = repmat(Sb', [1, B]);                  
            if gpu == 1  
                for t = 1:rB
                    maloop = ma(:, t);
                    maloop = repmat(reshape(maloop(idxFCwz{2}), [n(1), ki*ki, fi]), [1, fo, 1]);
                    deltaMwloop = [deltaM(:, t); zeros(1, 1, 'like', deltaM)];
                    deltaSwloop = [deltaS(:, t); zeros(1, 1, 'like', deltaM)];
                    [deltaMwloop, deltaSwloop] = arrayfun(@vectorizedDelta_V2, Sw, maloop, deltaMwloop(idx), deltaSwloop(idx));
                    deltaMwloop = sum(deltaMwloop, 1);
                    deltaSwloop = sum(deltaSwloop, 1);
                    deltaMw(:, t) = deltaMwloop(:);
                    deltaSw(:, t) = deltaSwloop(:);                  
                end
                if any(~isnan(Sb))
                    for t = 1:rB
                        deltaMbloop = reshape(deltaM(:, t), [ho*wo, fo*B]);
                        deltaSbloop = reshape(deltaS(:, t), [ho*wo, fo*B]);
                        [deltaMbloop, deltaSbloop] = arrayfun(@vectorizedDelta, Cbz, deltaMbloop, deltaSbloop);
                        deltaMb(:, t) = sum(reshape(sum(deltaMbloop, 1), [fo, B]), 2);
                        deltaSb(:, t) = sum(reshape(sum(deltaSbloop, 1), [fo, B]), 2);
                    end
                end
            else 
                for t = 1:rB
                    maloop = ma(:, t);
                    maloop = repmat(reshape(maloop(idxFCwz{2}), [n(1), ki*ki, fi]), [1, fo, 1]);
                    deltaMwloop = [deltaM(:, t); zeros(1, 1, 'like', deltaM)];
                    deltaSwloop = [deltaS(:, t); zeros(1, 1, 'like', deltaM)];
                    Cwz = Sw.*maloop;        
                    deltaMwloop = Cwz.*deltaMwloop;
                    deltaSwloop = Cwz.*deltaSwloop.*Cwz;
                    deltaMwloop = sum(deltaMwloop, 1);
                    deltaSwloop = sum(deltaSwloop, 1);
                    deltaMw(:, t) = deltaMwloop(:);
                    deltaSw(:, t) = deltaSwloop(:);
                    if any(~isnan(Sb))
                        deltaMbloop = reshape(deltaM(:, t), [ho*wo, fo*B]);
                        deltaSbloop = reshape(deltaS(:, t), [ho*wo, fo*B]);
                        deltaMbloop = Cbz.*deltaMbloop;
                        deltaSbloop = Cbz.*deltaSbloop.*Cbz;
                        deltaMb(:, t) = sum(reshape(sum(deltaMbloop, 1), [fo, B]), 2);
                        deltaSb(:, t) = sum(reshape(sum(deltaSbloop, 1), [fo, B]), 2);
                    end
                end
            end
            deltaMw = sum(deltaMw, 2);
            deltaSw = sum(deltaSw, 2);
            deltaMb = sum(deltaMb, 2);
            deltaSb = sum(deltaSb, 2);
        end
        function [deltaMz, deltaSz, deltaMxs, deltaSxs] = tconvHiddenStateBackwardPass(Sz, Sxs, J, mw, deltaM, deltaS, idx, idxFCzwa, wi, hi, fi, B, rB, gpu)
            deltaMz  = Sz;
            deltaSz  = Sz;
            deltaMxs = Sxs;
            deltaSxs = Sxs;
            mw       = [mw; zeros(1, 1, 'like', mw)];
            mw       = repmat(mw(idxFCzwa{1}), [1, 1, B]);    
            n        = size(idx); 
            Caz      = bsxfun(@times, J, Sz);             
            deltaM   = [deltaM; zeros(1, size(deltaM, 2), 'like', deltaM)];
            deltaS   = [deltaS; zeros(1, size(deltaS, 2), 'like', deltaS)];
            if gpu == 1
                if isempty(Sxs)
                    for t = 1:rB
                        deltaMzloop = deltaM(:, t);
                        deltaSzloop = deltaS(:, t);
                        deltaMzloop = repmat(reshape(deltaMzloop(idx), [n(1), wi*hi, B]), [1, fi, 1]);
                        deltaSzloop = repmat(reshape(deltaSzloop(idx), [n(1), wi*hi, B]), [1, fi, 1]);
                        Cazloop     = reshape(Caz(:, t), [1, wi*hi*fi, B]);
                        
                        [deltaMzloop, deltaSzloop] = arrayfun(@vectorizedDelta_V2, mw, Cazloop, deltaMzloop, deltaSzloop);
                        deltaMzloop = sum(deltaMzloop, 1);
                        deltaSzloop = sum(deltaSzloop, 1);
                        deltaMz(:, t) = deltaMzloop(:);
                        deltaSz(:, t) = deltaSzloop(:);
                    end
                else
                    Caxs = bsxfun(@times, J, Sxs);   
                    for t = 1:rB
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        deltaMloop = repmat(reshape(deltaMloop(idx), [n(1), wi*hi, B]), [1, fi, 1]);
                        deltaSloop = repmat(reshape(deltaSloop(idx), [n(1), wi*hi, B]), [1, fi, 1]);
                        Cazloop    = reshape(Caz(:, t), [1, wi*hi*fi, B]);
                        Caxsloop   = reshape(Caxs(:, t), [1, wi*hi*fi, B]);
                        [deltaMzloop, deltaSzloop, deltaMxsloop, deltaSxsloop] = arrayfun(@vectorized4delta, mw, Cazloop, Caxsloop, deltaMloop, deltaSloop);
                        
                        deltaMzloop    = sum(deltaMzloop, 1);
                        deltaSzloop    = sum(deltaSzloop, 1);
                        deltaMxsloop   = sum(deltaMxsloop, 1);
                        deltaSxsloop   = sum(deltaSxsloop, 1);
                        deltaMz(:, t)  = deltaMzloop(:);
                        deltaSz(:, t)  = deltaSzloop(:);
                        deltaMxs(:, t) = deltaMxsloop(:);
                        deltaSxs(:, t) = deltaSxsloop(:);
                    end
                end
            else
                if isempty(Sxs)
                    for t = 1:rB
                        deltaMzloop = deltaM(:, t);
                        deltaSzloop = deltaS(:, t);
                        deltaMzloop = repmat(reshape(deltaMzloop(idx), [n(1), wi*hi, B]), [1, fi, 1]);
                        deltaSzloop = repmat(reshape(deltaSzloop(idx), [n(1), wi*hi, B]), [1, fi, 1]);
                        Cazloop     = reshape(Caz(:, t), [1, wi*hi*fi, B]);
                        deltaMzloop = sum(mw.*Cazloop.*deltaMzloop, 1);
                        deltaSzloop = sum(mw.*Cazloop.*deltaSzloop.*mw.*Cazloop, 1);
                        deltaMz(:, t) = deltaMzloop(:);
                        deltaSz(:, t) = deltaSzloop(:);
                    end
                else
                    Caxs = bsxfun(@times, J, Sxs);
                    for t = 1:rB
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        deltaMloop = repmat(reshape(deltaMloop(idx), [n(1), wi*hi, B]), [1, fi, 1]);
                        deltaSloop = repmat(reshape(deltaSloop(idx), [n(1), wi*hi, B]), [1, fi, 1]);
                        Cazloop    = reshape(Caz(:, t), [1, wi*hi*fi, B]);
                        Caxsloop   = reshape(Caxs(:, t), [1, wi*hi*fi, B]);                       
                        deltaMzloop    = sum(mw.*Cazloop.*deltaMloop, 1);
                        deltaSzloop    = sum(mw.*Cazloop.*deltaSloop.*mw.*Cazloop, 1);
                        deltaMxsloop   = sum(mw.*Caxsloop.*deltaMloop, 1);
                        deltaSxsloop   = sum(mw.*Caxsloop.*deltaSloop.*mw.*Caxsloop, 1);
                        deltaMz(:, t)  = deltaMzloop(:);
                        deltaSz(:, t)  = deltaSzloop(:);
                        deltaMxs(:, t) = deltaMxsloop(:);
                        deltaSxs(:, t) = deltaSxsloop(:);
                    end
                end
            end
        end  
        
        % Shortcut for residual network
        function [mz, Sz] = xshortcutMeanVar(mz, Sz, mw, Sw, ma, Sa, idxFmwa, wo, ho, fi, fo, k, B, rB, padding, gpu)           
            mb = zeros(1, 1, 'like', ma(1));
            Sb = zeros(1, 1, 'like', Sa(1));
            mw = reshape(repmat(reshape(mw, [k*k*fi, fo]), [B*wo*ho, 1]), [k*k*fi, fo*B*wo*wo]);  
            Sw = reshape(repmat(reshape(Sw, [k*k*fi, fo]), [B*wo*ho, 1]), [k*k*fi, fo*B*wo*wo]); 
            if padding ~= 0 
                ma = [ma; zeros(1, size(ma, 2), 'like', ma)];
                Sa = [Sa; zeros(1, size(Sa, 2), 'like', Sa)];
            end
            for t = 1:rB
                maloop = ma(:, t);
                Saloop = Sa(:, t);
                maloop = repmat(maloop(idxFmwa{2}), [1, fo]);
                Saloop = repmat(Saloop(idxFmwa{2}), [1, fo]);
                if gpu == 1
                    [mzloop, Szloop] = arrayfun(@vectorizedMeanVar, maloop, mw, Saloop, Sw);
                    mzloop = bsxfun(@plus, sum(mzloop, 1), mb);
                    Szloop = bsxfun(@plus, sum(Szloop, 1), Sb);
                else
                    [mzloop, Szloop] = vectorizedMeanVar(maloop, mw, Saloop, Sw);
                    mzloop = sum(mzloop, 1) + mb;
                    Szloop = sum(Szloop, 1) + Sb;
                end
                mz(:, t) = reshape(permute(reshape(mzloop, [wo, ho, B, fo]), [1 2 4 3]), [wo*ho*fo*B, 1]);
                Sz(:, t) = reshape(permute(reshape(Szloop, [wo, ho, B, fo]), [1 2 4 3]), [wo*ho*fo*B, 1]);
            end
        end
        function [deltaMxs, deltaSxs, deltaMdxs, deltaSdxs] = xshortDelta(deltaM, deltaS, Sxs, Sdxs, J, mwx, idx, idxFCzwaXsc, fi, B, rB, q, gpu)           
            if ~isempty(idx)
                deltaMxs  = zeros(size(Sxs), 'like', Sxs);
                deltaSxs  = deltaMxs;
                deltaMdxs = deltaMxs;
                deltaSdxs = deltaMxs;
                B    = cast(B, 'like', deltaM);
                fi   = cast(fi, 'like', deltaM);
                wh   = sqrt(cast(q/(fi*B), 'like', deltaM));
                
                n    = size(idxFCzwaXsc{1}, 2);
                q2   = size(idxFCzwaXsc{1}, 1);
                wh2  = sqrt(q2/fi);
                mwx  = [mwx ; zeros(1, 1, 'like', mwx)];
                mwx  = reshape(repmat(reshape(mwx(idxFCzwaXsc{1}), [wh2*wh2, n, fi]), [B, 1, 1]), [q2*B, n]);
                Cax  = bsxfun(@times, J, Sxs);
                Cadx = bsxfun(@times, J, Sdxs);
                if gpu == 1
                    for t = 1:rB
                        Caxloop  = Cax(:, t);
                        Cadxloop = Cadx(:, t);
                        Caxloop  = reshape(permute(reshape(Caxloop(idxFCzwaXsc{2}), [wh2, wh2, fi, B]), [1, 2, 4, 3]), [q2*B, 1]);
                        Cadxloop = reshape(permute(reshape(Cadxloop(idxFCzwaXsc{2}), [wh2, wh2, fi, B]), [1, 2, 4, 3]), [q2*B, 1]);
                        
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        deltaMloop = repmat(deltaMloop(idx), [fi, 1]);
                        deltaSloop = repmat(deltaSloop(idx), [fi, 1]);
                        [deltaMxsloop, deltaSxsloop, deltaMdxsloop, deltaSdxsloop] = arrayfun(@vectorized4delta, mwx, Caxloop, Cadxloop, deltaMloop, deltaSloop);
                        deltaMxs(idxFCzwaXsc{2}, t)  = reshape(permute(reshape(sum(deltaMxsloop, 2), [wh, wh, B, fi]), [1, 2, 4, 3]), [wh*wh*fi*B, 1]);
                        deltaSxs(idxFCzwaXsc{2}, t)  = reshape(permute(reshape(sum(deltaSxsloop, 2), [wh, wh, B, fi]), [1, 2, 4, 3]), [wh*wh*fi*B, 1]);
                        deltaMdxs(idxFCzwaXsc{2}, t) = reshape(permute(reshape(sum(deltaMdxsloop, 2), [wh, wh, B, fi]), [1, 2, 4, 3]), [wh*wh*fi*B, 1]);
                        deltaSdxs(idxFCzwaXsc{2}, t) = reshape(permute(reshape(sum(deltaSdxsloop, 2), [wh, wh, B, fi]), [1, 2, 4, 3]), [wh*wh*fi*B, 1]);
                    end
                else
                    for t = 1:rB
                        Caxloop  = reshape(permute(reshape(Cax(:, t), [wh2, wh2, fi, B]), [1, 2, 4, 3]), [q2*B, 1]);
                        Cadxloop = reshape(permute(reshape(Cadx(:, t), [wh2, wh2, fi, B]), [1, 2, 4, 3]), [q2*B, 1]);
                        
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        deltaMloop = repmat(deltaMloop(idx), [fi, 1]);
                        deltaSloop = repmat(deltaSloop(idx), [fi, 1]);
                        deltaMxsloop  = sum(Caxloop.*deltaMloop, 2);
                        deltaSxsloop  = sum((Caxloop.^2).*deltaSloop, 2);
                        deltaMdxsloop = sum(Cadxloop.*deltaMloop, 2);
                        deltaSdxsloop = sum((Cadx.^2).*deltaSloop, 2);
                        deltaMxs(idxFCzwaXsc{2}, t)  = reshape(permute(reshape(deltaMxsloop, [wh, wh, B, fi]), [1, 2, 4, 3]), [wh*wh*fi*B, 1]);
                        deltaSxs(idxFCzwaXsc{2}, t)  = reshape(permute(reshape(deltaSxsloop, [wh, wh, B, fi]), [1, 2, 4, 3]), [wh*wh*fi*B, 1]);                        
                        deltaMdxs(idxFCzwaXsc{2}, t) = reshape(permute(reshape(deltaMdxsloop, [wh, wh, B, fi]), [1, 2, 4, 3]), [wh*wh*fi*B, 1]);
                        deltaSdxs(idxFCzwaXsc{2}, t) = reshape(permute(reshape(deltaSdxsloop, [wh, wh, B, fi]), [1, 2, 4, 3]), [wh*wh*fi*B, 1]);
                    end
                end
            else               
                if gpu == 1                  
                    if ~isempty(Sxs)
                        [deltaMxs, deltaSxs, deltaMdxs, deltaSdxs] = arrayfun(@vectorized4delta, J, Sxs, Sdxs, deltaM, deltaS);
                    else
                        [deltaMdxs, deltaSdxs] = arrayfun(@vectorizedDelta_V2, J, Sdxs, deltaM, deltaS);
                        deltaMxs = [];
                        deltaSxs = [];
                    end                   
                else
                    if ~isempty(Sxs)
                        Cxx = J.*Sxs;
                        deltaMxs = Cxx.*deltaM;
                        deltaSxs = (Cxx.^2).*deltaS;
                    end
                    Cdxx = J.*Sdxs;
                    deltaMdxs = Cdxx.*deltaM;
                    deltaSdxs = (Cdxx.^2).*deltaS;
                end                
            end          
        end 
        
        % Reinforcement learning
        function [dm, dS, dmlog, dSlog] = policyDistribution(z, mz, Sz, mlog, Slog, gpu)
            [msigma, Ssigma] = act.expFun(mlog, Slog, gpu);
            if any(abs(msigma)>1E3)
                check=1;
            end
            % mean value
            dm = z.*msigma - mz.*msigma;
            dS = (z.^2).*Ssigma + Sz.*Ssigma + Sz.*(msigma.^2) + Ssigma.*(mz.^2);
            
            % log-sigma value
            mc = z.^2 - 2.*z.*mz + mz.^2 + Sz;
            Sc = Sz.*(4*(z.^2) + 2*Sz + 4*(mz.^2));
            
            dmlog = - 0.5+ 0.5*mc.*msigma;
            dSlog = 0.25*(Sc.*Ssigma + Sc.*(msigma.^2) + Ssigma.*(mc.^2));
        end
        function [dm, dS, dmlog, dSlog] = actorLoss(dmP, dSP, dmlogP, dSlogP, R, mt, St, mv, Sv)
            % Adavatage 
            mA = R + mt - mv;
            SA = St + Sv;
            % Mean
            dm = dmP.*mA;
            dS = dSP.*SA + (dSP).*(mA.^2) + (SA).*(dmP.^2);
            % Variance
            dmlog = dmlogP.*mA;
            dSlog = dSlogP.*SA + (dSlogP).*(mA.^2) + (SA).*(dmlogP.^2);           
        end
        function [m, S] = advantageMeanVar(prob, y, Sa, mt, St, mv, Sv, R, lr)
            m = lr.*(y-prob).*(R + mt - mv);
            S = (lr.^2).*(Sa.*(St + Sv) + Sa.*((R + mt - mv).^2) + (St + Sv).*((y-prob).^2));
        end
        
        % Shared functions for update step
        function [deltaM, deltaS] = inovationVector(SzF, dMz, dSz, gpu)
            if gpu == 1
                iSzF  = bsxfun(@rdivide, 1, SzF);
                iSzF(isinf(iSzF)) = zeros(1,1, 'like', dMz);
                [deltaM, deltaS] = arrayfun(@vectorizedDelta, iSzF, dMz, dSz);
            else              
                iSzF   = 1./SzF; 
                iSzF(isinf(iSzF)) = zeros(1,1, 'like', dMz);
                [deltaM, deltaS]  = vectorizedDelta(iSzF, dMz, dSz);
            end           
        end 
        function [deltaMz, deltaSz, K] = fowardHiddenStateUpdate(mzF, SzF, Cyz, y, gpu)           % added K -- BD
            if gpu == 1
                dz  = y - mzF;
                SzF = 1./SzF;
                SzF(isinf(SzF)) = 0;
                K = bsxfun(@times, Cyz, SzF);
                deltaMz = bsxfun(@times, K, dz);
                deltaSz = bsxfun(@times, -K, Cyz);
            else
                dz  = y - mzF;
%                 if isnan(1./SzF)
%                     check
%                 end
                SzF = 1./SzF;
                SzF(isinf(SzF)) = 0;
                K = Cyz.*SzF;
%                 if isnan(K)
%                     check
%                 end
                deltaMz = K.*dz;
                deltaSz = -K.*Cyz;
            end
        end   
        function theta = globalParameterUpdate(theta, deltaTheta, gpu)          
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
            [deltaMw, deltaSw, deltaMb, deltaSb, deltaMwx, deltaSwx, deltaMbx, deltaSbx] = tagi.extractParameters(deltaTheta);
            if gpu==1
                [mw, Sw]   = arrayfun(@twoPlus, mw, Sw, deltaMw, deltaSw);
                [mb, Sb]   = arrayfun(@twoPlus, mb, Sb, deltaMb, deltaSb);
                [mwx, Swx] = arrayfun(@twoPlus, mwx, Swx, deltaMwx, deltaSwx);
                [mbx, Sbx] = arrayfun(@twoPlus, mbx, Sbx, deltaMbx, deltaSbx);
            else
                [mw, Sw]   = twoPlus(mw, Sw, deltaMw, deltaSw);
                [mb, Sb]   = twoPlus(mb, Sb, deltaMb, deltaSb);
                [mwx, Swx] = twoPlus(mwx, Swx, deltaMwx, deltaSwx);
                [mbx, Sbx] = twoPlus(mbx, Sbx, deltaMbx, deltaSbx);
            end
            theta      = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);
        end
        function theta = globalParameterUpdateMultiGPUs(theta, deltaTheta, numParamsPerlayer, numDevices)  
            numParams  = sum(numParamsPerlayer, 2);           
            deltaTheta = cat(2, deltaTheta{:});
            [deltaMw, deltaSw, deltaMb, deltaSb, deltaMwx, deltaSwx, deltaMbx, deltaSbx] = tagi.extractParameters_V2(deltaTheta);
            deltaMw  = cat(1, deltaMw{:});
            deltaSw  = cat(1, deltaSw{:});
            deltaMb  = cat(1, deltaMb{:});
            deltaSb  = cat(1, deltaSb{:});
            deltaMwx = cat(1, deltaMwx{:});
            deltaSwx = cat(1, deltaSwx{:});
            deltaMbx = cat(1, deltaMbx{:});
            deltaSbx = cat(1, deltaSbx{:});  
            
            deltaMw  = sum(reshape(cat(1, deltaMw{:}), [numParams(1), numDevices]), 2);
            deltaSw  = sum(reshape(cat(1, deltaSw{:}), [numParams(1), numDevices]), 2);
            deltaMb  = sum(reshape(cat(1, deltaMb{:}), [numParams(2), numDevices]), 2);
            deltaSb  = sum(reshape(cat(1, deltaSb{:}), [numParams(2), numDevices]), 2);
            deltaMwx = sum(reshape(cat(1, deltaMwx{:}), [numParams(3), numDevices]), 2);
            deltaSwx = sum(reshape(cat(1, deltaSwx{:}), [numParams(3), numDevices]), 2);
            deltaMbx = sum(reshape(cat(1, deltaMbx{:}), [numParams(4), numDevices]), 2);
            deltaSbx = sum(reshape(cat(1, deltaSbx{:}), [numParams(4), numDevices]), 2);            
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.catParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);            
            [mw, Sw]   = arrayfun(@twoPlus, mw, Sw, deltaMw, deltaSw);
            [mb, Sb]   = arrayfun(@twoPlus, mb, Sb, deltaMb, deltaSb);
            [mwx, Swx] = arrayfun(@twoPlus, mwx, Swx, deltaMwx, deltaSwx);
            [mbx, Sbx] = arrayfun(@twoPlus, mbx, Sbx, deltaMbx, deltaSbx);
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.distributeParameters2Layers(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx, numParamsPerlayer);
            theta = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);
        end
        
        % Noise update to be completed!!!
        function [l, v2] = detachMeanVar(x, nl, nv2, B, rB)
            x  = reshape(x, [nl+nv2, B*rB]);
            l  = reshape(x(1:nl, :), [B*nl, rB]);
            v2 = reshape(x(nl+1:end, :), [B*nv2, rB]);
        end
        function x = attachMeanVar(l, v2, nl, nv2, B, rB)
            l  = reshape(l, [nl, B*rB]);
            v2 = reshape(v2, [nv2, B*rB]);
            x  = [l;v2];
            x  = reshape(x, [(nl+nv2)*B, rB]);
        end
        function [deltaMlz, deltaSlz, deltaMv2z, deltaSv2z] = noiseUpdate4classification(Slz, mla, Sla, J, Jv2, mv2a, Sv2a, Cv2a, y, sv, udIdx, gpu)
            Cyz = J(udIdx).*Slz(udIdx) ;
            Syf = Sla(udIdx) + mv2a(udIdx) + Sv2a(udIdx)./(4*mv2a(udIdx)) + sv.^2;
            [deltaMlz, deltaSlz] = tagi.fowardHiddenStateUpdate(mla(udIdx), Syf, Cyz, y(udIdx), gpu);
            [deltaMv, deltaSv] = tagi.fowardHiddenStateUpdate(mla(udIdx), Syf, mv2a(udIdx) + Sv2a(udIdx)./(4*mv2a(udIdx)), y(udIdx), gpu);
            mvUd = deltaMv;
            SvUd = mv2a(udIdx) + Sv2a(udIdx)./(4*mv2a(udIdx)) + deltaSv;
            % Update activated standard deviation for z
            yv2  = mvUd.^2 + SvUd;
            Sv2f = 2*SvUd.^2 + 4*(mvUd.^2).*SvUd;   
            [deltaMv2z, deltaSv2z] = tagi.noiseBackwardUpdate(mv2a(udIdx), 3*Sv2a(udIdx) + 2*mv2a(udIdx).^2, Jv2(udIdx).*Cv2a(udIdx), yv2, Sv2f, gpu); 
        end
        function [deltaMlz, deltaSlz, deltaMv2z, deltaSv2z, deltaMv2a, deltaSv2a] = noiseUpdate4regression(Slz, mla, Sla, J, Jv2, mv2a, Sv2a, Cv2a, y, gpu)           
            Cyz  = J.*Slz ;
            Syf  = Sla + mv2a;                           

            [deltaMlz, deltaSlz] = tagi.fowardHiddenStateUpdate(mla, Syf, Cyz, y, gpu);        
            [deltaMv, deltaSv]   = tagi.fowardHiddenStateUpdate(mla, Syf, mv2a, y, gpu);       
            
            mvUd  = deltaMv;
            SvUd  = mv2a + deltaSv;                                
            
            % Update activated standard deviation for z          
            mv2y            = mvUd.^2 + SvUd;
            sv2y            = 2*SvUd.^2 + 4*(mvUd.^2).*SvUd;
            mv2             = mv2a;                                
            sv2             = 3*Sv2a + 2*mv2a.^2;
            C_v2_v2hat      = Sv2a;

            % Smoother Update
            [deltaMv2a, deltaSv2a]      = tagi.noiseBackwardUpdateA(mv2,sv2,C_v2_v2hat,mv2y,sv2y, gpu);                                                                %BD
            [deltaMv2z, deltaSv2z]      = tagi.noiseBackwardUpdate(mv2,sv2, Jv2.*Cv2a, mv2y, sv2y, gpu);
%             mv2hat_LL_pos = v2hat_LL(1);
%             Sv2hat_LL_pos = v2hat_LL(2);
                
                

        end
        function [deltaMlz, deltaSlz, deltaMv2z, deltaSv2z, deltaMv2a, deltaSv2a, mv2hat_LL_pos, Sv2hat_LL_pos] = noiseOnlyUpdate4regression(Jv2, mv2a, Sv2a, Cv2a, E_v2hat, var_v2hat, C_v2hat, varW, v2hat_LL, y, gpu,epoch,idx_LL_M)                          %BD
%           Cyz = J.*Slz ;
            Syf = varW;
            mla = 0;
%             [deltaMlz, deltaSlz] = tagi.fowardHiddenStateUpdate(mla, Syf, Cyz, y, gpu);
            [deltaMv, deltaSv] = tagi.fowardHiddenStateUpdate(mla, Syf, varW, y, gpu);
            
            mvUd     = deltaMv;
            SvUd     = varW + deltaSv;
            deltaMlz = mvUd;
            deltaSlz = SvUd;
            
            % Update activated standard deviation for z          
            mv2y            = mvUd.^2 + SvUd;
            sv2y            = 2*SvUd.^2 + 4*(mvUd.^2).*SvUd;
%             mv2a            = varW;     %comment
            if E_v2hat == varW
                mv2             = E_v2hat;
                sv2             = 3*var_v2hat + 2*E_v2hat.^2;
            else
                mv2             = varW;
%                 sv2             = 2*varW^2;
                sv2             = 3*var_v2hat + 2*varW.^2;
            end
%             sv2             = 3*Sv2a + 2*mv2a.^2;
%             C_v2_v2hat      = var_v2hat;
%             Jv2             = 1;
            % Smoother Update
            if ~isempty(C_v2hat)
                [deltaMv2a, deltaSv2a]     = tagi.noiseBackwardUpdateA(mv2,sv2,C_v2hat,mv2y,sv2y);                                                                %BD
                if ~isempty(idx_LL_M)
                    % Prior moments of H = [V1+V2 V1  V2]^T
                    mv2hat_H_prior  =  [mv2a;v2hat_LL(1)];
                    Sv2hat_H_prior  =  diag([Sv2a,v2hat_LL(2)]);
                    if epoch == 1
                        % prior of v2hat
                        Prior_mv2hat     = mv2a + v2hat_LL(1);
                        Prior_sv2hat     = Sv2a + v2hat_LL(2);
                        % computing posterior of v2hat
                        Pos_mv2hat       = Prior_mv2hat     + deltaMv2a;                                    %BD
                        Pos_sv2hat       = Prior_sv2hat     + deltaSv2a;                                    %BD
                        
                        if idx_LL_M == 1
                            C_v1v2_v1 = Sv2a;
                            C_v1v2_v2 = v2hat_LL(2);
                        elseif idx_LL_M == 2
                            C_v1v2_v1 = Sv2a*v2hat_LL(1);
                            C_v1v2_v2 = v2hat_LL(2)*mv2a;
                        end
                        K_v2hat       = [C_v1v2_v1./var_v2hat; C_v1v2_v2./var_v2hat];
                        
                        mv2hat_H_pos  = mv2hat_H_prior + K_v2hat*(Pos_mv2hat-Prior_mv2hat);
                        Sv2hat_H_pos  = Sv2hat_H_prior + K_v2hat*(Pos_sv2hat-Prior_sv2hat)*K_v2hat';
                        mv2hat_1_pos  = mv2hat_H_pos(1);
                        sv2hat_1_pos  = Sv2hat_H_pos(1,1);
                        mv2hat_LL_pos = mv2hat_H_pos(2);
                        Sv2hat_LL_pos = Sv2hat_H_pos(2,2);
                        % Smoother equations for aV2hat_1 --> V2hat_1
                        J_mv2hat  = K_v2hat*(Pos_mv2hat-Prior_mv2hat);
                        J_sv2hat  = K_v2hat*(Pos_sv2hat-Prior_sv2hat)*K_v2hat';
                        deltaMv2z = J_mv2hat(1);
                        deltaSv2z = J_sv2hat(1,1);
                    else
                        [deltaMv2z, deltaSv2z]  = tagi.noiseBackwardUpdate(mv2,sv2, Jv2.*Cv2a, mv2y, sv2y, gpu);
                        mv2hat_LL_pos = v2hat_LL(1);
                        Sv2hat_LL_pos = v2hat_LL(2);
                    end
                else
                    [deltaMv2z, deltaSv2z]  = tagi.noiseBackwardUpdate(mv2,sv2, Jv2.*Cv2a, mv2y, sv2y, gpu);
                    mv2hat_LL_pos = v2hat_LL(1);
                    Sv2hat_LL_pos = v2hat_LL(2);
                    %                     deltaSv2z     =  0;
                    
                end
%                 [deltaMv2z, deltaSv2z]     = tagi.noiseBackwardUpdate(mv2, sv2, Jv2.*Cv2a, mv2y, sv2y, gpu);
                % Only learning expected value
%                 deltaSv2a = 0;
%                 deltaSv2z = 0;
            else
                [deltaMv2z, deltaSv2z]     = tagi.noiseBackwardUpdate(mv2, sv2, Sv2a, mv2y, sv2y, gpu);
                deltaMv2a = [];
                deltaSv2a = [];
            end
        end
        function [deltaMlz, deltaSlz, deltaMv2z, deltaSv2z, mv2a, Sv2a] = homoNoiseUpdate4regression(Slz, mla, Sla, J, mv2a, Sv2a, y, gpu)
            Cyz = J.*Slz ;
            Syf = Sla + mv2a;
            
            [deltaMlz, deltaSlz] = tagi.fowardHiddenStateUpdate(mla, Syf, Cyz, y, gpu);
            [deltaMv, deltaSv] = tagi.fowardHiddenStateUpdate(mla, Syf, mv2a, y, gpu);
            
            mvUd = deltaMv;
            SvUd = mv2a + deltaSv;
            
            % Update activated standard deviation for z
            pos_mv2  = mvUd.^2 + SvUd;
            pos_sv2  = 2*SvUd.^2 + 4*(mvUd.^2).*SvUd; 
%             prior_mv2hat = zeros(size(mvUd,1),1);
%             prior_Sv2hat = zeros(size(mvUd,1),1);
%             prior_mv2hat(1) = mv2a;
%             prior_Sv2hat(1) = Sv2a;
            i = 1;
            while i <= size(mvUd,1)
                Pr_mv2   = mv2a;
                Pr_sv2   = 3*Sv2a + 2*mv2a.^2;
                Cv2a     = Sv2a;
                [deltaMv2z, deltaSv2z] = tagi.noiseBackwardUpdate(Pr_mv2, Pr_sv2, Cv2a, pos_mv2(i), pos_sv2(i), gpu);
                mv2a = mv2a + deltaMv2z;
                Sv2a = Sv2a + deltaSv2z;
%                 prior_mv2hat(i+1) = mv2a;
%                 prior_Sv2hat(i+1) = Sv2a;
                i = i+1;
            end
            
        end
        function [mlz, Slz, mv2z, Sv2z] = noiseUpdate4decoding(mlz, Slz, mla, Sla, J, mv2z, Sv2z, Jv2, mv2a, Sv2a, Cv2a, y, sv, nl, nv2, B, gpu)
            if nl~=nv2
                mv2aM = reshape(repmat(mv2a', [nl, 1]), [B*nl, 1]);
            else
                mv2aM = mv2a;
            end
            Cyz   = J.*Slz ;
            Syf   = Sla + mv2aM + sv.^2;
            [mlz, Slz] = tagi.fowardHiddenStateUpdate(mlz, Slz, mla, Syf, Cyz, y, gpu);
            [deltaM, deltaS] = tagi.forwardNoiseUpdate(mla, Syf, mv2aM, y, gpu);
            if nl~=nv2
                deltaM = sum(reshape(deltaM, [nl, B]), 1);
                deltaS = sum(reshape(deltaS, [nl, B]), 1);
                mvUd   = deltaM';
                SvUd   = mv2a + deltaS';
            else
                mvUd   = deltaM;
                SvUd   = mv2a + deltaS;
            end
            if any(isnan(SvUd))
                check=1;
            end
            % Update activated standard deviation for z
            yv2  = mvUd.^2 + SvUd;
            Sv2f = Sv2a + 2*mv2a.^2 + 2*SvUd.^2 + 4*(mvUd.^2).*SvUd;   
            [mv2z, Sv2z] = tagi.noiseBackwardUpdate(mv2z, Sv2z, mv2a, Sv2a + 2*mv2a.^2, Jv2.*Cv2a, yv2, Sv2f, gpu);
            if any(isnan(Sv2z))
                check=1;
            end
        end
        function [deltaMlz, deltaSlz, deltaMv2z, deltaSv2z] = noiseUpdate4encoding(Slz, mla, Sla, J, Jv2, mv2a, Sv2a, Cv2a, y, Sy, gpu)
            Cyz = J.*Slz ;
            [deltaMlz, deltaSlz] = tagi.noiseBackwardUpdate_V2(mla, Sla+mv2a, Cyz, y, Sy, gpu);
            [deltaMv, deltaSv] = tagi.noiseBackwardUpdate(mla, Sla+mv2a, mv2a, y, Sy, gpu);
            mvUd = deltaMv;
            SvUd = mv2a + deltaSv;
            % Update activated standard deviation for z
            yv2  = mvUd.^2 + SvUd;
            Sv2f = Sv2a +2*SvUd.^2 + 4*(mvUd.^2).*SvUd;   
            [deltaMv2z, deltaSv2z] = tagi.noiseBackwardUpdate_V2(mv2a, Sv2a + 2*mv2a.^2, Jv2.*Cv2a, yv2, Sv2f, gpu);
        end
        function [deltaMv2a, deltaSv2a] = noiseBackwardUpdateA(mv2,sv2,C_v2_v2hat,mv2y,sv2y,gpu)                                                               %BD
            if gpu == 1
               funM        = @(x, y, z) x.*(y-z);
               funS        = @(x, y, z) x.*(y-z).*x;
               J           =  C_v2_v2hat./(sv2);
               deltaMv2a   =  arrayfun(funM,J,mv2y,mv2);
               deltaSv2a   =  arrayfun(funS,J,sv2y,sv2);
            else
                J           =  C_v2_v2hat./(sv2);
                deltaMv2a   =  J.*(mv2y - mv2); 
                deltaSv2a   =  J.^2.*sv2y - C_v2_v2hat.^2./(sv2);
            end
        end
        function [deltaMz, deltaSz] = noiseBackwardUpdate(maF, SaF, CzzF, maB, SaB, gpu)            
            if gpu == 1
                funM    = @(x, y, z) x.*(y-z);
                funS    = @(x, y, z) x.*(y-z).*x;
                Jz      = CzzF./SaF; 
                deltaMz = arrayfun(funM, Jz, maB, maF);
                deltaSz = arrayfun(funS, Jz, SaB, SaF);
            else
                Jz      = CzzF./SaF; 
                deltaMz = Jz.*(maB - maF);
                deltaSz = Jz.*(SaB - SaF).*Jz;
            end
        end  
        
        % Initialization for weights and bias   
        function theta  = initializeWeightBias(net)
%             rng(1223)           % uncommented --BD
%             Initialization
              
%             rng('default')
            nodes     = double(net.nodes);
            numLayers = length(net.nodes);
            layer     = net.layer;
            idxw      = net.idxw;
            idxwXsc   = net.idxwXsc;
            idxbXsc   = net.idxbXsc;
            idxb      = net.idxb;
            biasStd   = 1E-2;
%             B         = single(net.batchSize);
            B         = 1;
            rB        = net.repBatchSize;
            noParam   = nan;
            gainM     = cast(net.gainM, net.dtype);
            gainS     = cast(net.gainS, net.dtype);            
            mw        = tagi.createInitCellwithArray(numLayers-1);
            Sw        = tagi.createInitCellwithArray(numLayers-1);
            mb        = tagi.createInitCellwithArray(numLayers-1);
            Sb        = tagi.createInitCellwithArray(numLayers-1);
            mwx       = tagi.createInitCellwithArray(numLayers-1);
            Swx       = tagi.createInitCellwithArray(numLayers-1);
            mbx       = tagi.createInitCellwithArray(numLayers-1);
            Sbx       = tagi.createInitCellwithArray(numLayers-1);
            beta      = 1;
            for j = 2:numLayers
                if ~isempty(idxw{j-1})                    
                    if layer(j) == net.layerEncoder.conv || layer(j) == net.layerEncoder.tconv % Conv. layer
                        fanIn  = (cast(net.kernelSize(j-1), net.dtype).^2)*cast(net.filter(j-1), net.dtype);
                        if net.xsc(j-1)~=0
                            fanIn = 2*fanIn;
                        end
                        if strcmp(net.initParamType, 'Xavier')
                            if j<numLayers&&(layer(j+1) == net.layerEncoder.mp || layer(j+1) == net.layerEncoder.ap)
                                fanOut= ((cast(net.kernelSize(j-1), net.dtype).^2)*cast(net.filter(j), net.dtype))/(cast(net.kernelSize(j), net.dtype).^2);
                            else
                                fanOut= ((cast(net.kernelSize(j-1), net.dtype).^2)*cast(net.filter(j), net.dtype));
                            end
                            Sw{j-1} = (gainS(j-1))*(2/(fanIn+fanOut))*ones(length(idxw{j-1}), 1, net.dtype);
                        elseif strcmp(net.initParamType, 'He')
                            ind_w_o  = reshape(idxw{j-1},[fanIn,fanOut]); %BD
                            Sw_mean  = (gainS(j-1))*(1/(fanIn))*ones(size(ind_w_o,1), 1, net.dtype); %BD
                            Sw_v2hat = (net.gainS_v2hat)*(1/(fanIn))*ones(size(ind_w_o,1), 1, net.dtype); %BD
                            Sw{j-1}  = [Sw_mean;Sw_v2hat];%BD
                            %mw{j-1}  = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(Sw{j-1});
                            %Sw{j-1} = (gainS(j-1))*(1/(fanIn))*ones(length(idxw{j-1}), 1, net.dtype);
                        end 
                        mw{j-1} = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(Sw{j-1});
                        if ~isempty(idxb{j-1})
                            Sb{j-1} = (1/fanIn)*ones(length(idxb{j-1}), 1, net.dtype);
                            mb{j-1} = randn(length(Sb{j-1}), 1).*sqrt(Sb{j-1});
                        end
                    elseif layer(j) == net.layerEncoder.ln || layer(j) == net.layerEncoder.bn
                        Sb{j-1} = 1E-4*gainS(j-1)*ones(length(idxb{j-1}), 1, net.dtype);
                        mb{j-1} = 0*rand(length(Sb{j-1}), 1, net.dtype).*sqrt(Sb{j-1});
                        Sw{j-1} = 1*ones(length(idxw{j-1}), 1, net.dtype);
                        mw{j-1} = 1*ones(length(idxw{j-1}), 1, net.dtype);
                    else
                        fanIn  = nodes(j-1);
                        fanOut = nodes(j);
                        if strcmp(net.initParamType, 'Xavier')
                            Sw{j-1} = (gainS(j-1))*(2/(fanIn+fanOut))*ones(length(idxw{j-1}), 1, net.dtype);
                            mw{j-1} = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(Sw{j-1});
%                             if j == numLayers
%                                 ind_w_o  = reshape(idxw{j-1},[fanIn,fanOut]);                                % indices for the outputs, i.e., mean and V2hat  -- BD
%                                 Sw_mean  = (gainS(j-1))*(2/(fanIn+fanOut))*ones(size(ind_w_o,1), 1, net.dtype);
%                                 Sw_v2hat = (net.gainS_v2hat)*(2/(fanIn+fanOut))*ones(size(ind_w_o,1), 1, net.dtype);
%                                 Sw{j-1}  = [Sw_mean;Sw_v2hat];
%                                 mw{j-1}  = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(Sw{j-1});
%                             else
%                                 Sw{j-1} = (gainS(j-1))*(2/(fanIn+fanOut))*ones(length(idxw{j-1}), 1, net.dtype);
%                                 mw{j-1} = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(Sw{j-1});
%                             end
                        elseif strcmp(net.initParamType, 'He')
                            if j == numLayers && fanOut == 2
                                ind_w_o  = reshape(idxw{j-1},[fanIn,fanOut]);                                % indices for the outputs, i.e., mean and V2hat  -- BD
                                Sw_mean  = (gainS(j-1))*(1/(fanIn*B))*ones(size(ind_w_o,1), 1, net.dtype);
                                Sw_v2hat = (net.gainS_v2hat)*(1/(fanIn*B))*ones(size(ind_w_o,1), 1, net.dtype);
                                Sw{j-1}  = [Sw_mean;Sw_v2hat];
                                mw{j-1}  = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(beta*Sw{j-1});
%                             elseif j == numLayers && fanOut == 1
%                                 Sw{j-1} = (net.gainS_v2hat).*(1/(fanIn))*ones(length(idxw{j-1}), 1, net.dtype);
%                                 mw{j-1} = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(Sw{j-1});
                            
                            else
                                Sw{j-1} = (gainS(j-1))*(1/(fanIn*B))*ones(length(idxw{j-1}), 1, net.dtype);
                                mw{j-1} = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(beta*Sw{j-1});
                            end
                            
                        end
                        
                        if ~isempty(idxb{j-1})
                            if j == numLayers && fanOut == 2                                                             %BD
                                Sb_mean  = (1/(fanIn*B));
                                Sb_v2hat = (net.gainSb_v2hat)*(1/(fanIn*B));
                                Sb{j-1}  = [Sb_mean;Sb_v2hat];

                                mb_mean  = randn(1).*sqrt(beta*Sb_mean);
%                                 mb_mean  = 0;
                                mb_v2hat = 0;
                                %mb_v2hat = randn(1).*sqrt(Sb_v2hat);
                                mb{j-1}  = [mb_mean;mb_v2hat];
                            elseif j == numLayers && fanOut == 1 && strcmp(net.init,'W')
                                Sb{j-1} = (1/(fanIn))*ones(length(idxb{j-1}), 1, net.dtype);
                                mb{j-1} = 1e-03;
                            elseif j == numLayers && fanOut == 1 && strcmp(net.init,'D')
                                Sb{j-1} = (1/(fanIn))*ones(length(idxb{j-1}), 1, net.dtype);
                                mb{j-1} = randn(1,1).*sqrt(Sb{j-1});
                            else
                                Sb{j-1} = (1/(fanIn))*ones(length(idxb{j-1}), 1, net.dtype);   %(1/(fanIn))
                                mb{j-1} = randn(length(Sb{j-1}), 1).*sqrt(beta*Sb{j-1});

                            end
                            
                            
                        end
                    end  
                else
                    mw{j-1} = noParam;
                    Sw{j-1} = noParam; 
                    Sb{j-1} = noParam;
                    mb{j-1} = noParam;
                end 
                if net.xsc(j)~=0&&(net.filter(net.xsc(j))~=net.filter(j)||net.imgW(net.xsc(j))~=net.imgW(j))
                    idxXsc = net.xsc(j);                                     
                    fanIn  = cast(net.filter(idxXsc), net.dtype);
                    fanOut = cast(net.filter(j), net.dtype);
                    if strcmp(net.initParamType, 'Xavier')
                        Swx{idxXsc} = (gainS(idxXsc))*(2/(fanIn+fanOut))*ones(length(idxwXsc{idxXsc}), 1, net.dtype);
                    elseif strcmp(net.initParamType, 'He')
                        Swx{idxXsc} = (1/(fanIn))*ones(length(idxwXsc{idxXsc}), 1, net.dtype);
                    end
                    mwx{idxXsc} = randn(length(Swx{idxXsc}), 1).*sqrt(Swx{idxXsc});
                    if ~isempty(idxbXsc{idxXsc})
                        Sbx{idxXsc} = 1E-6*ones(length(idxbXsc{idxXsc}), 1, net.dtype);
                        mbx{idxXsc} = 0*randn(length(Sbx{idxXsc}), 1).*sqrt(Sbx{idxXsc});
                    end                   
                    if net.gpu == 1
                        mwx{idxXsc} = gpuArray(mwx{idxXsc});
                        Swx{idxXsc} = gpuArray(Swx{idxXsc});
                        mbx{idxXsc} = gpuArray(mbx{idxXsc});
                        Sbx{idxXsc} = gpuArray(Sbx{idxXsc});
                    end
                end
                clear fanIn
                % Send to gpu
                if net.gpu == 1
                    mw{j-1} = gpuArray(mw{j-1});
                    Sw{j-1} = gpuArray(Sw{j-1});
                    mb{j-1} = gpuArray(mb{j-1});
                    Sb{j-1} = gpuArray(Sb{j-1});                    
                end
            end 
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.catParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);
           theta = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx); 
        end
        function [Sw, Sb, var_w_v2hat, var_b_v2hat]  = initializeOnlyVariance_HP_Gain(net)
            nodes                   = double(net.nodes);
            numLayers               = length(net.nodes);
            idxb                    = net.idxb;
            idxw                    = net.idxw;
            net.gainS               = net.gain_HP(1,1)*ones(1,length(net.layer)-1);
            net.gainS_v2hat         = net.gain_HP(2,1);
            net.var_gainS           = net.gain_HP(1,2)*ones(1,length(net.layer)-1);
            net.var_gainS_v2hat     = net.gain_HP(2,2);
            B = 1;
            % initialization
            Sw          = tagi.createInitCellwithArray(numLayers-1);
            Sb          = tagi.createInitCellwithArray(numLayers-1);
            var_w_v2hat = tagi.createInitCellwithArray(numLayers-1);
            var_b_v2hat = tagi.createInitCellwithArray(numLayers-1);
            for j = 2:numLayers
                    fanIn  = nodes(j-1);
                    fanOut = nodes(j);
                    if strcmp(net.initParamType, 'Xavier')
                        Sw{j-1} = (gainS(j-1))*(2/(fanIn+fanOut))*ones(length(idxw{j-1}), 1, net.dtype);


                    elseif strcmp(net.initParamType, 'He')
                        if j == numLayers && fanOut == 2
                            ind_w_o  = reshape(idxw{j-1},[fanIn,fanOut]);                                % indices for the outputs, i.e., mean and V2hat  -- BD
                            Sw_mean  = (net.gainS(j-1))*(1/(fanIn*B))*ones(size(ind_w_o,1), 1, net.dtype);
                            Sw_v2hat = (net.gainS_v2hat)*(1/(fanIn*B))*ones(size(ind_w_o,1), 1, net.dtype);
                            Sw{j-1}  = [Sw_mean;Sw_v2hat];
                            
                            var_w_v2hat_mean  = net.var_gainS(j-1)*(1/(fanIn^2*B))*ones(size(ind_w_o,1), 1, net.dtype);
                            var_w_v2hat_v2hat = net.var_gainS_v2hat*(1/(fanIn^2*B))*ones(size(ind_w_o,1), 1, net.dtype);
                            var_w_v2hat{j-1}  = [var_w_v2hat_mean;var_w_v2hat_v2hat];
    

                        else
                            Sw{j-1}          = (net.gainS(j-1))*(1/(fanIn*B))*ones(length(idxw{j-1}), 1, net.dtype);
                            var_w_v2hat{j-1} =  net.var_gainS(j-1)*(1/(fanIn*B))*ones(length(idxw{j-1}), 1, net.dtype);
    %                                 mw{j-1} = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(Sw{j-1});
                        end

                    end
                    if ~isempty(idxb{j-1})
                        if j == numLayers && fanOut == 2                                                             %BD
                            Sb_mean  = (1/(fanIn*B));
                            Sb_v2hat = (net.gainSb_v2hat)*(1/(fanIn*B));
                            Sb{j-1}  = [Sb_mean;Sb_v2hat];
                            var_b_v2hat_mean  = net.var_gainS(j-1)*(1/(fanIn^2*B));
                            var_b_v2hat_v2hat = net.var_gainS_v2hat*(1/(fanIn^2*B));
                            var_b_v2hat{j-1}  = [var_b_v2hat_mean;var_b_v2hat_v2hat];
                        else
                            Sb{j-1}          = (1/(fanIn))*ones(length(idxb{j-1}), 1, net.dtype);   %0.01*(1/(fanIn))
                            var_b_v2hat{j-1} =  net.var_gainS(j-1)*(1/(fanIn^2*B))*ones(length(idxb{j-1}), 1, net.dtype);
                        end
                        
                        
                    end

            end
           Sw  = cat(1, Sw{:});
           Sb  = cat(1, Sb{:});
           var_w_v2hat  = cat(1, var_w_v2hat{:});
           var_b_v2hat  = cat(1, var_b_v2hat{:});
           
        end
        function theta  = initializeWeights_HP_BNI(net, factor)
            nodes     = double(net.nodes);
            numLayers = length(net.nodes);
            idxb      = net.idxb;
            idxw      = net.idxw;
            gainM     = cast(net.gainM, net.dtype);
            gainS     = cast(net.gainS, net.dtype); 
            mw        = tagi.createInitCellwithArray(numLayers-1);
            Sw        = tagi.createInitCellwithArray(numLayers-1);
            mb        = tagi.createInitCellwithArray(numLayers-1);
            Sb        = tagi.createInitCellwithArray(numLayers-1);
            mwx       = tagi.createInitCellwithArray(numLayers-1);
            Swx       = tagi.createInitCellwithArray(numLayers-1);
            mbx       = tagi.createInitCellwithArray(numLayers-1);
            Sbx       = tagi.createInitCellwithArray(numLayers-1);
            beta_bias = factor(1);
                 for j = 2:numLayers
                        fanIn  = nodes(j-1);
                        fanOut = nodes(j);
                        if strcmp(net.initParamType, 'He')
                            if j == numLayers && fanOut == 2
                                ind_w_o  = reshape(idxw{j-1},[fanIn,fanOut]);                                % indices for the outputs, i.e., mean and V2hat  -- BD
                                Sw_mean  = (gainS(j-1))*(1/(fanIn))*ones(size(ind_w_o,1), 1, net.dtype);
                                Sw_v2hat = (net.gainS_v2hat)*(1/(fanIn))*ones(size(ind_w_o,1), 1, net.dtype);
                                Sw{j-1}  = factor(j-1)*(1-1/fanIn)*[Sw_mean;Sw_v2hat];
                                mw{j-1}  = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(Sw{j-1});
                            else
                                Sw{j-1} = factor(j-1)*(gainS(j-1)*(1-1/fanIn)/(fanIn))*ones(length(idxw{j-1}), 1, net.dtype);
                                mw{j-1} = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(Sw{j-1});
                            end
                            
                        end
                        
                        if ~isempty(idxb{j-1})
                            if j == numLayers && fanOut == 2                                                             %BD
                                Sb_mean  = (1/(fanIn));
                                Sb_v2hat = (net.gainSb_v2hat)*(1/(fanIn));
                                Sb{j-1}  = beta_bias*[Sb_mean;Sb_v2hat];

%                                 mb_mean  = randn(1).*sqrt(Sb_mean);
                                mb_mean  = randn(1).*sqrt(Sb_mean);
                                mb_v2hat = 0;
%                                 mb_v2hat = randn(1).*sqrt(Sb_v2hat);
                                mb{j-1}  = [mb_mean;mb_v2hat];
                            
                            else
                                Sb{j-1} = beta_bias*(1/(fanIn))*ones(length(idxb{j-1}), 1, net.dtype);
                                mb{j-1} = randn(length(Sb{j-1}), 1).*sqrt(Sb{j-1});

                            end
                            
                            
                        end
                end
           [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.catParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);
           theta = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx); 
        end
        function states = initializeStates(nodes, B, rB, xsc, dtype, gpu)
            % Normal net
            numLayers = length(nodes);          
            mz  = tagi.createStateCellarray(nodes, numLayers, B, rB, dtype, gpu); 
            Sz  = mz; 
            ma  = mz;
            Sa  = mz;
            J   = mz;
            % Residual net
            idx = xsc~=0;
            mdxs = cell(numLayers, 1);
            mdxs(idx) = mz(idx);
            Sdxs = mdxs;
            mxs  = mdxs;
            Sxs  = mdxs;
            states = tagi.compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs);
        end
        function [deltaxs, deltadxs] = initializeShortcutStateDelta(xsc, idxXsc, x, B, rB)
            layers   = xsc(xsc~=0);
            deltaxs  = cell(length(xsc), 1);
            deltadxs = cell(length(xsc), 1);
            for j = layers
                if ~isempty(idxXsc{j})
                    deltaxs{j}  = zeros(length(idxXsc{j})*B, rB, 'like', x{j});
                    deltadxs{j} = deltaxs{j};
                else
                    deltadxs{j} = zeros(size(x{j}), 'like', x{j});
                    deltaxs{j}  = zeros(size(x{j}), 'like', x{j});
                end
            end
        end
        function states = initializeInputs(states, mz0, Sz0, ma0, Sa0, J0, mdxs0, Sdxs0, mxs0, Sxs0, xsc)
            [mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs] = tagi.extractStates(states);
            % Normal net
            mz{1} = mz0;
            if any(isempty(Sz0))
                Sz{1} = zeros(size(mz0), 'like', mz0);
            else
                Sz{1} = Sz0;
            end
            if any(isempty(ma0))
                ma{1} = mz0;
            else
                ma{1} = ma0;
            end 
            if any(isempty(Sa0))
                Sa{1} = Sz{1};
            else
                Sa{1} = Sa0;
            end   
            if any(isempty(J0))
                J{1} = ones(size(mz0), 'like', mz0);
            else
                J{1} = J0;
            end  
            % Residual net
            if any(isempty(mdxs0))&&~all(xsc==0)
                mdxs{1} = mz0;
            else
                mdxs{1} = mdxs0;
            end
            if any(isempty(Sdxs0))&&~all(xsc==0)
                Sdxs{1} = zeros(size(mz0), 'like', mz0);
            else
                Sdxs{1} = Sdxs0;
            end
            if any(isempty(mxs0))&&~all(xsc==0)
                mxs{1} = mz0;
            else
                mxs{1} = mxs0;
            end
            if any(isempty(Sxs0))&&~all(xsc==0)
                Sxs{1} = zeros(size(mz0), 'like', mz0);
            else
                Sxs{1} = Sxs0;
            end
            states = tagi.compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs);
        end
        function maxIdx = initializeMaxPoolingIndices(nodes, layers, layerEncoder, B, rB, dtype, gpu)
            if gpu==1
                zeroPad = zeros(1, 1, dtype, 'gpuArray');
            else
                zeroPad = zeros(1, 1, dtype);
            end
            numLayers = length(nodes);
            maxIdx = cell(numLayers, 1);
            maxPoolingLayers = find(layers==layerEncoder.mp);
            if ~isempty(maxPoolingLayers)
                for j = maxPoolingLayers
                    maxIdx{j} = zeros(nodes(j)*B, rB, 'like', zeroPad);
                end
            end
        end
        function normStat = initializeNormStat(nodes, filter, B, rB, layers, layerEncoder, x)
            numLayers = length(nodes);
            mra = cell(numLayers, 1);
            layNorm = layers==layerEncoder.ln;
            batNormConv = layers==layerEncoder.bn&(layers==layerEncoder.conv|layers==layerEncoder.tconv|layers==layerEncoder.mp|layers==layerEncoder.ap);
            batNormfc = layers==layerEncoder.bn&layers==layerEncoder.fc;
            for j = layNorm
                mra{j} = zeros(B, rB, 'like', x);
            end
            for j = batNormfc
                mra{j} = zeros(nodes(j), rB, 'like', x);
            end
            for j = batNormConv
                mra{j} = zeros(filter(j), rB, 'like', x);
            end
            Sra = mra;
            normStat = tagi.compressNormStat(mra, Sra);
        end  
        function deltaTheta = initializeDeltaTheta(theta, rB, numLayers)
            deltaTheta = cell(numLayers-1, 1);
            for j = 1:numLayers-1
                deltaTheta{j} = repmat(theta{j}, [1, rB]);
            end
        end
        function mw = fcOrthWeights(mw, Sw, ni, no)
            M = reshape(mw, [ni, no])';
            [r,c] = size(M);
            if r == c
                [~,~,W] = svd(M);
                mw = reshape(W', [ni*no, 1]);
            elseif r > c
                N = M(1:c,:);
                [~,~,W] = svd(N);
                W  = [W;M(c+1:end,:)];
                mw = reshape(W', [ni*no, 1]);
            else
                d = c-r;
                D = randn(d,c)*sqrt(Sw(1));
                M =[M;D];
                [~,~,W] = svd(M);
                W  = W(1:r,:);
                mw = reshape(W', [ni*no, 1]);
            end          
        end
        function mw = convOrthWeights(mw, Sw, ki, fi, fo)
            M = reshape(mw, [ki*ki*fi, fo])';
            [r,c] = size(M);
            if r == c
                [~,~,W] = svd(M);
                 mw = reshape(W', [ki*ki*fi*fo, 1]);
            elseif r > c
                N = M(1:c,:);
                [~,~,W] = svd(N);
                W  = [W;M(c+1:end,:)];
                mw = reshape(W', [ki*ki*fi*fo, 1]);
            else
                d = c-r;
                D = randn(d,c)*sqrt(Sw(1));
                M =[M;D];
                [~,~,W] = svd(M);
                W  = W(1:r,:);
                mw = reshape(W', [ki*ki*fi*fo, 1]);
            end            
        end
        function [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = catParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx)
            mw  = cat(1, mw{:});
            Sw  = cat(1, Sw{:});
            mb  = cat(1, mb{:});
            Sb  = cat(1, Sb{:});
            mwx = cat(1, mwx{:});
            Swx = cat(1, Swx{:});
            mbx = cat(1, mbx{:});
            Sbx = cat(1, Sbx{:});
        end
        function [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = distributeParameters2Layers(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx, numParams)
            mw  = mat2cell(mw, numParams(1, :));
            Sw  = mat2cell(Sw, numParams(1, :));
            mb  = mat2cell(mb, numParams(2, :));
            Sb  = mat2cell(Sb, numParams(2, :));
            mwx = mat2cell(mwx, numParams(3, :));
            Swx = mat2cell(Swx, numParams(3, :));
            mbx = mat2cell(mbx, numParams(4, :));
            Sbx = mat2cell(Sbx, numParams(4, :));
        end    
        
        % Storing
        function states = compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs)
            states = cell(9, 1);
            states{1} = mz;
            states{2} = Sz;
            states{3} = ma;
            states{4} = Sa;
            states{5} = J;
            states{6} = mdxs;
            states{7} = Sdxs;
            states{8} = mxs;
            states{9} = Sxs;
        end
        function [mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs] = extractStates(states)
            mz   = states{1};
            Sz   = states{2};
            ma   = states{3};
            Sa   = states{4};
            J    = states{5};
            mdxs = states{6};
            Sdxs = states{7};
            mxs  = states{8};
            Sxs  = states{9};
        end
        function [mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs] = extractStatesMultiGPUs(states)
            spmd
                mz   = states{1};
                Sz   = states{2};
                ma   = states{3};
                Sa   = states{4};
                J    = states{5};
                mdxs = states{6};
                Sdxs = states{7};
                mxs  = states{8};
                Sxs  = states{9};
            end
        end
        function theta = compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx)
            theta     = cell(8, 1);
            theta{1}  = mw;
            theta{2}  = Sw;
            theta{3}  = mb;
            theta{4}  = Sb;
            theta{5}  = mwx;
            theta{6}  = Swx;
            theta{7}  = mbx;
            theta{8}  = Sbx;
        end
        function [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = extractParameters(theta)
            mw  = theta{1};
            Sw  = theta{2};
            mb  = theta{3};
            Sb  = theta{4};
            mwx = theta{5};
            Swx = theta{6};
            mbx = theta{7};
            Sbx = theta{8};
        end
        function [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = extractParameters_V2(theta)
            mw  = theta(1, :);
            Sw  = theta(2, :);
            mb  = theta(3, :);
            Sb  = theta(4, :);
            mwx = theta(5, :);
            Swx = theta(6, :);
            mbx = theta(7, :);
            Sbx = theta(8, :);
        end
        function normStat = compressNormStat(mra, Sra)
            normStat = cell(2, 1);
            normStat{1} = mra;
            normStat{2} = Sra;
        end
        function [mra, Sra] = extractNormStat(normStat)
            mra = normStat{1};
            Sra = normStat{2};
        end   
        
        % Create cell with an array
        function x = createInitCellwithArray(numLayers)
            x = cell(numLayers, 1);
            x(:) = {nan};
        end
        function z = createStateCellarray(nodes, numLayers, B, rB, dtype, gpu)   
            z = cell(numLayers, 1);
            if gpu == 1
                zeroPad = zeros(1,1,dtype, 'gpuArray');
            else
                zeroPad = zeros(1,1,dtype);
            end
            for j = 2:numLayers               
                z{j} = zeros(nodes(j)*B, rB, 'like', zeroPad);
            end
        end                       
        function normStat = createInitNormStat(net)
            mra    = cell(length(net.nodes) -1, 1);            
            Sra    = cell(length(net.nodes) -1, 1);
            if net.gpu == 1
                mra(:) = {zeros(1, 1, net.dtype, 'gpuArray')};
                Sra(:) = {zeros(1, 1, net.dtype, 'gpuArray')};
            else
                mra(:) = {zeros(1, 1, net.dtype)};
                Sra(:) = {zeros(1, 1, net.dtype)};
            end
            normStat = tagi.compressNormStat(mra, Sra);
        end
    end
end