%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         act
% Description:  Activation function
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      November 12, 2019
% Updated:      October 20, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef act   
    methods (Static)
        function [m, S, J] = meanVar(z, mz, Sz, funIdx, B, rB, gpu)         
            if funIdx == 1 % tanh
                if gpu
                    dtanhf = @(x) 1-tanh(x).^2;
                    m = tanh(mz);
                    J = arrayfun(dtanhf, mz);
                else
                    dtanhf = @(x) 1-tanh(x).^2;
                    m = dtanhf(mz).*(z-mz)+tanh(mz);
                    J = dtanhf(z);  
                end
            elseif funIdx == 2 % sigmoid
                if gpu==1
                    sigmoid_mz  = exp(-mz);
                    sigmoid_mz  = bsxfun(@plus, 1, sigmoid_mz);
                    sigmoid_mz  = bsxfun(@rdivide, 1, sigmoid_mz);                  
                    dsigmoid_mz = bsxfun(@minus, 1, sigmoid_mz);
                    dsigmoid_mz = bsxfun(@times, sigmoid_mz, dsigmoid_mz);                                     
                    m  = sigmoid_mz;                     
                    J  = dsigmoid_mz;  
                else
                    sigmoid  = @(x) 1./(1+exp(-x));
                    dsigmoid = @(x) sigmoid(x).*(1-sigmoid(x));
                    m = sigmoid(mz);
                    J = dsigmoid(z);
                end
            elseif funIdx == 3 % cdf
                if gpu==1
                    m = normcdf(mz);
                    J = normpdf(mz);
                else
                    m  = normpdf(mz).*(z-mz)+normcdf(mz);
                    J  = normpdf(z);
                end
            elseif funIdx == 4 % relu
                if gpu==1
                    idx = mz>0;
                    J   = cast(idx, 'like', mz);
                    m   = bsxfun(@times, z, J);
                else
                    m   = max(0, mz);
                    J   = single(z>0);
                end            
            elseif funIdx == 5 % softplus
                if gpu==1
                    alpha = 2;
                    k = alpha*mz<1000;
                    e = bsxfun(@plus, 1, exp(alpha*mz.*k));
                    m = (log(e) + mz.*(1-k))/alpha;
                    J = k.*bsxfun(@rdivide, exp(alpha*mz.*k), e) + (1-k);
                else
                    m = log(1 + exp(mz));
                    J = 1./(1 + exp(-mz));
%                     k = alpha*mz<100;
%                     m = log(1 + exp(alpha*mz.*k))/alpha + mz.*(1-k)/alpha;
%                     J = k.*exp(alpha*mz.*k)./(1 + exp(alpha*mz.*k))+ (1-k);
                end
            elseif funIdx == 6 % leaky relu
                alpha = cast(0.2, 'like', mz);
                if gpu==1                   
                    idx = mz>0;
                    J   = cast(idx, 'like', mz);                   
                    m   = bsxfun(@times, z, J);  
                    J(~idx) = alpha;
                    m(~idx) = alpha*z(~idx);
                else
                    idx = mz>0;
                    m   = max(0, mz);
                    J   = single(z>0);
                    J(~idx) = alpha;
                    m(~idx) = alpha*z(~idx);
                end 
            elseif funIdx == 7 % exponential relu
                 alpha = cast(0.001, 'like', mz);
                if gpu == 1
                    idx = mz>0;
                    m   = mz;
                    m(~idx ) = alpha*(exp(mz(~idx)+0.5*Sz(~idx)) - 1);
                    J = cast(idx, 'like', mz);  
                    J(~idx) = alpha*exp(mz(~idx)+0.5*Sz(~idx));                          
                else
                    idx = mz>0;
                    m   = mz;
                    m(~idx ) = alpha*(exp(mz(~idx)+0.5*Sz(~idx)) - 1);
                    J = cast(idx, 'like', mz);  
                    J(~idx) = alpha*exp(mz(~idx)+0.5*Sz(~idx));
                end 
            elseif funIdx == 8
                if gpu==1
                    m = 1*sin(mz);
                    J = 1*cos(mz);
                else
                    m = sin(mz);
                    J = cos(mz);
                end
            elseif funIdx == 9
                alpha = 2;
                if gpu==1
                    dtanhf = @(x) 1-tanh(x).^2;
                    m = alpha.*(tanh(mz)+1);
                    J = arrayfun(dtanhf, mz);
                    J = alpha.*J;
                else
                    dtanhf = @(x) 1-tanh(x).^2;
                    m = alpha.*(tanh(mz)+1);
                    J = arrayfun(dtanhf, mz);
                    J = alpha.*J;
                end 
            elseif funIdx == 10 % softmax
                ny = length(mz)/(B*rB);
                mz = reshape(mz, [ny, B*rB]);
                if gpu==1
                    maxMz   = max(mz);
                    mzShift = bsxfun(@minus, mz, maxMz);
                    expMz   = exp(mzShift);
                    m       = bsxfun(@rdivide, expMz, sum(expMz));
                    m       = m(:);
                    fun     = @(x) (1-x).*x;
                    J       = arrayfun(fun, m);
                else
                    maxMz   = max(mz);
                    mzShift = bsxfun(@minus, mz, maxMz);
                    expMz   = exp(mzShift);
                    m       = bsxfun(@rdivide, expMz, sum(expMz));
                    fun     = @(x) (1-x).*x;
                    J       = arrayfun(fun, m);
                end
            
                
            end
            if gpu==1
                fun = @(x, y) (x.^2).*y;            
                S   = arrayfun(fun, J, Sz);
            else
                S = J.*Sz.*J;
            end
            if funIdx == 7
                S(~idx) = alpha^2*exp(2*mz(~idx)+Sz(~idx)).*(exp(Sz(~idx)) - 1);
            end
        end
        function [ma, Sa, Cza] = NoiseActFun(mz, Sz, funIdx, gpu)                                %BD
            if funIdx == 1
                if gpu == 1
                    ma = bsxfun(@plus,mz,0.5.*Sz);
                    ma = exp(ma);
                    fun = @(m,S) exp(2*m + S).*(exp(S) - 1);
                    Sa  = arrayfun(fun,mz,Sz);
%                     Sa  = exp(2*mz + Sz).*(exp(Sz) - 1);
                    Cza = Sz.*ma;
                else
                    alpha = 1;
                    ma  = alpha*exp(mz + 0.5*Sz);
                    Sa  = alpha^2*exp(2*mz + Sz).*(exp(Sz) - 1);
                    Cza = alpha*Sz.*exp(mz + 0.5*Sz);
                end
            elseif funIdx == 2
                if gpu==1
                    alpha = 2;
                    k     = alpha*mz<1000;
                    e     = bsxfun(@plus, 1, exp(alpha*mz.*k));
                    ma    = (log(e) + mz.*(1-k))/alpha;
                    J     = k.*bsxfun(@rdivide, exp(alpha*mz.*k), e) + (1-k);
                    Sa    = J.^2.*Sz;
                    Cza   = J.*Sz;
                else
                    ma    = log(1 + exp(mz));
                    J     = 1./(1 + exp(-mz));
                    Sa    = J.^2.*Sz;
                    Cza   = J.*Sz;
                end
            elseif funIdx == 3 %sigmoid             %BD
                if gpu==1
                    sigmoid_mz  = exp(-mz);
                    sigmoid_mz  = bsxfun(@plus, 1, sigmoid_mz);
                    sigmoid_mz  = bsxfun(@rdivide, 1, sigmoid_mz);                  
                    dsigmoid_mz = bsxfun(@minus, 1, sigmoid_mz);
                    dsigmoid_mz = bsxfun(@times, sigmoid_mz, dsigmoid_mz);                                     
                    ma    = sigmoid_mz;                     
                    J     = dsigmoid_mz;
                    Sa    = J.^2.*Sz;
                    Cza   = J.*Sz;
                else
                    sigmoid  = @(x) 1./(1+exp(-x));
                    dsigmoid = @(x) sigmoid(x).*(1-sigmoid(x));
                    ma       = sigmoid(mz);
                    J        = dsigmoid(mz);
                    Sa       = J.^2.*Sz;
                    Cza      = J.*Sz;
                end
            elseif funIdx == 4 % relu(X)            %BD
                    ma = mz;
                    J  = ones(length(mz),1);
%                 if isreal(mz)
                    if any(mz < 0)
                        ind = find(mz < 0);
                       ma(ind) = -mz(ind);
                       J(ind)  = -1;
%                     elseif any(mz < 0)
%                        ma = -mz;
%                        J  = -1;
                    end
%                 else
%                     check;
%                 end
%                 if exist('J','var') == 0
%                     check;
%                 end
                    Sa    = J.^2.*Sz;
                    Cza   = J.*Sz;
            elseif funIdx == 5 % mrelu 
                sz           = sqrt(Sz);
                alpha        = -mz./sz;
                omega        = max(1E-6,1-normcdf(alpha));
                beta         = normpdf(alpha)./omega;
                kappa        = 1+alpha.*beta-beta.^2;
                m_trunc      = mz+beta.*sz;
                ma           = m_trunc.*omega;
                Sa           = omega.*kappa.*Sz+omega.*(1-omega).*ma.^2;
                
                J            = sqrt(omega.*kappa);
                Cza          = J.*sqrt(kappa).*Sz;% 
                
            end
        end
        function [ma, Sa, Cza] = crossEntropy(Sz, y, p, gpu)
            if gpu==1
                ma  = sum(y.*log(p), 1);
                Sa  = sum(((y-p).^2*Sz), 1);
                Cza = (y-p).*Sz;
            else
                ma  = sum(y.*log(p), 1);
                Sa  = sum(((y-p).^2*Sz), 1);
                Cza = (y-p).*Sz;
            end
        end
        function Sa = softmaxVar(prob, Sz, B)
            N = size(prob, 1);
            K = N/B;
            Ru = repmat(eye(K)~=1, [B, 1]);
            Rd = repmat(eye(K), [B, 1]);
            Pm = reshape(repmat(reshape(prob, [K, B]), [K, 1]), [K, N])';
            Pm = -prob.*Pm.*Ru;
            Pd = (prob.*(1-prob)).*Rd;
            J = Pm + Pd;
            J2 = reshape((J').^2, [K, K, B]);
            Sz = reshape(Sz, [1, K, B]);
            Sa = sum(J2.*Sz, 2);
            Sa = Sa(:);
        end
        function [Pos_z_mv2hat,Pos_z_sv2hat] = Act2Z(Prior_z_mv2hat, Prior_z_sv2hat,Prior_act_mv2hat, Prior_act_sv2hat, Cv2a, Pos_act_mv2hat, Pos_act_Pv2hat)          %BD
            K_ZtoA           = Cv2a./Prior_act_sv2hat;
            Pos_z_mv2hat     =  Prior_z_mv2hat + K_ZtoA*(Pos_act_mv2hat - Prior_act_mv2hat);
            Pos_z_sv2hat     =  Prior_z_sv2hat + K_ZtoA^2*(Pos_act_Pv2hat - Prior_act_sv2hat);
        end
    end
end