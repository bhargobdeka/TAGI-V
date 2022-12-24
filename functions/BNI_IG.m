function varW = BNI_IG(m_v2hat,var_v2hat)
%% Inverse Gamma
syms a b

% Mean
try
    S=solve([m_v2hat==b/(a-1),var_v2hat==b^2/((a-1)^2*(a-2))],[a b]);
%     S=solve([m_v2hat==b/(a+1),var_v2hat==b^2/((a-1)^2*(a-2))],[a b]);
catch
    warning('cannot solve');
end
aE=double(S.a);
bE=double(S.b);
% aO=double(real(S.a(3)));
% bO=double(real(S.b(3)));

% % Mode 
% S=solve([m_v2hat==b/(a+1),var_v2hat==b^2/((a-1)^2*(a-2))],[a b]);
% aO=double(real(S.a(3)));
% bO=double(real(S.b(3)));

%% Student's T
vE=2*aE;
hs2E=bE/aE;

% vO=2*aO;
% hs2O=bO/aO;

varW=hs2E*vE/(vE-2); %mean
% varW=hs2O*vO/(vO-2); %mode
end