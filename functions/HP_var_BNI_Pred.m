function [theta, Mw2hat_prior, Sw2hat_prior, M_theta2_prior, S_theta2_prior, J_w2hat] = HP_var_BNI_Pred(theta,net)

theta_var = theta;
[mw_var, ~, mb_var, ~,mwx, Swx, mbx, Sbx]  = tagi.extractParameters(theta_var);

mw = mw_var; mb = mb_var;
n_w_l1 = length(net.idxw{1});
%n_b_l1 = length(net.idxb{1});
n_w_l2 = length(net.idxw{2});
%n_b_l2 = length(net.idxb{2});

% Re-initialize with net.HP_BNI
% theta    = tagi.initializeWeights_HP_BNI(net);

% Prior of w2hat
Mw2hat_prior   = net.HP_BNI(:,1); %order of w and b??
Sw2hat_prior   = net.HP_BNI(:,2);
Sw = Mw2hat_prior(1:(n_w_l1+n_w_l2));
Sb = Mw2hat_prior((n_w_l1+n_w_l2+1):end);
% Regenerating theta
theta = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);
% Prior for theta^2
M_theta2_prior = Mw2hat_prior;
S_theta2_prior = 3.*Sw2hat_prior + 2.*Mw2hat_prior.^2;
J_w2hat        = Sw2hat_prior.*S_theta2_prior.^-1;
end