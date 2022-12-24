function [theta, m_theta_l1, S_theta_l1, m_theta_l2, S_theta_l2, m_mu_theta_l1, S_mu_theta_l1, m_mu_theta_l2, S_mu_theta_l2, J_mu_l1, J_mu_l2, n_w_l1, n_w_l2, n_b_l1] = HP_mean_Layerwise_Pred(theta, net)
theta_mu = theta;
[mw_mu, Sw_mu, mb_mu, Sb_mu, mwx, Swx, mbx, Sbx]  = tagi.extractParameters(theta_mu);
% mean of theta
mw = mw_mu; mb = mb_mu; Sw = Sw_mu; Sb = Sb_mu;

% lengths of w's and b's in each layer
n_w_l1 = length(net.idxw{1});
n_w_l2 = length(net.idxw{2});
n_b_l1 = length(net.idxb{1});

% mean and variance of layer 1
mw_l1 = mw(1:n_w_l1); mb_l1 = mb(1:n_b_l1);
Sw_l1 = Sw(1:n_w_l1); Sb_l1 = Sb(1:n_b_l1);
m_mu_theta_l1 = [mw_l1;mb_l1];
S_mu_theta_l1 = [Sw_l1;Sb_l1];
% mean and variance of layer 2
mw_l2         = mw(n_w_l1+1:end);   mb_l2    = mb(n_b_l1+1:end);
Sw_l2         = Sw(n_w_l1+1:end);   Sb_l2    = Sb(n_b_l1+1:end);
m_mu_theta_l2 = [mw_l2;mb_l2];
S_mu_theta_l2 = [Sw_l2;Sb_l2];
% variance of theta
Sw_p_l1       = Sw_l1 + net.xv_HP;
Sb_p_l1       = Sb_l1 + net.xv_HP;
Sw_p_l2       = Sw_l2 + net.xv_HP;     %./n_w_l1
Sb_p_l2       = Sb_l2 + net.xv_HP;
% theta vector layerwise
m_theta_l1    = m_mu_theta_l1;
m_theta_l2    = m_mu_theta_l2;
S_theta_l1    = [Sw_p_l1;Sb_p_l1];
S_theta_l2    = [Sw_p_l2;Sb_p_l2];
% Gain
J_mu_l1       = S_mu_theta_l1.*S_theta_l1.^-1;
J_mu_l2       = S_mu_theta_l2.*S_theta_l2.^-1;
% Concatenating variances for theta to one vector
Sw            = [Sw_p_l1;Sw_p_l2];
Sb            = [Sb_p_l1;Sb_p_l2];

% Regenerating theta
theta = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);
end