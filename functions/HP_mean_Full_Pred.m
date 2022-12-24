function [theta, m_theta, S_theta, m_mu_theta, S_mu_theta, J_mu] = HP_mean_Full_Pred(theta, xv_HP)

theta_mu = theta;
[mw_mu, Sw_mu, mb_mu, Sb_mu, mwx, Swx, mbx, Sbx]  = tagi.extractParameters(theta_mu);
% mean of theta
mw = mw_mu; mb = mb_mu; Sw = Sw_mu; Sb = Sb_mu;
m_mu_theta = [mw;mb];
S_mu_theta = [Sw;Sb];
Sw_p       = Sw + xv_HP;
Sb_p       = Sb + xv_HP;
m_theta    = m_mu_theta;
S_theta    = [Sw_p;Sb_p];
J_mu       = S_mu_theta.*S_theta.^-1;
theta      = tagi.compressParameters(mw, Sw_p, mb, Sb_p, mwx, Swx, mbx, Sbx);
end

