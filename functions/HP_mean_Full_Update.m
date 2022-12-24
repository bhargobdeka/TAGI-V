function theta = HP_mean_Full_Update(theta,m_theta, S_theta, m_mu_theta, S_mu_theta, J_mu, mwx, Swx, mbx, Sbx)
[mw_pos, Sw_pos, mb_pos, Sb_pos]  = tagi.extractParameters(theta);
m_theta_pos = [mw_pos;mb_pos];
S_theta_pos = [Sw_pos;Sb_pos];
m_mu_theta  = m_mu_theta       + J_mu.*(m_theta_pos - m_theta);
S_mu_theta  = S_mu_theta       + J_mu.*(S_theta_pos - S_theta).*J_mu;

m_mu_w_pos  = m_mu_theta(1:length(mw_pos)); m_mu_b_pos = m_mu_theta(length(mw_pos)+1:end);
S_mu_w_pos  = S_mu_theta(1:length(mw_pos)); S_mu_b_pos = S_mu_theta(length(mw_pos)+1:end);
theta       = tagi.compressParameters(m_mu_w_pos, S_mu_w_pos, m_mu_b_pos, S_mu_b_pos, mwx, Swx, mbx, Sbx);
end