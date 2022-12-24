function theta = HP_mean_layerwise_Update(theta, m_theta_l1, S_theta_l1, m_theta_l2, S_theta_l2, m_mu_theta_l1, S_mu_theta_l1, m_mu_theta_l2, S_mu_theta_l2, J_mu_l1, J_mu_l2, n_w_l1, n_w_l2, n_b_l1, mwx, Swx, mbx, Sbx)

[mw_pos, Sw_pos, mb_pos, Sb_pos]  = tagi.extractParameters(theta);
% Layer 1 - pos
mw_pos_l1 = mw_pos(1:n_w_l1); mb_pos_l1 = mb_pos(1:n_b_l1);
m_theta_l1_pos = [mw_pos_l1;mb_pos_l1];
Sw_pos_l1 = Sw_pos(1:n_w_l1); Sb_pos_l1 = Sb_pos(1:n_b_l1);
S_theta_l1_pos = [Sw_pos_l1;Sb_pos_l1];
% Layer 2 - pos
mw_pos_l2         = mw_pos(n_w_l1+1:end); mb_pos_l2 = mb_pos(n_b_l1+1:end);
m_theta_l2_pos    = [mw_pos_l2;mb_pos_l2];
Sw_pos_l2         = Sw_pos(n_w_l1+1:end); Sb_pos_l2 = Sb_pos(n_b_l1+1:end);
S_theta_l2_pos    = [Sw_pos_l2;Sb_pos_l2];

% Updating mu_theta layer-wise
% Layer 1
m_mu_theta_l1 = m_mu_theta_l1       + J_mu_l1.*(m_theta_l1_pos - m_theta_l1);
S_mu_theta_l1 = S_mu_theta_l1       + J_mu_l1.*(S_theta_l1_pos - S_theta_l1).*J_mu_l1;

% Layer 2
m_mu_theta_l2 = m_mu_theta_l2 + J_mu_l2.*(m_theta_l2_pos - m_theta_l2);
S_mu_theta_l2 = S_mu_theta_l2 + J_mu_l2.*(S_theta_l2_pos - S_theta_l2).*J_mu_l2;

% Arranging the weights
m_mu_w_pos = [m_mu_theta_l1(1:n_w_l1);m_mu_theta_l2(1:n_w_l2)];
m_mu_b_pos = [m_mu_theta_l1(n_w_l1+1:end);m_mu_theta_l2(n_w_l2+1:end)];

S_mu_w_pos = [S_mu_theta_l1(1:n_w_l1);S_mu_theta_l2(1:n_w_l2)];
S_mu_b_pos = [S_mu_theta_l1(n_w_l1+1:end);S_mu_theta_l2(n_w_l2+1:end)];

theta = tagi.compressParameters(m_mu_w_pos, S_mu_w_pos, m_mu_b_pos, S_mu_b_pos, mwx, Swx, mbx, Sbx);