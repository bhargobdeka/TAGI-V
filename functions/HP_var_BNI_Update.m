function [theta, HP_BNI] = HP_var_BNI_Update(theta, Mw2hat_prior, Sw2hat_prior, M_theta2_prior, S_theta2_prior, J_w2hat)

[mw_pos, Sw_pos, mb_pos, Sb_pos]  = tagi.extractParameters(theta);
M_theta2_w_pos    = mw_pos.^2 + Sw_pos;
S_theta2_w_pos    = 2.*Sw_pos.^2+4.*Sw_pos.*mw_pos.^2;
M_theta2_b_pos    = mb_pos.^2 + Sb_pos;
S_theta2_b_pos    = 2.*Sb_pos.^2+4.*Sb_pos.*mb_pos.^2;
M_theta2_pos      = [M_theta2_w_pos; M_theta2_b_pos];
S_theta2_pos      = [S_theta2_w_pos; S_theta2_b_pos];
Mw2hat_pos        = Mw2hat_prior + J_w2hat.*(M_theta2_pos - M_theta2_prior);
Sw2hat_pos        = Sw2hat_prior + J_w2hat.*(S_theta2_pos - S_theta2_prior).*J_w2hat;
HP_BNI(:,1)       = Mw2hat_pos;
HP_BNI(:,2)       = Sw2hat_pos;
end