function [Prior_W, Prior_B, m_prior_W, s_prior_W, m_prior_B, s_prior_B] = createHPWeights(net, v2hat_W, v2hat_B)
m_prior_W      = cell(length(net.nodes)-1,1);
s_prior_W      = cell(length(net.nodes)-1,1);
m_prior_B      = cell(length(net.nodes)-1,1);
s_prior_B      = cell(length(net.nodes)-1,1);
for i = 1:length(net.nodes)-1
    m_prior_W{i}  = v2hat_W(1,i)*ones(net.nodes(i+1)*net.nodes(i),1);
    s_prior_W{i}  = v2hat_W(2,i)*ones(net.nodes(i+1)*net.nodes(i),1);
    m_prior_B{i}  = v2hat_B(1,i)*ones(net.nodes(i+1),1);
    s_prior_B{i}  = v2hat_B(2,i)*ones(net.nodes(i+1),1);
end
Prior_W   = [m_prior_W s_prior_W];
Prior_B   = [m_prior_B s_prior_B];

end