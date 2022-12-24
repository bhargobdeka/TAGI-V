function [Prior_v2hat_W2, Prior_v2hat_B2, J_v2hat_W2, J_v2hat_B2] = Pr_W_to_W2(net, Prior_v2hat_W, Prior_v2hat_B)
Prior_v2hat_W2    = cell(length(net.nodes)-1,2);
Prior_v2hat_B2    = cell(length(net.nodes)-1,2);
J_v2hat_W2        = cell(length(net.nodes)-1,1);
J_v2hat_B2        = cell(length(net.nodes)-1,1);
% mean of W^2
for i = 1:length(net.nodes)-1
    Prior_v2hat_W2{i,1} = Prior_v2hat_W{i,1};
    Prior_v2hat_B2{i,1} = Prior_v2hat_B{i,1};
    Prior_v2hat_W2{i,2} = 3*Prior_v2hat_W{i,2} + 2*Prior_v2hat_W{i,1}.^2;
    Prior_v2hat_B2{i,2} = 3*Prior_v2hat_B{i,2} + 2*Prior_v2hat_B{i,1}.^2;
    J_v2hat_W2{i,1}     = Prior_v2hat_W{i,2}./Prior_v2hat_W2{i,2};
    J_v2hat_B2{i,1}     = Prior_v2hat_B{i,2}./Prior_v2hat_B2{i,2};
end
end
