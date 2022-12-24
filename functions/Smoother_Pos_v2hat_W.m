function [Pos_v2hat_W, Pos_v2hat_B] = Smoother_Pos_v2hat_W(net,Prior_v2hat_W, Prior_v2hat_B, Pos_W2, Pos_B2, Prior_W2, Prior_B2, J_W, J_B)
Pos_v2hat_W  = cell(length(net.nodes)-1,2);
    Pos_v2hat_B  = cell(length(net.nodes)-1,2);
    for i = 1:length(net.nodes)-1
    % Mean update    
    Pos_v2hat_W{i,1}          = Prior_v2hat_W{i,1} + J_W{i}.*(Pos_W2{i,1}-Prior_W2{i,1});
    Pos_v2hat_B{i,1}          = Prior_v2hat_B{i,1} + J_B{i}.*(Pos_B2{i,1}-Prior_B2{i,1});
    % Var update
    Pos_v2hat_W{i,2}          = Prior_v2hat_W{i,2} + J_W{i}.*(Pos_W2{i,2}-Prior_W2{i,2}).*J_W{i};
    Pos_v2hat_B{i,2}          = Prior_v2hat_B{i,2} + J_B{i}.*(Pos_B2{i,2}-Prior_B2{i,2}).*J_B{i};
    end
end