function [Pos_W2, Pos_B2] = Pos_W_to_W2(net, Pos_W, Pos_B)
Pos_W2    = cell(length(net.nodes)-1,2);
Pos_B2    = cell(length(net.nodes)-1,2);
% mean of W^2
for i = 1:length(net.nodes)-1
    Pos_W2{i,1} = Pos_W{i,1}.^2 + Pos_W{i,2};
    Pos_B2{i,1} = Pos_B{i,1}.^2 + Pos_B{i,2};
    Pos_W2{i,2} = 2*Pos_W{i,2}.^2 + 4*Pos_W{i,1}.^2.*Pos_W{i,2};
    Pos_B2{i,2} = 2*Pos_B{i,2}.^2 + 4*Pos_B{i,1}.^2.*Pos_B{i,2};
end
end
