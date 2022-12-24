function [Pos_W, Pos_B] = Pos_Weights(net,theta)
   [mw_pos, Sw_pos, mb_pos, Sb_pos]    = tagi.extractParameters(theta);
    Pos_W  = cell(length(net.nodes)-1,2);
    Pos_B  = cell(length(net.nodes)-1,2);
    indexW       = net.numParamsPerlayer(1,:);
    indexB       = net.numParamsPerlayer(2,:);
    i = 0;
    k = 0;
    j = 1;
    while j <= length(indexW)
    Pos_W{j,1} = mw_pos(i+1:indexW(j)+i);
    Pos_W{j,2} = Sw_pos(i+1:indexW(j)+i);
    Pos_B{j,1} = mb_pos(k+1:indexB(j)+k);
    Pos_B{j,2} = Sb_pos(k+1:indexB(j)+k);
    i = indexW(j);
    k = indexB(j);
    j = j + 1;
    end
end