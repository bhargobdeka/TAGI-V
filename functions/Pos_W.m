function [Pos_v2hat_W, Pos_v2hat_B] = Pos_W(net,theta)
[mw_pos, Sw_pos, mb_pos, Sb_pos]  = tagi.extractParameters(theta);
    Pos_v2hat_W  = cell(3,2);
    Pos_v2hat_B  = cell(3,2);
    indexW       = net.numParamsPerlayer(1,:);
    indexB       = net.numParamsPerlayer(2,:);
    i = 0;
    k = 0;
    j = 1;
    while i <= length(indexW)
    Pos_v2hat_W{j,1} = mw_pos(i+1:indexW(j));
    Pos_v2hat_W{j,2} = Sw_pos(i+1:indexW(j));
    Pos_v2hat_B{j,1} = mb_pos(k+1:indexB(j));
    Pos_v2hat_B{j,2} = Sb_pos(k+1:indexB(j));
    i = indexW(j)+1;
    k = indexB(j)+1;
    j = j + 1;
    end
end