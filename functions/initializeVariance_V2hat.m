function [var_w_v2hat, var_b_v2hat]  = initializeVariance_V2hat(net, factor)
nodes                   = double(net.nodes);
numLayers               = length(net.nodes);
idxb                    = net.idxb;
idxw                    = net.idxw;
net.gainS               = net.gain_HP(1,1)*ones(1,length(net.layer)-1);
net.gainS_v2hat         = net.gain_HP(2,1);
net.var_gainS           = net.gain_HP(1,2)*ones(1,length(net.layer)-1);
net.var_gainS_v2hat     = net.gain_HP(2,2);
B = 1;
factor = 0.1;
% initialization
var_w_v2hat = tagi.createInitCellwithArray(numLayers-1);
var_b_v2hat = tagi.createInitCellwithArray(numLayers-1);
for j = 2:numLayers
    fanIn  = nodes(j-1);
    fanOut = nodes(j);
    
    if strcmp(net.initParamType, 'He')
        if j == numLayers && fanOut == 2
            ind_w_o  = reshape(idxw{j-1},[fanIn,fanOut]);                                % indices for the outputs, i.e., mean and V2hat  -- BD
            
            var_w_v2hat_mean  = (factor*net.var_gainS(j-1)*(1/(fanIn*B)))*ones(size(ind_w_o,1), 1, net.dtype);
            var_w_v2hat_v2hat = (factor*net.var_gainS_v2hat*(1/(fanIn*B)))*ones(size(ind_w_o,1), 1, net.dtype);
            var_w_v2hat{j-1}  = [var_w_v2hat_mean;var_w_v2hat_v2hat];
            
            
        else
            var_w_v2hat{j-1} =  (factor*net.var_gainS(j-1)*(1/(fanIn*B)))*ones(length(idxw{j-1}), 1, net.dtype);
            
        end
        
    end
    if ~isempty(idxb{j-1})
        if j == numLayers && fanOut == 2                                                             %BD
            var_b_v2hat_mean  = (factor*net.var_gainS(j-1)*(1/(fanIn*B)));
            var_b_v2hat_v2hat = (factor*net.var_gainS_v2hat*(1/(fanIn*B)));
            var_b_v2hat{j-1}  = [var_b_v2hat_mean;var_b_v2hat_v2hat];
        else
%             Sb{j-1}          = (1/(fanIn))*ones(length(idxb{j-1}), 1, net.dtype);   %0.01*(1/(fanIn))
            var_b_v2hat{j-1}  =  (factor*net.var_gainS(j-1)*(1/(fanIn*B)))*ones(length(idxb{j-1}), 1, net.dtype);
        end
        
        
    end
    
end

var_w_v2hat  = cat(1, var_w_v2hat{:});
var_b_v2hat  = cat(1, var_b_v2hat{:});

end