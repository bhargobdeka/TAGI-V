function var_W = var_mode(EEW2g,SEW2g,aOg,mv2a,Sv2a)
        if mv2a > max(max(EEW2g))
            mv2a = max(max(EEW2g));
        elseif mv2a < min(min(EEW2g))
            mv2a = min(min(EEW2g));
        end
        if Sv2a > max(max(SEW2g))
            Sv2a = max(max(SEW2g));
        elseif Sv2a < min(min(SEW2g))
            Sv2a = min(min(SEW2g));
        end
        try
            aO   = interp2(EEW2g,SEW2g,aOg,mv2a,Sv2a);
        catch
            aO   = interp2(single(EEW2g),single(SEW2g),single(aOg),single(mv2a),single(Sv2a));
        end
        bO   =(mv2a*(aO+1));
        vO   = 2*aO;
        if vO == 2
            check;
        end
        hs2O  = bO/aO;
        SWO   = hs2O*vO/(vO-2); %mode

        var_W = SWO;
end