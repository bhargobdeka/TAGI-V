clear;clc
%% LL
Layer = 2;metric=2;
if Layer==1
    %% LL
    filename1 = 'UCI_navalLLtest400Epoch.txt';
    filename2 = 'PBP/lltest_PBP.txt';
    filename3 = 'MCD/lltest_MCD.txt';
    TAGI_LL    = importdata(filename1);
    PBP_LL    = importdata(filename2);
    MCD_LL    = importdata(filename3);

    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 600, 400])
    plot(1:400,TAGI_LL,'r','LineWidth',1);hold on;
    plot(1:400,PBP_LL,'b','LineWidth',1);hold on;
    plot(1:400,MCD_LL,'g','LineWidth',1);
    xlabel('Epoch')
    ylabel('LL')
    %ylim([-7, -2.40])
    legend('TAGI-BNI', 'PBP', 'MCD')
    title('Test Log-Likelihood Naval')
    drawnow
    % savefig('LLtest_Naval.fig')

    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 600, 400])
    plot((1:400)*1.355,TAGI_LL,'r','LineWidth',1);hold on;
    plot((1:400)*4.843,PBP_LL,'b','LineWidth',1);hold on;
    plot((1:400)*1.424,MCD_LL,'g','LineWidth',1);
    xlabel('Epoch')
    ylabel('LL')
    %ylim([-7, -2.40])
    legend('TAGI-BNI', 'PBP', 'MCD')
    title('Test Log-Likelihood Naval')
    drawnow
    savefig('LLtest_Naval_timed.fig')
    %% RMSE
    filename1 = 'UCI_navalRMSEtest400Epoch.txt';
    filename2 = 'PBP/RMSEtest_PBP.txt';
    filename3 = 'MCD/RMSEtest_MCD.txt';
    TAGI_RMSE  = importdata(filename1);
    PBP_RMSE   = importdata(filename2);
    MCD_RMSE   = importdata(filename3);
    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 600, 400])
    plot(1:400,TAGI_RMSE,'r','LineWidth',1);hold on;
    plot(1:400,PBP_RMSE,'b','LineWidth',1);hold on;
    plot(1:400,MCD_RMSE,'g','LineWidth',1);
    xlabel('Epoch')
    ylabel('RMSE')
    %ylim([-7, -2.40])
    legend('TAGI-BNI', 'PBP', 'MCD')
    title('Test RMSE Naval')
    drawnow
    % savefig('RMSEtest_Naval.fig')
elseif Layer==2
    if metric == 1
        filename1 = 'UCI_navalLLtest_2L100nodes.txt';
        filename2 = 'PBP/lltest_PBP.txt';
        filename3 = 'MCD/lltest_MCD.txt';
        filename4 = 'UCI_navalLLtest400Epoch.txt';
        filename5 = 'UCI_TAGI_JMLRnavalLLtest100Epoch.txt';
        filename6 = 'results_DE/lltest_DE.txt';
        DE        = importdata(filename6);
        TAGI_JMLR = importdata(filename5);
        TAGI_LL   = importdata(filename4);
        TAGI_LL1  = importdata(filename1);
        PBP_LL    = importdata(filename2);
        MCD_LL    = importdata(filename3);

        blue = [-485.70194646,418.74950551;-447.75855118,588.3775748;-409.81474016,660.89287559;-371.87134866,711.55373858;-333.92756787,750.63828661;-295.98378709,782.39136378;-258.0400063,808.68560504;-220.09659969,832.5317178;-182.15319307,852.87645732;-144.20903811,870.95468598;-106.2656315,887.68281449;-68.32222488,903.02666079;-30.37806917,916.66249701;7.56533631,929.45023748;45.50911748,941.16198803;83.45252409,952.21032189;121.39630488,961.98803528;159.34008567,971.40971339;197.28349228,981.10980283;235.22689512,989.7182778;273.17105386,998.15716157;311.11445669,1005.58487055;349.05824126,1013.41688693;387.00200315,1020.32543244];
        red =  [-448.50330709,-112.88005039;-419.54539843,-112.88005039];

        % plot(blue(:,1),blue(:,2),'b');hold on;plot(red(:,1),red(:,2),'r');

        x1 = -485.73505512;
        x2 = 1297.24440945;

        y1 = -282.55732913;
        y2 = 1170.45382299;

        xmin = 0;
        xmax = 12400;

        ymin =  2;
        ymax =  6;

        xmin_PBPMV = 271.26;
        xmin_VMG   = 271.26;
        blue_new(:,1) = (blue(:,1)-x1)/(x2-x1)*(xmax-xmin_PBPMV)+xmin_PBPMV;
        blue_new(:,2) = (blue(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;

        red_new(:,1) = (red(:,1)-x1)/(x2-x1)*(xmax-xmin_VMG)+xmin_VMG;
        red_new(:,2) = (red(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        red_new(3,:) = [2000 2.4671];
        red_new(4,:) = [6000 2.4671];
        red_new(5,:) = [10000 2.4671];
        red_new(6,:) = [0 2.4671];

        FigHandle = figure;
        set(FigHandle, 'Position', [100, 100, 600, 400])
        plot((1:400)*0.493,TAGI_LL,'r','LineWidth',1);hold on;
        plot((1:100)*3.26,TAGI_LL1,'--r','LineWidth',1);hold on;
        plot((1:400)*5.375,PBP_LL,'b','LineWidth',1);hold on;
        plot((1:400)*0.631,MCD_LL,'g','LineWidth',1);
        plot(blue_new(:,1),blue_new(:,2),'c','LineWidth',1);hold on;
        plot(red_new(:,1),red_new(:,2),'m');hold on;
        plot((1:100)*1.93,TAGI_JMLR,':r','LineWidth',2);
        plot((1:400)*0.88,DE,'k','LineWidth',2); %'color',[.5 .4 .7]
        xlabel('time(s)')
        ylabel('LL')
    %     ylim([0, 1.5])
        legend('TAGI-BNI','TAGI-BNI 2L','PBP','MCD','PBP-MV','VMG','TAGI-JMLR','DE') %'
        title('Test Log-Likelihood Naval')
        fig = gcf;
        ax = fig.CurrentAxes;
        set(ax,'xscale','log')
        drawnow
    elseif metric == 2
        filename1 = 'UCI_navalRMSEtest_2L100nodes.txt';
        filename2 = 'PBP/RMSEtest_PBP.txt';
        filename3 = 'MCD/RMSEtest_MCD.txt';
        filename4 = 'UCI_navalRMSEtest400Epoch.txt';
        filename5 = 'UCI_TAGI_JMLRnavalRMSEtest100Epoch.txt';
        filename6 = 'results_DE/RMSEtest_DE.txt';
        DE        = importdata(filename6);
        TAGI_JMLR = importdata(filename5);
        TAGI_LL   = importdata(filename4);
        TAGI_LL1  = importdata(filename1);
        PBP_LL    = importdata(filename2);
        MCD_LL    = importdata(filename3);
        
        blue = [-415.72977638,830.47337575;-385.51782047,308.26594016;-355.30588346,283.81946457;-325.09393512,273.85421102;-294.88198677,267.45112441;-264.66974362,264.75295748;-234.45779528,266.01528189;-204.24614551,257.63633386;-174.03420094,255.83882835;-143.82195402,253.87264252;-113.61000945,249.83013543;-83.39815181,245.02775433;-53.18593512,244.3760126;-22.9740771,242.2743685;7.23781039,241.53112441;37.44969789,239.85630236;67.66191496,242.33824252;97.87379906,242.97006614;128.08565669,238.90609134;158.29754457,238.33485354;188.50976126,236.40574488;218.72164913,237.68519055;248.93353701,235.06389921;279.14574992,235.61775118];
        
        red = [-409.69549606,889.20945638;-373.4489537,813.78247559;-337.20271748,815.42229921;-300.95647748,828.54001512;-264.71024126,810.50313827;-228.46400504,779.34854551;-192.21776882,808.86344315;-155.97123402,815.42229921;-119.72469921,805.584;-83.47852346,812.14285984;-47.23225701,800.66486173;-10.9859588,785.90740157;25.26033751,821.98115906;61.50660283,800.66486173;97.75289953,772.78968567;133.99919622,790.82656252;170.24546268,769.51024252;206.49175937,785.90740157;242.73805606,785.90740157;278.98435276,825.26059843;315.23061921,805.584;351.47691591,774.42940724;387.7232126,790.82656252;423.96948661,792.46613291;460.21576063,784.2675515;496.46207244,762.95125795;532.70834646,795.74557228;568.95462047,782.62785638;605.20093228,774.42927874;641.44750866,782.62785638;677.69348031,789.18671622;713.94013228,749.83351559;750.1863685,759.67181858;786.43268031,767.87039622;822.67899213,764.59097953;858.92526614,777.70869543;895.17154016,761.31153638;931.41785197,741.63492283;967.66412598,746.55410268;1003.9104,746.55410268];
        
        %plot(blue(:,1),blue(:,2),'b');hold on;plot(red(:,1),red(:,2),'r');
        
        x1 = -415.7559685;
        x2 = 1003.9104;
        
        y1 = 205.44782362;
        y2 = 1025.30582551;
        
        xmin = 0;
        xmax = 12500;
        
        ymin = 0;
        ymax =5e-03;
        
        xmin_PBPMV = 271.26;
        xmin_VMG   = 271.26;
        blue_new(:,1) = (blue(:,1)-x1)/(x2-x1)*(xmax-xmin_PBPMV)+xmin_PBPMV;
        blue_new(:,2) = (blue(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        
        red_new(:,1) = (red(:,1)-x1)/(x2-x1)*(xmax-xmin_VMG)+xmin_VMG;
        red_new(:,2) = (red(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        FigHandle = figure;
        set(FigHandle, 'Position', [100, 100, 600, 400])
        plot((1:400)*0.493,TAGI_LL,'r','LineWidth',1);hold on;
        plot((1:100)*3.26,TAGI_LL1,'--r','LineWidth',1);hold on;
        plot((1:400)*5.375,PBP_LL,'b','LineWidth',1);hold on;
        plot((1:400)*0.631,MCD_LL,'g','LineWidth',1);
        plot(blue_new(:,1),blue_new(:,2),'c','LineWidth',1);hold on;
        plot(red_new(:,1),red_new(:,2),'m');hold on;
        plot((1:100)*1.93,TAGI_JMLR,':r','LineWidth',2);
        plot((1:400)*0.88,DE,'k','LineWidth',2);
        xlabel('time(s)')
        ylabel('LL')
    %     ylim([0, 1.5])
        legend('TAGI-BNI','TAGI-BNI 2L','PBP','MCD','PBP-MV','VMG','TAGI-JMLR','DE') %'
        title('Test RMSE Naval')
        fig = gcf;
        ax = fig.CurrentAxes;
        set(ax,'xscale','log')
        drawnow
    end   
end