clear;clc
%% LL
Layer = 2;metric=2;
if Layer==1
    %% LL
    filename1 = 'UCI_powerPlantLLtest400Epoch.txt';
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
    % ylim([-7, -2.40])
    legend('TAGI-BNI', 'PBP', 'MCD')
    title('Test Log-Likelihood powerPlant')
    drawnow
    %savefig('LLtest_Power.fig')

    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 600, 400])
    plot((1:400)*0.841,TAGI_LL,'r','LineWidth',1);hold on;
    plot((1:400)*3.64,PBP_LL,'b','LineWidth',1);hold on;
    plot((1:400)*1.450,MCD_LL,'g','LineWidth',1);
    xlabel('Epoch x time per epoch')
    ylabel('LL')
    % ylim([-7, -2.40])
    legend('TAGI-BNI', 'PBP', 'MCD')
    title('Test Log-Likelihood powerPlant')
    drawnow
    savefig('LLtest_Power_timed.fig')
    %% RMSE
    filename1 = 'UCI_powerPlantRMSEtest400Epoch.txt';
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
    title('Test RMSE powerPlant')
    drawnow
    % savefig('RMSEtest_Power.fig')
elseif Layer==2
    if metric == 1
        filename1 = 'UCI_powerPlantLLtest_2L100nodes.txt';
        filename2 = 'PBP/lltest_PBP.txt';
        filename3 = 'MCD/lltest_MCD.txt';
        filename4 = 'UCI_powerPlantLLtest400Epoch.txt';
        filename5 = 'UCI_TAGI_JMLR_powerPlantLLtest100Epoch.txt';
        filename6 = 'results_DE/lltest_DE.txt';
        DE        = importdata(filename6);
        TAGI_JMLR = importdata(filename5);
        TAGI_LL   = importdata(filename4);
        TAGI_LL1  = importdata(filename1);
        PBP_LL    = importdata(filename2);
        MCD_LL    = importdata(filename3);

        blue = [-571.59537638,121.53501732;-540.71346142,248.77984252;-509.83150866,282.32057953;-478.9495937,317.40842835;-448.06767874,345.99496063;-417.18572598,360.30485669;-386.30381102,386.10009449;-355.42188472,383.69068346;-324.53995465,356.1983622;-293.65802835,347.92422047;-262.77641197,385.00573228;-231.89448189,394.89365669;-201.01255559,377.81926299;-170.13062551,374.42188346;-139.24869921,382.38209764;-108.36677291,400.40042835;-77.48484283,382.48762205;-46.60291654,392.91692598;-15.72129865,399.43264252;15.16062917,398.84364094;46.04255622,398.99368819;76.9244863,415.4159622;107.8064126,411.39484724;138.6883389,417.58677165];

        red = [-445.87415433,789.48806551;-289.27097197,836.63626961;-132.66811465,893.84874709;23.93536403,862.8577474;180.53821984,858.62038677;337.14138709,917.88532535;493.74455433,866.30101795;650.34772913,950.53085102];

        %plot(blue(:,1),blue(:,2),'b');hold on;plot(red(:,1),red(:,2),'r');
        x1 = -571.64968819;
        x2 = 650.34772913;

        y1 = 87.31400315;
        y2 = 1080.58250835;

        xmin = 0;
        xmax = 8133.33;

        ymin =  -2.9;
        ymax =  -2.75;

        xmin_PBPMV = 181.818;
        xmin_VMG   = 981.8;
        blue_new(:,1) = (blue(:,1)-x1)/(x2-x1)*(xmax-xmin_PBPMV)+xmin_PBPMV;
        blue_new(:,2) = (blue(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;

        red_new(:,1) = (red(:,1)-x1)/(x2-x1)*(xmax-xmin_VMG)+xmin_VMG;
        red_new(:,2) = (red(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        FigHandle = figure;
        set(FigHandle, 'Position', [100, 100, 600, 400])
        plot((1:400)*0.32,TAGI_LL,'r','LineWidth',1);hold on;
        plot((1:100)*1.61,TAGI_LL1,'--r','LineWidth',1);hold on;
        plot((1:400)*4.2,PBP_LL,'b','LineWidth',1);hold on;
        plot((1:400)*0.496,MCD_LL,'g','LineWidth',1);
        plot(blue_new(:,1),blue_new(:,2),'c','LineWidth',1);hold on;
        plot(red_new(:,1),red_new(:,2),'m');hold on;
        plot((1:100)*0.783,TAGI_JMLR,':r','LineWidth',2);
        plot((1:400)*0.70,DE,'k','LineWidth',2); %'color',[.5 .4 .7]
        xlabel('time(s)')
        ylabel('LL')
        ylim([-3, -2.7])
        legend('TAGI-BNI','TAGI-BNI 2L','PBP','MCD','PBP-MV','VMG','TAGI-JMLR','DE') %'
        title('Test Log-Likelihood Power')
        fig = gcf;
        ax = fig.CurrentAxes;
        set(ax,'xscale','log')
        drawnow
    elseif metric == 2
        filename1 = 'UCI_powerPlantRMSEtest_2L100nodes.txt';
        filename2 = 'PBP/RMSEtest_PBP.txt';
        filename3 = 'MCD/RMSEtest_MCD.txt';
        filename4 = 'UCI_powerPlantRMSEtest400Epoch.txt';
        filename5 = 'UCI_TAGI_JMLR_powerPlantRMSEtest100Epoch.txt';
        filename6 = 'results_DE/RMSEtest_DE.txt';
        DE        = importdata(filename6);
        TAGI_JMLR = importdata(filename5);
        TAGI_LL   = importdata(filename4);
        TAGI_LL1  = importdata(filename1);
        PBP_LL    = importdata(filename2);
        MCD_LL    = importdata(filename3);
        
        blue = [-414.4856315,925.95792;-377.96042835,815.00249575;-341.43483969,772.00267465;-304.90961008,736.45587402;-268.38438047,710.76196535;-231.85914709,691.36255748;-195.33357732,668.22682205;-158.80834772,670.29811654;-122.28311811,699.07944567;-85.75788472,702.19778268;-49.23255307,668.34825827;-12.70711824,660.5472;23.81794243,674.68021417;60.34337764,677.91764409;96.86881134,668.67783307;133.39390866,652.76598425;169.91930835,669.14097638;206.44474205,661.23779528;242.96983559,656.16990236;279.49527307,653.9536252;316.02070677,653.12091969;352.5457663,637.86168189;389.0712189,641.62465512;425.59664882,635.16536693];
        
        red = [-265.78991244,278.23574173;-80.56895244,246.04765984;104.65218142,206.41587402;289.87320945,228.90436535;475.09421102,235.57182992;660.31525039,180.63658583;845.53625197,228.63152126;1030.75729134,163.50776693];
        
        %plot(blue(:,1),blue(:,2),'b');hold on;plot(red(:,1),red(:,2),'r');
        x1 = -414.54992126;
        x2 = 1030.75729134;
        
        y1 = 81.42788031;
        y2 = 943.61146961;
        
        xmin = 0;
        xmax = 8133.33;
        
        ymin = 3.8;
        ymax = 4.4;
        
        xmin_PBPMV = 181.818;
        xmin_VMG   = 981.8;
        blue_new(:,1) = (blue(:,1)-x1)/(x2-x1)*(xmax-xmin_PBPMV)+xmin_PBPMV;
        blue_new(:,2) = (blue(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        
        % red_new(:,1) = (red(:,1)-x1)/(x2-x1)*(xmax-xmin_VMG)+xmin_VMG;
        red_new(:,1) = [981.8;2000;3100;4100;5000;6100;7000;xmax];
        red_new(:,2) = (red(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        
        FigHandle = figure;
        set(FigHandle, 'Position', [100, 100, 600, 400])
        plot((1:400)*0.32,TAGI_LL,'r','LineWidth',1);hold on;
        plot((1:100)*1.61,TAGI_LL1,'--r','LineWidth',1);hold on;
        plot((1:400)*4.2,PBP_LL,'b','LineWidth',1);hold on;
        plot((1:400)*0.496,MCD_LL,'g','LineWidth',1);
        plot(blue_new(:,1),blue_new(:,2),'c','LineWidth',1);hold on;
        plot(red_new(:,1),red_new(:,2),'m');hold on;
        plot((1:100)*0.783,TAGI_JMLR,':r','LineWidth',2);
        plot((1:400)*0.70,DE,'k','LineWidth',2); %'color',[.5 .4 .7]
        xlabel('time(s)')
        ylabel('LL')
%         ylim([-3, -2.7])
        legend('TAGI-BNI','TAGI-BNI 2L','PBP','MCD','PBP-MV','VMG','TAGI-JMLR','DE') %'
        title('Test RMSE Power')
        fig = gcf;
        ax = fig.CurrentAxes;
        set(ax,'xscale','log')
        drawnow
    end
end