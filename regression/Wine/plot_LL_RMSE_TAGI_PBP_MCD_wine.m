clear;clc
%% LL
Layer = 2;metric=2;
if Layer==1
    %% LL
    filename1 = 'UCI_wineLLtest400Epoch.txt';
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
    title('Test Log-Likelihood Wine')
    drawnow
    % savefig('LLtest_wine.fig')
    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 600, 400])
    plot((1:400)*0.0645,TAGI_LL,'r','LineWidth',1);hold on;
    plot((1:400)*0.540,PBP_LL,'b','LineWidth',1);hold on;
    plot((1:400)*0.103,MCD_LL,'g','LineWidth',1);
    xlabel('Epoch')
    ylabel('LL')
    %ylim([-7, -2.40])
    legend('TAGI-BNI', 'PBP', 'MCD')
    title('Test Log-Likelihood Wine')
    drawnow
    savefig('LLtest_wine_timed.fig')
    %% RMSE
    filename1 = 'UCI_wineRMSEtest400Epoch.txt';
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
    title('Test RMSE Wine')
    drawnow
    % savefig('RMSEtest_wine.fig')
elseif Layer==2
    if metric == 1
        filename1 = 'UCI_wineLLtest_2L100nodes.txt';
        filename2 = 'PBP/lltest_PBP.txt';
        filename3 = 'MCD/lltest_MCD.txt';
        filename4 = 'UCI_wineLLtest400Epoch.txt';
        filename5 = 'UCI_TAGI_JMLR_WineLLtest100Epoch.txt';
        filename6 = 'results_DE/lltest_DE.txt';
        DE        = importdata(filename6);
        TAGI_JMLR = importdata(filename5);
        TAGI_LL   = importdata(filename4);
        TAGI_LL1  = importdata(filename1);
        PBP_LL    = importdata(filename2);
        MCD_LL    = importdata(filename3);

        blue=[-513.65106142,532.4871685;-476.77402205,562.24603465;-439.89773858,629.19367559;-403.02107717,633.15915591;-366.1447748,606.8400378;-329.26773921,578.53190551;-292.39107402,606.58106457;-255.51477165,599.52268346;-218.63810646,585.89261102;-181.76107465,586.28942362;-144.88477228,570.86093858;-108.00810709,606.04195276;-71.1310715,576.74006929;-34.25477367,535.06748976;2.62189289,546.43551496;39.49819087,560.74287874;76.37522646,553.82864882;113.25189165,547.98742677;150.12819402,563.58810709;187.00522583,564.97689449;223.88189102,558.13489134;260.75819339,507.79226457;297.63485858,540.11943307;334.51189417,518.75482205];

        red=[-388.24486299,271.30586457;-225.96199181,157.13113701;-63.67948346,195.93154016;98.60338772,150.33290079;260.88626268,72.76176378;423.16913386,189.25205669;585.45165354,185.71940787;747.73413543,89.47540157;910.01729764,-85.73499213;1072.29985512,-56.84995276];

        %plot(blue(:,1),blue(:,2),'b');hold on;plot(red(:,1),red(:,2),'r');

        x1 = -513.51567874;
        x2 = 1072.29985512;

        y1 = -261.21267402;
        y2 = 926.25205039;

        xmin = 0;
        xmax = 1750;

        ymin =  -1.1;
        ymax =  -0.7;

        xmin_PBPMV = 39.68;
        xmin_VMG   = 166.67;
        blue_new(:,1) = (blue(:,1)-x1)/(x2-x1)*(xmax-xmin_PBPMV)+xmin_PBPMV;
        blue_new(:,2) = (blue(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;

        red_new(:,1) = (red(:,1)-x1)/(x2-x1)*(xmax-xmin_VMG)+xmin_VMG;
        red_new(:,2) = (red(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;

        FigHandle = figure;
        set(FigHandle, 'Position', [100, 100, 600, 400])
        plot((1:400)*0.062,TAGI_LL,'r','LineWidth',1);hold on;
        plot((1:100)*0.36,TAGI_LL1,'--r','LineWidth',1);hold on;
        plot((1:400)*0.80,PBP_LL,'b','LineWidth',1);hold on;
        plot((1:400)*0.093,MCD_LL,'g','LineWidth',1);
        plot(blue_new(:,1),blue_new(:,2),'c','LineWidth',1);hold on;
        plot(red_new(:,1),red_new(:,2),'m');hold on;
        plot((1:100)*0.1575,TAGI_JMLR,':r','LineWidth',2);
        plot((1:400)*0.185,DE,'k','LineWidth',2); %'color',[.5 .4 .7]
        xlabel('time(s)')
        ylabel('LL')
        ylim([-2, -0.7])
        legend('TAGI-BNI','TAGI-BNI 2L','PBP','MCD','PBP-MV','VMG','TAGI','DE') %'
        title('Test Log-Likelihood Wine')
        fig = gcf;
        ax = fig.CurrentAxes;
        set(ax,'xscale','log')
        drawnow
    elseif metric == 2
        filename1 = 'UCI_wineRMSEtest_2L100nodes.txt';
        filename2 = 'PBP/RMSEtest_PBP.txt';
        filename3 = 'MCD/RMSEtest_MCD.txt';
        filename4 = 'UCI_wineRMSEtest400Epoch.txt';
        filename5 = 'UCI_TAGI_JMLR_WineRMSEtest100Epoch.txt';
        filename6 = 'results_DE/RMSEtest_DE.txt';
        DE        = importdata(filename6);
        TAGI_JMLR = importdata(filename5);
        TAGI_LL   = importdata(filename4);
        TAGI_LL1  = importdata(filename1);
        PBP_LL    = importdata(filename2);
        MCD_LL    = importdata(filename3);
        blue = [31.2789411,275.08811339;59.3203389,263.89844409;87.36173102,202.39793386;115.40283591,208.656;143.44421291,277.14122835;171.48530646,317.71128189;199.52670614,293.72726929;227.56810961,299.35831181;255.60920315,317.34085039;283.65060661,307.23220157;311.69170016,321.44847874;339.73307339,284.57880945;367.77416693,305.13252283;395.81555906,345.24510236;423.85697008,317.77088504;451.89807874,296.71937008;479.93945197,295.44113386;507.98052283,303.54413858;536.02193386,286.89804094;564.06334488,279.10820787;592.10441575,285.10525984;620.14582677,322.24988976;648.18693543,296.18199685;676.22830866,309.37568504];
        red  = [126.63894425,451.77652913;250.03973669,544.42155591;373.44083528,507.34820787;496.84195276,545.34073701;620.24303622,597.38895118;743.64381732,502.02466772;867.04493858,515.54418898;990.44598425,581.0736;1113.84680315,679.38689764;1237.24788661,628.06635591];
        
        % plot(blue(:,1),blue(:,2),'b');hold on;plot(red(:,1),red(:,2),'r');
        x1 = 31.38191093;
        x2 = 1237.24788661;
        
        y1 = 58.10211024;
        y2 = 824.08010835;
        
        xmin = 0;
        xmax = 1705;
        
        ymin = 0.5;
        ymax = 0.7;
        
        xmin_PBPMV = 39.68;
        xmin_VMG   = 166.67;
        blue_new(:,1) = (blue(:,1)-x1)/(x2-x1)*(xmax-xmin_PBPMV)+xmin_PBPMV;
        blue_new(:,2) = (blue(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        
        red_new(:,1) = (red(:,1)-x1)/(x2-x1)*(xmax-xmin_VMG)+xmin_VMG;
        red_new(:,2) = (red(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        FigHandle = figure;
        set(FigHandle, 'Position', [100, 100, 600, 400])
        plot((1:400)*0.062,TAGI_LL,'r','LineWidth',1);hold on;
        plot((1:100)*0.36,TAGI_LL1,'--r','LineWidth',1);hold on;
        plot((1:400)*0.80,PBP_LL,'b','LineWidth',1);hold on;
        plot((1:400)*0.093,MCD_LL,'g','LineWidth',1);
        plot(blue_new(:,1),blue_new(:,2),'c','LineWidth',1);hold on;
        plot(red_new(:,1),red_new(:,2),'m');hold on;
        plot((1:100)*0.1575,TAGI_JMLR,':r','LineWidth',2);
        plot((1:400)*0.185,DE,'k','LineWidth',2); %'color',[.5 .4 .7]
        xlabel('time(s)')
        ylabel('LL')
%         ylim([-2, -0.7])
        legend('TAGI-BNI','TAGI-BNI 2L','PBP','MCD','PBP-MV','VMG','TAGI','DE') %'
        title('Test RMSE Wine')
        fig = gcf;
        ax = fig.CurrentAxes;
        set(ax,'xscale','log')
        drawnow
    end
end