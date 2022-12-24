clear;clc
%% LL
Layer = 2;metric=2;
if Layer==1
    filename1 = 'UCI_kin8nmLLtest400Epoch.txt';
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
    title('Test Log-Likelihood Kin8nm')
    drawnow
    % savefig('LLtest_Kin8nm.fig')
    %%
    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 600, 400])
    plot((1:400)*0.373,TAGI_LL,'r','LineWidth',1);hold on;
    plot((1:400)*2.863,PBP_LL,'b','LineWidth',1);hold on;
    plot((1:400)*0.7415,MCD_LL,'g','LineWidth',1);
    xlabel('Epoch x time per epoch')
    ylabel('LL')
    % ylim([-7, -2.40])
    legend('TAGI-BNI', 'PBP', 'MCD')
    title('Test Log-Likelihood kin8nm')
    drawnow
    savefig('LLtest_kin8nm_timed.fig')
    %% RMSE
    filename1 = 'UCI_kin8nmRMSEtest400Epoch.txt';
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
    title('Test RMSE Kin8nm')
    drawnow
    % savefig('RMSEtest_Kin8nm.fig')
elseif Layer==2
    if metric == 1
        filename1 = 'UCI_kin8nmLLtest_2L100nodes.txt';
        filename2 = 'PBP/lltest_PBP.txt';
        filename3 = 'MCD/lltest_MCD.txt';
        filename4 = 'UCI_kin8nmLLtest400Epoch.txt';
        filename5 = 'UCI_TAGI_JMLRkin8nmLLtest100Epoch.txt';
        filename6 = 'results_DE/lltest_DE.txt';
        DE        = importdata(filename6);
        TAGI_JMLR = importdata(filename5);
        TAGI_LL   = importdata(filename4);
        TAGI_LL1  = importdata(filename1);
        PBP_LL    = importdata(filename2);
        MCD_LL    = importdata(filename3);

        blue = [-524.82153071,220.26394961;-502.17891024,541.5071622;-479.53628976,679.81867087;-456.89340472,737.70886299;-434.25104882,800.3712378;-411.60816378,835.72568693;-388.96580787,856.5063685;-366.32292661,877.14209008;-343.68002646,888.26000504;-321.03768945,900.77567244;-298.39478929,911.31439748;-275.75245228,914.08089827;-253.10955213,921.10684346;-230.46693543,925.11675591;-207.82431496,928.99081701;-185.18169827,931.53117354;-162.53879811,935.4060548;-139.8964611,938.22498142;-117.25356094,939.35830299;-94.61094425,941.74392945;-71.96832756,945.75143811;-49.32570709,944.65441134;-26.68308926,948.08647181;-4.04047106,948.73841008;18.60242759,949.51424504;41.24476724,951.33089008;63.88766362,952.29829417;86.53000063,951.98882646;109.17290079,953.63327622;131.81551748,954.58921701];
        red  = [-515.49652913,397.83223937;-483.52917165,517.41233386;-451.56154961,596.03327244;-419.59388976,605.36462362;-387.62653228,638.24855433;-355.65891024,661.39744252;-323.69127685,710.49909921;-291.72364346,735.28542992;-259.75628976,705.18292913;-227.78865638,724.96713071;-195.82102299,738.96411969;-163.85366929,737.82017008;-131.88603591,730.01412283;-99.91840252,733.37880945;-67.9510526,746.05234016;-35.98341732,740.44459843;-4.01578394,767.40675024;27.95156901,765.14119937;59.91892157,730.12629921;91.88683465,780.43919622;123.85390866,796.97097071;155.82182173,756.52767496;187.78917543,768.37128945;219.75680882,819.78333732;251.7244422,833.53354205;283.69179591,792.23797795;315.65942929,787.75173165;347.62678299,806.10023433;379.5944315,778.95875528;411.56205354,775.05573921;443.52941102,744.28028976;475.49703307,737.10232441;507.46465512,813.52512;539.43231496,803.29656189;571.39967244,804.28361575;603.36725669,715.59099213;635.33480315,813.9289474;667.30234961,817.42813606;699.26989606,780.17007118;731.23744252,817.2711685];

        % plot(blue(:,1),blue(:,2),'b');hold on;plot(red(:,1),red(:,2),'r');
        x1 = -524.89054488;
        x2 = 731.23744252;

        y1 = 95.32667717;
        y2 = 992.56982929;

        xmin = 0;
        xmax = 9.1852e+03;

        ymin =  0.9;
        ymax =  1.3;

        xmin_PBPMV = 158.73;
        xmin_VMG   = 222.22;
        blue_new(:,1) = (blue(:,1)-x1)/(x2-x1)*(xmax-xmin_PBPMV)+xmin_PBPMV;
        blue_new(:,2) = (blue(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;

        red_new(:,1) = (red(:,1)-x1)/(x2-x1)*(xmax-xmin_VMG)+xmin_VMG;
        red_new(:,2) = (red(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin; 
        FigHandle = figure;
        set(FigHandle, 'Position', [100, 100, 600, 400])
        plot((1:400)*0.309,TAGI_LL,'r','LineWidth',1);hold on;
        plot((1:100)*1.83,TAGI_LL1,'--r','LineWidth',1);hold on;
        plot((1:400)*3.65,PBP_LL,'b','LineWidth',1);hold on;
        plot((1:400)*0.434,MCD_LL,'g','LineWidth',1);
        plot(blue_new(:,1),blue_new(:,2),'c','LineWidth',1);hold on;
        plot(red_new(:,1),red_new(:,2),'m');hold on;
        plot((1:100)*0.814,TAGI_JMLR,':r','LineWidth',2);
        plot((1:400)*0.64,DE,'k','LineWidth',2); %'color',[.5 .4 .7]
        xlabel('time(s)')
        ylabel('LL')
        ylim([0, 1.5])
        legend('TAGI-BNI','TAGI-BNI 2L','PBP','MCD','PBP-MV','VMG','TAGI','DE') %'
        title('Test Log-Likelihood Kin8nm')
        fig = gcf;
        ax = fig.CurrentAxes;
        set(ax,'xscale','log')
        drawnow
    elseif metric == 2
        filename1 = 'UCI_kin8nmRMSEtest_2L100nodes.txt';
        filename2 = 'PBP/RMSEtest_PBP.txt';
        filename3 = 'MCD/RMSEtest_MCD.txt';
        filename4 = 'UCI_kin8nmRMSEtest400Epoch.txt';
        filename5 = 'UCI_TAGI_JMLRkin8nmRMSEtest100Epoch.txt';
        filename6 = 'results_DE/RMSEtest_DE.txt';
        DE        = importdata(filename6);
        TAGI_JMLR = importdata(filename5);
        TAGI_LL   = importdata(filename4);
        TAGI_LL1  = importdata(filename1);
        PBP_LL    = importdata(filename2);
        MCD_LL    = importdata(filename3);
        
        blue = [-307.79801197,282.60548031;-276.81338457,-122.86949291;-245.82833008,-248.00360315;-214.8436989,-276.6688252;-183.85948724,-337.98890079;-152.87443276,-364.72433386;-121.88980157,-376.14334488;-90.90517417,-391.81878425;-59.92053921,-397.18125354;-28.93548813,-406.17154016;2.04914551,-413.91632126;33.0333566,-413.32119685;64.01798929,-418.53909921;95.00304,-420.3984378;125.98767118,-422.31537638;156.97230236,-424.0375937;187.95735685,-427.09810394;218.94199181,-429.71429291;249.92619969,-429.82427717;280.91083087,-432.24854173;311.89588157,-436.11719055;342.88051654,-434.52933543;373.86514772,-438.14543622;404.85021732,-438.71312126;435.83482205,-439.50020787;466.81946457,-441.50486929;497.80410709,-442.72248189;528.78874961,-442.15755591;559.77335433,-444.43373858;590.75799685,-445.56355276];

        red = [-295.0379263,121.77906142;-251.29236661,-53.08943622;-207.54722646,-134.79031181;-163.80250961,-121.89017953;-120.05737323,-170.98227402;-76.3126526,-197.14102677;-32.56709443,-282.06693543;11.17804611,-286.00856693;54.92276409,-253.40005039;98.66790425,-256.9832315;142.41346394,-262.71673701;186.15818079,-263.07579213;229.90332094,-259.85079685;273.6480378,-270.95901732;317.39317795,-287.44263307;361.13831811,-263.79239055;404.88343937,-313.24293543;448.62859843,-299.26779213;492.37334173,-248.74223622;536.11887874,-312.88448504;579.86358425,-355.88496378;623.60874331,-295.68434646;667.35386457,-300.34276535;711.09857008,-359.46840945;754.84414488,-389.92690394;798.58885039,-327.9343748;842.33400945,-331.15971024;886.0791685,-340.83480945;929.82428976,-301.05940157;973.56941102,-302.1343748;1017.31415433,-251.60885669;1061.05927559,-241.57557165;1104.80439685,-348.00170079;1148.54955591,-337.96818898;1192.29467717,-320.4095622;1236.03942047,-225.4503685;1279.78457953,-351.94337008;1323.53011654,-358.03498583;1367.27482205,-289.23424252;1411.0199811,-358.03498583];
        
        x1 = -307.89231118;
        x2 = 1411.0199811;
        
        y1 = -532.54469291;
        y2 = 363.29828031;
        
        xmin = 0;
        xmax = 9000;
        
        ymin=0.065;ymax =0.090;
        xmin_PBPMV = 158.73;
        xmin_VMG   = 222.22;
        blue_new(:,1) = (blue(:,1)-x1)/(x2-x1)*(xmax-xmin_PBPMV)+xmin_PBPMV;
        blue_new(:,2) = (blue(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        
        red_new(:,1) = (red(:,1)-x1)/(x2-x1)*(xmax-xmin_VMG)+xmin_VMG;
        red_new(:,2) = (red(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        FigHandle = figure;
        set(FigHandle, 'Position', [100, 100, 600, 400])
        plot((1:400)*0.309,TAGI_LL,'r','LineWidth',1);hold on;
        plot((1:100)*1.83,TAGI_LL1,'--r','LineWidth',1);hold on;
        plot((1:400)*3.65,PBP_LL,'b','LineWidth',1);hold on;
        plot((1:400)*0.434,MCD_LL,'g','LineWidth',1);
        plot(blue_new(:,1),blue_new(:,2),'c','LineWidth',1);hold on;
        plot(red_new(:,1),red_new(:,2),'m');hold on;
        plot((1:100)*0.814,TAGI_JMLR,':r','LineWidth',2);
        plot((1:400)*0.64,DE,'k','LineWidth',2);
        xlabel('time(s)')
        ylabel('LL')
%         ylim([0, 1.5])
        legend('TAGI-BNI','TAGI-BNI 2L','PBP','MCD','PBP-MV','VMG','TAGI','DE') %'
        title('Test RMSE Kin8nm')
        fig = gcf;
        ax = fig.CurrentAxes;
        set(ax,'xscale','log')
        drawnow
    end
end