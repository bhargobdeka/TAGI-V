clear;clc;
%% LL
Layer = 2;metric=2;
if Layer==1
    filename1 = 'UCI_concreteLLtest400Epoch.txt';
    filename2 = 'PBP/lltest_PBP.txt';
    filename3 = 'MCD/lltest_MCD.txt';
    TAGI_LL    = importdata(filename1);
    PBP_LL    = importdata(filename2);
    MCD_LL    = importdata(filename3);

    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 600, 400])
    plot(1:400,TAGI_LL,'r','LineWidth',1);hold on;
    plot(1:400,PBP_LL,'b','LineWidth',1');hold on;
    plot(1:400,MCD_LL,'g','LineWidth',1');
    xlabel('Epoch')
    ylabel('LL')
    ylim([-7, -2.40])
    legend('TAGI-BNI', 'PBP', 'MCD')
    title('Test Log-Likelihood Concrete')
    drawnow
    % savefig('LLtest_Concrete.fig')
    %%
    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 600, 400])
    plot((1:400)*0.046,TAGI_LL,'r','LineWidth',1);hold on;
    plot((1:400)*0.348,PBP_LL,'b','LineWidth',1);hold on;
    plot((1:400)*0.1023,MCD_LL,'g','LineWidth',1);
    xlabel('Epoch x time per epoch')
    ylabel('LL')
    % ylim([-7, -2.40])
    legend('TAGI-BNI', 'PBP', 'MCD')
    title('Test Log-Likelihood concrete')
    drawnow
    savefig('LLtest_concrete_timed.fig')
    %% RMSE
    filename1 = 'UCI_concreteRMSEtest400Epoch.txt';
    filename2 = 'PBP/RMSEtest_PBP.txt';
    filename3 = 'MCD/RMSEtest_MCD.txt';
    TAGI_RMSE  = importdata(filename1);
    PBP_RMSE   = importdata(filename2);
    MCD_RMSE   = importdata(filename3);
    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 600, 400])
    plot(1:400,TAGI_RMSE,'r','LineWidth',1');hold on;
    plot(1:400,PBP_RMSE,'b','LineWidth',1');hold on;
    plot(1:400,MCD_RMSE,'g','LineWidth',1');
    xlabel('Epoch')
    ylabel('RMSE')
    %ylim([-7, -2.40])
    legend('TAGI-BNI', 'PBP', 'MCD')
    title('Test RMSE Concrete')
    drawnow
    % savefig('RMSEtest_Concrete.fig')
elseif Layer==2
    if metric == 1
        filename1 = 'UCI_concreteLLtest_2L100nodes.txt';
        filename2 = 'PBP/lltest_PBP.txt';
        filename3 = 'MCD/lltest_MCD.txt';
        filename4 = 'UCI_concreteLLtest400Epoch.txt';
        filename5 = 'UCI_TAGI_JMLR_ConcreteLLtest100Epoch.txt';
        filename6 = 'results_DE/lltest_DE.txt';
        DE        = importdata(filename6);
        TAGI_JMLR = importdata(filename5);
        TAGI_LL   = importdata(filename4);
        TAGI_LL1  = importdata(filename1);
        PBP_LL    = importdata(filename2);
        MCD_LL    = importdata(filename3);

        blue = [-524.92531654,292.27729134;-506.29538268,451.21492913;-487.66597795,576.04183937;-469.03630866,643.31138268;-450.40663937,459.2311937;-431.77697008,690.31615748;-413.14730079,748.53725858;-394.51736693,749.85907654;-375.88795087,788.67712252;-357.25828157,834.88823433;-338.62861228,815.05579087;-319.99894299,847.26735496;-301.3692737,846.24185575;-282.73935874,856.06003276;-264.10993512,870.85357228;-245.48026583,876.66217323;-226.85059654,884.03386205;-208.22092724,873.04244787;-189.59125795,894.83699528;-170.96158866,887.40869669;-152.33191937,897.39616252;-133.70225008,878.95444535;-115.07258079,896.4272126;-96.4429115,896.56890709];
        red = [-516.37345512,210.72170079;-489.19143307,437.66747717;-462.00967559,513.23395276;-434.82818268,543.94824567;-407.6464252,529.23148346;-380.46489449,657.44575748;-353.28312945,623.6280189;-326.10161386,636.47285669;-298.91960315,614.87187402;-271.73808756,674.46546142;-244.55632252,678.1600252;-217.37480693,551.87928189;-190.19304567,647.79057638;-163.01128063,684.5023748;-135.82951937,757.24869921;-108.64775433,520.86942992;-81.46623874,621.52210394;-54.28447748,676.68222992;-27.10271357,705.43834961;0.0788022,759.76099276;27.26056517,628.65267402;54.44232945,685.52458583;81.62409071,700.56151181;108.8056063,655.53686929;135.98737134,549.57630236;163.16888693,758.59104378;190.35064819,743.65262362;217.53216378,659.61324094;244.71417449,704.95804724;271.89593953,751.07873386;299.07745512,726.64524094;326.25921638,737.02699843;353.44073197,728.30777953;380.62250079,782.03931969;407.80425827,612.1255937;434.98601575,743.48020157;462.16747087,663.98517165;489.34930394,671.84228031;516.53083465,698.07382677;543.71266772,684.82261417];

        %plot(blue(:,1),blue(:,2),'b');hold on;plot(red(:,1),red(:,2),'r');

        x1 = -524.86745197;
        x2 = 543.71266772;

        y1 = 121.33749921;
        y2 = 1106.559024;

        xmin = 0;
        xmax = 1288.89;

        ymin = -3.6;
        ymax =  -2.8;

        xmin_PBPMV = 28.57;
        xmin_VMG   = 35.71;
        blue_new(:,1) = (blue(:,1)-x1)/(x2-x1)*(xmax-xmin_PBPMV)+xmin_PBPMV;
        blue_new(:,2) = (blue(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;

        red_new(:,1) = (red(:,1)-x1)/(x2-x1)*(xmax-xmin_VMG)+xmin_VMG;
        red_new(:,2) = (red(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;

        FigHandle = figure;
        set(FigHandle, 'Position', [100, 100, 600, 400])
        plot((1:400)*0.038,TAGI_LL,'r','LineWidth',1);hold on;
        plot((1:100)*0.2044,TAGI_LL1,'--r','LineWidth',1);hold on;
        plot((1:400)*0.55,PBP_LL,'b','LineWidth',1);hold on;
        plot((1:400)*0.066,MCD_LL,'g','LineWidth',1);
        plot(blue_new(:,1),blue_new(:,2),'c','LineWidth',1);hold on;
        plot(red_new(:,1),red_new(:,2),'m');hold on;
        plot((1:100)*0.134,TAGI_JMLR,':r','LineWidth',2);
        plot((1:400)*0.129,DE,'k','LineWidth',2); %'color',[.5 .4 .7]
        
        xlabel('time(s)')
        ylabel('LL')
        ylim([-4, -2.5])
        legend('TAGI-BNI','TAGI-BNI 2L','PBP','MCD','PBP-MV','VMG','TAGI','DE') %'
        title('Test Log-Likelihood Concrete')
        fig = gcf;
        ax = fig.CurrentAxes;
        set(ax,'xscale','log')
        drawnow
    elseif metric == 2
        filename1 = 'UCI_concreteRMSEtest_2L100nodes.txt';
        filename2 = 'PBP/RMSEtest_PBP.txt';
        filename3 = 'MCD/RMSEtest_MCD.txt';
        filename4 = 'UCI_concreteRMSEtest400Epoch.txt';
        filename5 = 'UCI_TAGI_JMLR_ConcreteRMSEtest100Epoch.txt';
        filename6 = 'results_DE/RMSEtest_DE.txt';
        DE        = importdata(filename6);
        TAGI_JMLR = importdata(filename5);
        TAGI_LL   = importdata(filename4);
        TAGI_LL1  = importdata(filename1);
        PBP_LL    = importdata(filename2);
        MCD_LL    = importdata(filename3);
        
        blue = [-637.20340157,996.99194457;-606.3807874,925.27834583;-575.55775748,769.18988598;-544.73472756,703.59749291;-513.91211339,968.94988724;-483.08946142,664.36244409;-452.26646929,595.01253543;-421.44343937,609.80821417;-390.6208252,566.51247874;-359.79779528,516.94782992;-328.97515465,552.43823622;-298.1525178,515.3432315;-267.32950677,519.06844724;-236.50649197,510.17155276;-205.68348094,497.08932283;-174.86121827,493.42359685;-144.03820346,485.41765039;-113.21519244,498.91608189;-82.39217764,476.4839811;-51.56954079,483.79154646;-20.7469028,473.57049449;10.07611011,491.31972283;40.89874772,474.36037795;71.72176252,473.86885039];

        red  = [-623.05394646,857.67925039;-578.08187717,837.32474835;-533.10977008,815.73877795;-488.13732283,795.16392189;-443.16521575,793.24843087;-398.19310866,697.06586457;-353.22102803,720.88100787;-308.24893606,733.50519685;-263.27646992,717.21138898;-218.30475213,675.52531654;-173.33228598,681.58971969;-128.36019402,760.81505764;-83.38772787,675.02335748;-38.41563591,640.23552756;6.5564572,608.6655874;51.52854803,757.4611578;96.50064,716.20471181;141.47310992,667.68857953;186.44482394,638.08380472;231.41729386,609.38819528;276.38938583,687.06126614;321.36185197,642.40338898;366.33356976,632.97373228;411.30602835,644.48904567;456.27813543,738.90856063;501.25058268,572.50129134;546.22268976,571.33530709;591.1944189,656.14340787;636.16686614,632.80093228;681.13897323,581.76056693;726.11142047,609.07460787;771.08314961,586.01408504;816.05559685,592.34664567;861.02770394,578.57707087;906.00018898,657.13421102;950.97225827,637.10914016;995.94436535,642.21713386;1040.91643465,642.18689764;1085.88891969,622.11643465;1130.86098898,607.43108031];

        %plot(blue(:,1),blue(:,2),'b');hold on;plot(red(:,1),red(:,2),'r');

        x1 = -637.10781732;
        x2 = 1130.86098898;

        y1 = 309.50657008;
        y2 = 1218.38994898;

        xmin = 0;
        xmax = 1288.89;

        ymin = 4;
        ymax = 8;

        xmin_PBPMV = 28.57;
        xmin_VMG   = 35.71;
        blue_new(:,1) = (blue(:,1)-x1)/(x2-x1)*(xmax-xmin_PBPMV)+xmin_PBPMV;
        blue_new(:,2) = (blue(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;

        red_new(:,1) = (red(:,1)-x1)/(x2-x1)*(xmax-xmin_VMG)+xmin_VMG;
        red_new(:,2) = (red(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        
        FigHandle = figure;
        set(FigHandle, 'Position', [100, 100, 600, 400])
        plot((1:400)*0.038,TAGI_LL,'r','LineWidth',1);hold on;
        plot((1:100)*0.2044,TAGI_LL1,'--r','LineWidth',1);hold on;
        plot((1:400)*0.55,PBP_LL,'b','LineWidth',1);hold on;
        plot((1:400)*0.066,MCD_LL,'g','LineWidth',1);
        plot(blue_new(:,1),blue_new(:,2),'c','LineWidth',1);hold on;
        plot(red_new(:,1),red_new(:,2),'m');hold on;
        plot((1:100)*0.134,TAGI_JMLR,':r','LineWidth',2);
        plot((1:400)*0.129,DE,'k','LineWidth',2);
        xlabel('time(s)')
        ylabel('LL')
%         ylim([-4, -2.5])
        legend('TAGI-BNI','TAGI-BNI 2L','PBP','MCD','PBP-MV','VMG','TAGI','DE')
        title('Test RMSE Concrete')
        fig = gcf;
        ax = fig.CurrentAxes;
        set(ax,'xscale','log')
        drawnow
    end
end