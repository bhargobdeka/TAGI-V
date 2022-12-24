clear;clc
Layer = 2;metric=2;
if Layer==1
    %% LL
    filename1 = 'UCI_bostonHousingLLtest400Epoch.txt';
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
    title('Test Log-Likelihood Boston')
    drawnow
    % savefig('LLtest_Boston.fig')
    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 600, 400])
    plot((1:400)*0.027,TAGI_LL,'r','LineWidth',1);hold on;
    plot((1:400)*0.189,PBP_LL,'b','LineWidth',1);hold on;
    plot((1:400)*0.084,MCD_LL,'g','LineWidth',1);
    xlabel('Epoch x time per epoch')
    ylabel('LL')
    % ylim([-7, -2.40])
    legend('TAGI-BNI', 'PBP', 'MCD')
    title('Test Log-Likelihood Boston')
    drawnow
    savefig('LLtest_Boston_timed.fig')
    %% RMSE
    filename1 = 'UCI_bostonHousingRMSEtest400Epoch.txt';
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
    ylim([2, 9])
    legend('TAGI-BNI', 'PBP', 'MCD')
    title('Test RMSE Boston')
    drawnow
    % savefig('RMSEtest_Boston.fig')
elseif Layer==2
    if metric==1
        filename1 = 'UCI_bostonHousingLLtest_2L100nodes.txt';
        filename2 = 'PBP/lltest_PBP.txt';
        filename3 = 'MCD/lltest_MCD.txt';
        filename4 = 'UCI_bostonHousingLLtest400Epoch.txt';
        filename5 = 'UCI_TAGI_JMLR_BostonHousingLLtest100Epoch.txt';
        filename6 = 'results_DE/lltest_DE.txt';
        DE        = importdata(filename6);
        TAGI_JMLR = importdata(filename5);
        TAGI_LL   = importdata(filename4);
        TAGI_LL1  = importdata(filename1);
        PBP_LL    = importdata(filename2);
        MCD_LL    = importdata(filename3);
        blue = [-419.70859843,678.70091339;-370.86419528,771.50369008;-322.02006047,792.89485228;-273.17592567,777.41962205;-224.33154142,799.57865953;-175.48741039,801.49709858;-126.64327559,791.35954016;-77.79914079,798.53358614;-28.95500485,796.72731969;19.88962885,790.77089386;68.73368693,779.81804598;117.57784819,788.08896378;166.42200567,783.76675654;215.26641638,769.99799055;264.11057764,765.15192567;312.95473512,771.42964157;361.79889638,766.73120504;410.64306142,773.27254299;459.48748346,764.66066268;508.33164094,766.3836548;557.17579843,763.6526589;606.01995591,754.94566299;654.86437795,766.18264441;703.70853543,757.38516661];

        red = [-447.43048819,-111.93029291;-426.30825827,369.21630236;-405.18602835,420.27462047;-384.06353386,420.84264567;-362.94154961,355.50379843;-341.81906646,367.00614803;-320.69658331,335.65787717;-299.57459906,353.86170709;-278.45211591,270.62199685;-257.3298822,268.54106457;-236.2076485,217.5031937;-215.08516535,225.90455433;-193.96318488,225.46953071;-172.84070173,243.41166614;-151.71846803,179.38208504;-130.59623433,202.17565984;-109.47400063,215.6367874;-88.35151748,157.99234016;-67.22928378,66.79619528;-46.10705008,115.68230551;-24.98456844,17.61815433;-3.86258494,153.21169134;17.2598982,108.0560126;38.38203213,62.17560945;59.50441323,108.9535748;80.62679811,37.67145827;101.74892976,38.65432441;122.87131465,-13.70286614;143.99369575,47.12039055;165.11580472,-33.37232126;186.23821228,-23.81136378;207.36032126,-60.27651024;228.48270614,-82.6999937;249.6048378,-104.78660787;270.72722268,-0.06893858;291.84960378,-41.50491969;312.97198866,4.81371969;334.09412031,-49.10736378;355.2165052,64.73272441;376.33861039,-78.72529134];

        % plot(blue(:,1),blue(:,2),'b');hold on;plot(red(:,1),red(:,2),'r');
        x1 = -447.43048819;
        x2 = 703.73416063;

        y1 = -116.05799055;
        y2 = 903.95448945;

        xmin = 0;
        xmax = 900;

        ymin = -5;
        ymax =  -2;

        xmin_PBPMV    = 37.2;
        xmin_VMG      = 18.6;
        blue_new(:,1) = (blue(:,1)-x1)/(x2-x1)*(xmax-xmin_PBPMV)+xmin_PBPMV;
        blue_new(:,2) = (blue(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;

        red_new(:,1) = (red(:,1)-x1)/(x2-x1)*(xmax-xmin_VMG)+xmin_VMG;
        red_new(:,2) = (red(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        FigHandle = figure;
        set(FigHandle, 'Position', [100, 100, 600, 400])
        plot((1:400)*0.025,TAGI_LL,'r','LineWidth',1);hold on;
        plot((1:100)*0.1239,TAGI_LL1,'--r','LineWidth',1);hold on;
        plot((1:400)*0.25,PBP_LL,'b','LineWidth',1);hold on;
        plot((1:400)*0.041,MCD_LL,'g','LineWidth',1);hold on;
        plot(blue_new(:,1),blue_new(:,2),'c','LineWidth',1);hold on;
        plot(red_new(:,1),red_new(:,2),'m');hold on;
        plot((1:100)*0.099,TAGI_JMLR,':r','LineWidth',2);
        plot((1:400)*0.061,DE,'k','LineWidth',2);
        xlabel('time(s)')
        ylabel('LL')
        ylim([-7, -2])
        legend('TAGI-BNI','TAGI-BNI 2L','PBP','MCD','PBP-MV','VMG','TAGI','DE') %'
        title('Test Log-Likelihood Boston')
        fig = gcf;
        ax = fig.CurrentAxes;
        set(ax,'xscale','log')
        drawnow
    elseif metric == 2
        filename1 = 'UCI_bostonHousingRMSEtest_2L100nodes.txt';
        filename2 = 'PBP/RMSEtest_PBP.txt';
        filename3 = 'MCD/RMSEtest_MCD.txt';
        filename4 = 'UCI_bostonHousingRMSEtest400Epoch.txt';
        filename5 = 'UCI_TAGI_JMLR_BostonHousingRMSEtest100Epoch.txt';
        filename6 = 'results_DE/RMSEtest_DE.txt';
        DE        = importdata(filename6);
        
        TAGI_JMLR = importdata(filename5);
        TAGI_LL   = importdata(filename4);
        TAGI_LL1  = importdata(filename1);
        PBP_LL    = importdata(filename2);
        MCD_LL    = importdata(filename3);
        blue = [-405.97307717,522.99333543;
            -358.86986079,301.30533543;
            -311.76710173,274.58728819;
            -264.66411213,342.65846929;
            -217.56088441,289.00440945;
            -170.45789102,291.16705512;
            -123.35490142,314.96201575;
            -76.25186268,301.43380157;
            -29.14889235,303.81702047;
            17.95400542,311.78142992;
            65.05723087,326.36507717;
            112.16010709,311.82583937;
            159.2633348,314.7656315;
            206.3662337,329.99315906;
            253.46920063,334.23647244;
            300.57242835,322.79822362;
            347.67532346,325.58297953;
            394.77830551,314.58308031;
            441.88146142,321.90602835;
            488.98442835,317.15512441;
            536.08731969,317.27531339;
            583.19051339,323.63629606;
            630.29340472,308.67080315;
            677.39640945,315.02052283];
        
        red = [-432.70680945,975.76312063;
            -412.33727244,880.17899717;
            -391.968,807.61579465;
            -371.59872,790.09348157;
            -351.22944,807.23603906;
            -330.86016,772.50645543;
            -310.49088,782.67323339;
            -290.1216,755.05735181;
            -269.75231622,807.34591748;
            -249.38303622,791.08298835;
            -229.01352189,814.17304063;
            -208.64424189,829.05460157;
            -188.27496189,817.03720819;
            -167.90567811,801.52180535;
            -147.53639811,820.9576063;
            -127.16711811,815.49738331;
            -106.79760378,808.8001285;
            -86.42832378,824.42722394;
            -66.05913449,884.21531339;
            -45.68978268,852.12031748;
            -25.32022148,906.52736504;
            -4.95089386,831.5673789;
            15.41843376,857.02189606;
            35.78778482,888.71871874;
            56.15701795,836.70843213;
            76.52634709,874.12356283;
            96.89567244,875.28348472;
            117.26500157,910.77889134;
            137.63433071,856.52415874;
            158.00358425,912.23337071;
            178.37291339,928.75414677;
            198.74224252,909.5540863;
            219.11156787,905.39801953;
            239.48089701,904.05734551;
            259.85015433,898.59095811;
            280.21947969,876.47385827;
            300.58880882,877.87347402;
            320.95813795,915.67651654;
            341.32746331,845.95920756;
            361.69672063,901.91430425];
        
        x1 = -432.70680945;
        x2 = 677.42120315;
        y2 = 1021.64257512;
        y1 = 209.11109291;
        
        xmin = 0;
        xmax = 900;
        
        ymin=2;ymax =6;
        
        xmin_PBPMV    = 37.2;
        xmin_VMG      = 18.6;
        blue_new(:,1) = (blue(:,1)-x1)/(x2-x1)*(xmax-xmin_PBPMV)+xmin_PBPMV;
        blue_new(:,2) = (blue(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        
        red_new(:,1) = (red(:,1)-x1)/(x2-x1)*(xmax-xmin_VMG)+xmin_VMG;
        red_new(:,2) = (red(:,2)-y1)/(y2-y1)*(ymax-ymin)+ymin;
        
        FigHandle = figure;
        set(FigHandle, 'Position', [100, 100, 600, 400])
        plot((1:400)*0.025,TAGI_LL,'r','LineWidth',1);hold on;
        plot((1:100)*0.1239,TAGI_LL1,'--r','LineWidth',1);hold on;
        plot((1:400)*0.25,PBP_LL,'b','LineWidth',1);hold on;
        plot((1:400)*0.041,MCD_LL,'g','LineWidth',1);hold on;
        plot(blue_new(:,1),blue_new(:,2),'c','LineWidth',1);hold on;
        plot(red_new(:,1),red_new(:,2),'m');hold on;
        plot((1:100)*0.099,TAGI_JMLR,':r','LineWidth',2);
        plot((1:400)*0.061,DE,'k','LineWidth',2);
        xlabel('time(s)')
        ylabel('LL')
%         ylim([-7, -2])
        legend('TAGI-BNI','TAGI-BNI 2L','PBP','MCD','PBP-MV','VMG','TAGI','DE')
        title('Test RMSE Boston')
        fig = gcf;
        ax = fig.CurrentAxes;
        set(ax,'xscale','log')
        drawnow
    end
end