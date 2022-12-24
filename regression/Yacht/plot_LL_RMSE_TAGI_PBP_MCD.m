clear;clc
%% LL
filename1 = 'UCI_yachtLLtest400Epoch.txt';
filename2 = 'PBP/lltest_PBP.txt';
filename3 = 'MCD/lltest_MCD.txt';
filename4 = 'DVI/lltest_DVI.txt';
filename5 = 'results_DE/lltest_DE.txt';
filename6 = 'UCI_TAGI_JMLR_yachtLLtest400Epoch.txt';
filename7 = 'UCI_yachtLLtest_2L100nodes.txt';
TAGI_2L   = importdata(filename7);
TAGI_LL   = importdata(filename1);
PBP_LL    = importdata(filename2);
MCD_LL    = importdata(filename3);
DVI_LL    = importdata(filename4);
DE_LL     = importdata(filename5);
TAGI_JMLR = importdata(filename6);
FigHandle = figure;
set(FigHandle, 'Position', [100, 100, 600, 400])
plot(1:100,TAGI_LL(1:100),'r');hold on;
plot(1:100,PBP_LL(1:100),'b');hold on;
plot(1:100,MCD_LL(1:100),'g');hold on;
plot(1:100,DVI_LL(1:100),'color',[0.5 0 0.8]);hold on;
plot(1:100,DE_LL(1:100),'k');hold on;
plot(1:100,TAGI_JMLR(1:100),'--r');hold on;
plot(1:100,TAGI_2L(1:100),'.r');
xlabel('Epoch')
ylabel('LL')
ylim([-15, 1])
legend('TAGI-BNI', 'PBP', 'MCD','DVI','DE','TAGI','TAGI-BNI 2L')
title('Test Log-Likelihood yacht')
drawnow

%% RMSE
filename1 = 'UCI_yachtRMSEtest400Epoch.txt';
filename2 = 'PBP/RMSEtest_PBP.txt';
filename3 = 'MCD/RMSEtest_MCD.txt';
filename4 = 'DVI/RMSEtest_DVI.txt';
filename5 = 'results_DE/RMSEtest_DE.txt';
filename6 = 'UCI_TAGI_JMLR_yachtRMSEtest400Epoch.txt';
filename7 = 'UCI_yachtRMSEtest_2L100nodes.txt';
TAGI_2L   = importdata(filename7);
TAGI_JMLR = importdata(filename6);
TAGI_LL   = importdata(filename1);
PBP_LL    = importdata(filename2);
MCD_LL    = importdata(filename3);
DVI_LL    = importdata(filename4);
DE_LL     = importdata(filename5);
FigHandle = figure;
set(FigHandle, 'Position', [100, 100, 600, 400])
plot(1:100,TAGI_LL(1:100),'r');hold on;
plot(1:100,PBP_LL(1:100),'b');hold on;
plot(1:100,MCD_LL(1:100),'g');hold on;
plot(1:100,DVI_LL(1:100),'color',[0.5 0 0.8]);hold on;
plot(1:100,DE_LL(1:100),'k');hold on;
plot(1:100,TAGI_JMLR(1:100),'--r');hold on;
plot(1:100,TAGI_2L(1:100),'.r');
xlabel('Epoch')
ylabel('LL')
ylim([0, 16])
legend('TAGI-BNI', 'PBP', 'MCD','DVI','DE','TAGI','TAGI-BNI 2L')
title('Test RMSE yacht')
drawnow