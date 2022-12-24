clear;clc;
filename1 = 'UCI_wineLLtest100Epoch_Gain_1_.txt';
filename2 = 'UCI_wineLLtest400Epoch.txt';
TAGI_NoGain    = importdata(filename1);
TAGI_Gain    = importdata(filename2);
figure;
scatter(1:100,TAGI_NoGain,'k');hold on;scatter(1:100,TAGI_Gain(1:100),'r');
xlabel('Epoch')
ylabel('LL')
%ylim([-7, -2.40])
legend('He init', 'Modified He init')
title('Test Log-Likelihood Wine')
drawnow

filename1 = 'UCI_wineRMSEtest100Epoch_Gain_1_.txt';
filename2 = 'UCI_wineRMSEtest400Epoch.txt';
TAGI_NoGain    = importdata(filename1);
TAGI_Gain    = importdata(filename2);
figure;
scatter(1:100,TAGI_NoGain,'k');hold on;scatter(1:100,TAGI_Gain(1:100),'r');
xlabel('Epoch')
ylabel('RMSE')
%ylim([-7, -2.40])
legend('He init', 'Modified He init')
title('Test RMSE Wine')
drawnow
