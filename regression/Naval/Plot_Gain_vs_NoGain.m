clear;clc;
filename1 = 'UCI_navalLLtest100Epoch_Gain_1_.txt';
filename2 = 'UCI_navalLLtest400Epoch.txt';
TAGI_NoGain    = importdata(filename1);
TAGI_Gain    = importdata(filename2);
figure;
scatter(1:100,TAGI_NoGain,'k');hold on;scatter(1:100,TAGI_Gain(1:100),'r');
xlabel('Epoch')
ylabel('LL')
%ylim([-7, -2.40])
legend('He init', 'Modified He init')
title('Test Log-Likelihood Naval')
drawnow

filename1 = 'UCI_navalRMSEtest100Epoch_Gain_1_.txt';
filename2 = 'UCI_navalRMSEtest400Epoch.txt';
TAGI_NoGain    = importdata(filename1);
TAGI_Gain    = importdata(filename2);
figure;
scatter(1:100,TAGI_NoGain,'k');hold on;scatter(1:100,TAGI_Gain(1:100),'r');
xlabel('Epoch')
ylabel('RMSE')
%ylim([-7, -2.40])
legend('He init', 'Modified He init')
title('Test RMSE Naval')
drawnow
