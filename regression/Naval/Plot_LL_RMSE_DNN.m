clear;clc;
t = 0.32;
filename1 = 'results_DNN/lltest_DNN.txt';
filename2 = 'results_DNN/RMSEtest_DNN.txt';
DNN_LL    = importdata(filename1);
DNN_RMSE  = importdata(filename2);
%% epoch setting
FigHandle = figure;
set(FigHandle, 'Position', [100, 100, 600, 400])
plot((1:100),DNN_LL,'r','LineWidth',1);
xlabel('time(s)')
ylabel('LL')
%ylim([-7, -2])
legend('DNN') %'
title('Test Log-Likelihood ES Naval')

FigHandle = figure;
set(FigHandle, 'Position', [100, 100, 600, 400])
plot((1:100),DNN_RMSE,'r','LineWidth',1);
xlabel('Epoch')
ylabel('Log-likelihood')
%ylim([-7, -2])
legend('DNN') %'
title('Test RMSE ES Naval')
%% time setting
FigHandle = figure;
set(FigHandle, 'Position', [100, 100, 600, 400])
plot((1:100)*t,DNN_LL,'r','LineWidth',1);
xlabel('time (s)')
ylabel('Log-likelihood')
%ylim([-7, -2])
legend('DNN') %'
title('Test Log-Likelihood TS Naval')
fig = gcf;
ax = fig.CurrentAxes;
set(ax,'xscale','log')
drawnow

FigHandle = figure;
set(FigHandle, 'Position', [100, 100, 600, 400])
plot((1:100)*t,DNN_RMSE,'r','LineWidth',1);
xlabel('time (s)')
ylabel('RMSE')
%ylim([-7, -2])
legend('DNN') %'
title('Test RMSE TS Naval')
fig = gcf;
ax = fig.CurrentAxes;
set(ax,'xscale','log')
drawnow
