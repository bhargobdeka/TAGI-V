clear;clc
filename = 'DVI/lltest_DVI.txt';
DVI_LL    = importdata(filename);
FigHandle = figure;
set(FigHandle, 'Position', [100, 100, 600, 400])
plot((1:100)*1498.41,DVI_LL(1:100),'r');hold on;
xlabel('time (s)')
ylabel('LL')
%ylim([0, 16])
legend('DVI')
title('Test protein DVI')
drawnow

filename = 'DVI/RMSEtest_DVI.txt';
DVI_LL    = importdata(filename);
FigHandle = figure;
set(FigHandle, 'Position', [100, 100, 600, 400])
plot((1:100)*1498.41,DVI_LL(1:100),'r');hold on;
xlabel('time (s)')
ylabel('RMSE')
%ylim([0, 16])
legend('DVI')
title('Test protein DVI')
drawnow