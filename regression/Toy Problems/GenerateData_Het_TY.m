clear 
% clc
% close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
rand_seed=4;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
%% Data
rng(1223) % Seed
ntrain     = 40;

f          = @(x) (5*x).^3/50;
noise      = @(x) (3*(x).^4)+0.02;      %(3*(x).^4)+0.02, (3*(x).^4)+0.01, (0.1*(x+0.5).^3)+0.1,  (0.4*(x).^4)+0.01, 0.45.*(x+0.5).^2+0.02

xtrain     = unifrnd(-1,1,[ntrain,1]);
xtrue_plot = linspace(-1,1,100);
noiseTrain = noise(xtrain);

ytrue_plot = f(xtrue_plot);
ytrain     = f(xtrain) + normrnd(zeros(length(noiseTrain), 1), sqrt(noiseTrain));

nx         = size(xtrain, 2);
ny         = size(ytrain, 2);

figure;
scatter(xtrain,ytrain,'ob');hold on
plot(xtrue_plot,ytrue_plot,'k')
xlabel('x');
ylabel('y');

TrainData_TY(:,1) = xtrain;
TrainData_TY(:,2) = ytrain;
save('TrainData_TY_100.mat','TrainData_TY');