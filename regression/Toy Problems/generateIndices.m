%% Creating indices for train and test set
%author - Bhargob Deka
clear; clc;
L     = 1000;
A     = [1:1:1000]';
format shortG
trainIdx = cell(5,1);
testIdx  = cell(5,1);
for i = 0:4
    % Cross varidation (train: 90%, test: 10%)
    cv = cvpartition(L,'HoldOut',0.1);
    index_train = training(cv);
    index_test  = test(cv);
    idx = index_train == index_test;
    if ~any(idx~=0)
        disp('We are good :)')
        Idx_train   = A(index_train);
        Idx_test    = A(index_test);
    end
    trainIdx{i+1} = Idx_train;
    testIdx{i+1}  = Idx_test;
%     dlmwrite(['index_train_' num2str(i) '.txt'],index_train);
%     dlmwrite(['index_test_' num2str(i) '.txt'],index_test);
end
filename    = 'DepewegTrainIndices.mat';
save(filename, 'trainIdx')
filename    = 'DepewegTestIndices.mat';
save(filename, 'testIdx')