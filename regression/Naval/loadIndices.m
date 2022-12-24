%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         loadIndices
% Description:  Load reference indices from dropout paper Gal et al. (2015)
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      December 12, 2019
% Updated:      December 13, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test set
path    = char([cd ,'/dataIndices/test']);
d       = dir(path);
Nfiles  = numel(d);
myfile  = fullfile(path, {d.name});
testIdx = cell(20, 1);
N       = 3;
for i = N:Nfiles
    fid  = fopen(myfile{i},'r');
    data = textscan(fid,'%f %[^\n]','Headerlines',0,'emptyvalue',0);
    testIdx{i-(N-1)} = data{1}+1;
    fclose(fid);
end

% Training set
strcd    = cd;
path     = char([cd ,'/dataIndices/train']);
d        = dir(path);
out      = [];
Nfiles   = numel(d);
myfile   = fullfile(path, {d.name});
trainIdx = cell(20, 1);
N        = 3;
for i = N:Nfiles
    fid  = fopen(myfile{i},'r');
    data = textscan(fid,'%f %[^\n]','Headerlines',0,'emptyvalue',0);
    trainIdx{i-(N-1)} = data{1}+1;
    fclose(fid);
end

% Checking 
check = zeros(20,1);
for i = 1:20
    idx = trainIdx{i} == testIdx{i}';
    check(i) = any(any(idx~=0)); 
end
if any(check==0)
    disp('We are good :)')
end

% Save
folder      = char([cd ,'/data/']);
filename    = 'navalTestIndices.mat';
save([folder, filename], 'testIdx')
folder      = char([cd ,'/data/']);
filename    = 'navalTrainIndices.mat';
save([folder, filename], 'trainIdx')



