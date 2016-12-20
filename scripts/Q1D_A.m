%%

clc
clear all
close all

%add data directory to path
if contains(pwd, 'NotPatRecCW2')
    dataPath = strcat( extractBefore(pwd, 'NotPatRecCW2'), 'NotPatRecCW2/data');
    addpath(char(dataPath));
else
    dataPath = ''; %dataPath is empty vector
    fprintf('Move to NotPatRecCW2 directory\n');
end

%load raw data
load wine_separatedData.mat
load wine_covMatrix

%% K-NN for Raw Feature Vectors

% k-NN, with k = 1
% from each testing vector sub every training vector and find min
w = zeros(1,length(training_raw));
NNclasses = zeros(1, length(testing_raw));
for i = 1:length(testing_raw)
    for j = 1:length(training_raw)
        w(j) = norm(testing_raw(i,:) - training_raw(j,:));
    end
    [minVal, idx] = min(w);
    NNclasses(i) = training_classes(idx);
end

%% Calculate accuracy
accVal1 = (length(testing_raw)-nnz(NNclasses - testing_classes))*100/length(testing_raw);

%% K-NN for Norm Feature Vectors

% k-NN, with k = 1
% from each testing vector sub every training vector and find min
w = zeros(1,length(training_norm));
NNclasses = zeros(1, length(testing_norm));
for i = 1:length(testing_norm)
    for j = 1:length(training_norm)
        w(j) = norm(testing_norm(i,:) - training_norm(j,:));
    end
    [minVal, idx] = min(w);
    NNclasses(i) = training_classes(idx);
end

%% Calculate accuracy
accVal2 = (length(testing_raw)-nnz(NNclasses - testing_classes))*100/length(testing_raw);
