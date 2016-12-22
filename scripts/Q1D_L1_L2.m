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

%% L1 Metric Norm

L1classes = zeros(1,length(testing_norm));

for i = 1:length(testing_norm)
    w = zeros(1,length(training_norm));
    for j = 1:length(training_norm)
        
        w(j) = sum(abs(testing_norm(i,:) - training_norm(j,:)));
        
    end
    [minVal, idx] = min(w);
    L1classes(i) = training_classes(idx);
end


L1accNorm = (length(testing_norm)-nnz(L1classes - testing_classes))*100/length(testing_norm);

%% L1 Metric Raw

L1classes = zeros(1,length(testing_raw));

for i = 1:length(testing_raw)
    w = zeros(1,length(training_raw));
    
    for j = 1:length(training_raw)
        w(j) = sum(abs(testing_raw(i,:) - training_raw(j,:)));
    end
    [minVal, idx] = min(w);
    L1classes(i) = training_classes(idx);
    h  = w;
    clear w
end


L1accRaw = (length(testing_raw)-nnz(L1classes - testing_classes))*100/length(testing_raw);

%% L2 Metric Norm

L2classes = zeros(1,length(testing_norm));

for i = 1:length(testing_norm)
    w = zeros(1,length(training_norm));
    
    for j = 1:length(training_norm)
        w(j) = sqrt(sum((testing_norm(i,:)-training_norm(j,:)).^2));
    end
    [minVal, idx] = min(w);
    L2classes(i) = training_classes(idx);
    clear w
end


L2accNorm = (length(testing_norm)-nnz(L2classes - testing_classes))*100/length(testing_norm);

%% L2 Metric Raw

L2classes = zeros(1,length(testing_raw));

for i = 1:length(testing_raw)
    
    w = zeros(1,length(training_raw));
    
    for j = 1:length(training_raw)
        w(j) = sqrt(sum((testing_raw(i,:)-training_raw(j,:)).^2));
    end
    [minVal, idx] = min(w);
    L2classes(i) = training_classes(idx);
    clear w
end


L2accRaw = (length(testing_raw)-nnz(L2classes - testing_classes))*100/length(testing_raw);

