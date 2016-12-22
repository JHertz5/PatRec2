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

%% Chi2 Metric Norm

Chi2classes = zeros(1,length(testing_norm));

for i = 1:length(testing_norm)
    w = zeros(1,length(training_norm));
    for j = 1:length(training_norm)
        w(j) = 0.5*sum(((testing_norm(i,:) - training_norm(j,:)).^2)./(testing_norm(i,:) + training_norm(j,:)));
    end
    [minVal, idx] = min(w);
    Chi2classes(i) = training_classes(idx);
end


Chi2accNorm = (length(testing_norm)-nnz(Chi2classes - testing_classes))*100/length(testing_norm);

%% Chi2 Metric Raw

Chi2classes = zeros(1,length(testing_raw));

for i = 1:length(testing_raw)
    w = zeros(1,length(training_raw));
    for j = 1:length(training_raw)
        
        w(j) = 0.5*sum(((testing_raw(i,:) - training_raw(j,:)).^2)./(testing_raw(i,:) + training_raw(j,:)));
      
    end
    [minVal, idx] = min(w);
    Chi2classes(i) = training_classes(idx);
end


Chi2accRaw = (length(testing_raw)-nnz(Chi2classes - testing_classes))*100/length(testing_raw);
