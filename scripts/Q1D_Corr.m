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

%% Correlation Metric Norm

Corrclasses = zeros(1,length(testing_norm));

for i = 1:length(testing_norm)
    w = zeros(1,length(training_norm));
    for j = 1:length(training_norm)
        % normalise the norm vectors again so the identity corr(x,y) = x*y
        % holds (normally corr(x,y) = (x-y)^@ = x^2 + y^2 - 2xy, where x^2
        % and y^2 are consts.
        
        w(j) = sum(((testing_norm(i,:)./sum(testing_norm(i,:))) .* (training_norm(j,:)./sum(training_norm(j,:)))));
    end
    [minVal, idx] = max(w);
    Corrclasses(i) = training_classes(idx);
    
    clear w
end


CorrAccNorm = (length(testing_norm)-nnz(Corrclasses - testing_classes))*100/length(testing_norm);

%% Correlation Metric Raw

Corrclasses = zeros(1,length(testing_norm));

for i = 1:length(testing_raw)
    w = zeros(1,length(training_raw));
    for j = 1:length(training_raw)
        % normalise the norm vectors again so the identity corr(x,y) = x*y
        % holds (normally corr(x,y) = (x-y)^@ = x^2 + y^2 - 2xy, where x^2
        % and y^2 are consts.
        
        w(j) = sum((testing_raw(i,:) - training_raw(j,:)).^2);
    end
    [minVal, idx] = min(w);
    Corrclasses(i) = training_classes(idx);
    
    clear w
end


CorrAccRaw = (length(testing_norm)-nnz(Corrclasses - testing_classes))*100/length(testing_norm);
