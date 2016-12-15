clc
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

%% Estimate Covariance Matrices for all classes' training data

cov_allRaw = cov(training_raw);
cov_allNorm = cov(training_norm);

%% Estimate Covariance Matrices for individual classes' training data

class1Indexes = find(training_classes == 1);
class2Indexes = find(training_classes == 2);
class3Indexes = find(training_classes == 3);

cov_1Raw = cov(training_raw( class1Indexes, : ));
cov_1Norm = cov(training_norm( class1Indexes, : ));
cov_2Raw = cov(training_raw( class2Indexes, : ));
cov_2Norm = cov(training_norm( class2Indexes, : ));
cov_3Raw = cov(training_raw( class3Indexes, : ));
cov_3Norm = cov(training_norm( class3Indexes, : ));

%%
if ~isempty(dataPath)
    save(char(strcat(dataPath, '/wine_covMatrix')),'cov_allRaw','cov_allNorm','cov_1Raw','cov_1Norm','cov_2Raw','cov_2Norm','cov_3Raw','cov_3Norm')
else
    save('wine_covMatrix','cov_allRaw','cov_allNorm','cov_1Raw','cov_1Norm','cov_2Raw','cov_2Norm','cov_3Raw','cov_3Norm')
end