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

%% Mahalanobis Distance Using Covraince Matrices from B.b

class1 = find(training_classes == 1);
class2 = find(training_classes == 2);
class3 = find(training_classes == 3);

Mahalclasses = zeros(1,length(testing_norm));

% this is the same as doing (x-m)^T cov^-1 (x-m) for each cov matrix
w(1,:) = mahal(testing_norm(:,:),training_norm(class1,:));
w(2,:) = mahal(testing_norm(:,:),training_norm(class2,:));
w(3,:) = mahal(testing_norm(:,:),training_norm(class3,:));


for i = 1:length(testing_norm)
    switch min(w(:,i))
        case w(1,i)
            Mahalclasses(i) = 1;
        case w(2,i)
            Mahalclasses(i) = 2;
        case w(3,i)
            Mahalclasses(i) = 3;
    end
end

MahalAccNorm = (length(testing_norm)-nnz(Mahalclasses - testing_classes))*100/length(testing_norm);

%% %% Mahalanobis Distance Using Covraince Matrices from B.b