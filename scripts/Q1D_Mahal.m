%%

clc
clear variables
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
load wine_covMatrix.mat

%% Mahalanobis Distance Using Individual Class Covariance Matrices from B.b

numTraining = length(training_classes);
numTesting = length(testing_classes);
numClasses = 3;

class1Indexes = find(training_classes == 1);
class2Indexes = find(training_classes == 2);
class3Indexes = find(training_classes == 3);

mahalDistance_covSeparate = zeros(numTesting, numClasses);
mahalDistance_covAll = zeros(numTesting, numTraining);

mahalClass_covSeparate = zeros(1,numTesting,'uint8');
mahalClass_covAll = zeros(1,numTesting,'uint8');

for testingIndex = 1:numTesting
    %test one vector of testing data at a time
    
    % For separated cov matrices: 
    % For each class, use the class' cov matrix and find the smallest mahal
    % distance within that class. At the end, assign the testing point to
    % the class that had the smallest mahal distance
    
    % Class 1 covariance matrix
    G = chol(cov_1Norm^-1); % Cholesky Decomposition
    mahalDistance_covSeparate(testingIndex,1) = min(sum((G*testing_norm(testingIndex,:)' - G*training_norm(class1Indexes,:)').^2));
    
    % Class 2 covariance matrix
    G = chol(cov_2Norm^-1); % Cholesky Decomposition
    mahalDistance_covSeparate(testingIndex,2) = min(sum((G*testing_norm(testingIndex,:)' - G*training_norm(class2Indexes,:)').^2));

    % Class 3 covariance matrix
    G = chol(cov_3Norm^-1); % Cholesky Decomposition
    mahalDistance_covSeparate(testingIndex,3) = min(sum((G*testing_norm(testingIndex,:)' - G*training_norm(class3Indexes,:)').^2));

    % Find which class the testing point is closest to
    [~,minIndex] = min(mahalDistance_covSeparate(testingIndex,:));
    mahalClass_covSeparate(testingIndex) = minIndex;
    
    % For full cov matrix:
    % For each training point, use the full cov matrix and find the mahal 
    % distance from the testing point. At the end, assign the testing 
    % point to the class of the training point that had the smallest mahal 
    % distance
    
    G = chol(cov_allNorm^-1); % Cholesky Decomposition
    mahalDistance_covAll(testingIndex,:) = sum((G*testing_norm(testingIndex,:)' - G*training_norm').^2);
    
    % Find which training point the testing point is closest to
    [~,minIndex] = min(mahalDistance_covAll(testingIndex,:));
    mahalClass_covAll(testingIndex) = training_classes(minIndex);
    
end

%% Evaluate results

% Find failure indices
mahalFailureIndices_covSeparate = find(testing_classes ~= mahalClass_covSeparate);
mahalFailureIndices_covAll = find(testing_classes ~= mahalClass_covAll);

mahalSuccess_covSeparate = ones(1,numTesting);
mahalSuccess_covSeparate(mahalFailureIndices_covSeparate) = 0;
mahalSuccess_covAll = ones(1,numTesting);
mahalSuccess_covAll(mahalFailureIndices_covAll) = 0;

mahalAcc_covSeparate = sum(mahalSuccess_covSeparate)*100/numTesting;
mahalAcc_covAll = sum(mahalSuccess_covAll)*100/numTesting;

fprintf('Accuracy for separate covariance matrix for each class: %i%%\n', mahalAcc_covSeparate);
fprintf('Accuracy for single covariance matrix with all classes: %i%%\n', mahalAcc_covAll);

%% Plot graphs

figure
scatter(1:40,mahalClass_covSeparate,200,'LineWidth',2)
hold on
scatter(1:40,mahalClass_covAll,200,'x','LineWidth',2)

ylim([1 3])
line([13.5 13.5], ylim, 'Color', 'black', 'LineWidth', 2)
line([29.5 29.5], ylim, 'Color', 'black', 'LineWidth', 2)
set(gca,'YTick',[1 2 3],'XTick',[7 22 35], 'XTickLabels',{'1' '2' '3'},'FontSize', 20);
ylabel('Assigned Class','interpreter','latex','fontsize',30)
xlabel('True Class','interpreter','latex','fontsize',30)
