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

class1Indexes = find(training_classes == 1);
class2Indexes = find(training_classes == 2);
class3Indexes = find(training_classes == 3);

mahalDistance_covSeparate = zeros(numTesting, numTraining);
mahalDistance_covAll = zeros(numTesting, numTraining);

mahalClass_covSeparate = zeros(1,numTesting);
mahalClass_covAll = zeros(1,numTesting);

for testingIndex = 1:numTesting
    %test one vector of testing data at a time
    
    % Class 1 covariance matrix
    G = chol(cov_1Norm^-1); % Cholesky Decomposition
    mahalDistance_covSeparate(testingIndex,:) = sum((G*testing_norm(testingIndex,:)' - G*training_norm').^2);
    
    % Class 2 covariance matrix
    G = chol(cov_2Norm^-1); % Cholesky Decomposition
    mahalDistance_class2Temp = sum((G*testing_norm(testingIndex,:)' - G*training_norm').^2);
    mahalDistance_covSeparate(testingIndex,:) = min(mahalDistance_covSeparate(testingIndex,:), mahalDistance_class2Temp);
    
    % Class 3 covariance matrix
    G = chol(cov_3Norm^-1); % Cholesky Decomposition
    mahalDistance_class3Temp = sum((G*testing_norm(testingIndex,:)' - G*training_norm').^2);
    mahalDistance_covSeparate(testingIndex,:) = min(mahalDistance_covSeparate(testingIndex,:), mahalDistance_class3Temp);
    
    [~,minIndex] = min(mahalDistance_covSeparate(testingIndex,:));
    mahalClass_covSeparate(testingIndex) = training_classes(minIndex);
    
    % Full covariance matrix
    G = chol(cov_allNorm^-1); % Cholesky Decomposition
    mahalDistance_covAll(testingIndex,:) = sum((G*testing_norm(testingIndex,:)' - G*training_norm').^2);
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

mahalAcc_covSeparate = sum(mahalSuccess_covSeparate)*100/numTesting
mahalAcc_covAll = sum(mahalSuccess_covAll)*100/numTesting

%% Plot graphs

figure
plot(mahalSuccess_covSeparate, 'LineWidth', 2)
% plot(mahalDistance_covSeparate(20,:), 'LineWidth', 2)
hold on
plot(mahalSuccess_covAll, 'LineWidth', 2)
% plot(mahalDistance_covAll(20,:), 'LineWidth', 2)
% ylabel('Mahalanobis Distance','interpreter','latex','fontsize',30)
% xlabel('Training Data Index','interpreter','latex','fontsize',30)
ylabel('Successful Classification','interpreter','latex','fontsize',30)
xlabel('Training Data Index','interpreter','latex','fontsize',30)
% line([40 40], ylim, 'Color', 'black')
% line([87 87], ylim, 'Color', 'black')
line([13 13], ylim, 'Color', 'black', 'LineWidth', 2)
line([29 29], ylim, 'Color', 'black', 'LineWidth', 2)

legend('Separated Data Cov. Matrix', 'All Data Cov. Matrix', 'Class Boundary')

