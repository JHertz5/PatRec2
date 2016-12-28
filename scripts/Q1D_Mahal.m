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
load wine_covMatrix

%% Mahalanobis Distance Using Individual Class Covariance Matrices from B.b

numTraining = length(training_classes);
numTesting = length(testing_classes);

mahalDistance_cov1 = zeros(numTesting, numTraining);
mahalDistance_cov2 = zeros(numTesting, numTraining);
mahalDistance_cov3 = zeros(numTesting, numTraining);
mahalDistance_covAll = zeros(numTesting, numTraining);

mahalClass_cov1 = zeros(1,numTesting);
mahalClass_cov2 = zeros(1,numTesting);
mahalClass_cov3 = zeros(1,numTesting);
mahalClass_covAll = zeros(1,numTesting);

for testingIndex = 1:numTesting
    %test one vector of testing data at a time
    
    % Class 1 covariance matrix
    G = chol(cov_1Norm^-1);
    mahalDistance_cov1(testingIndex,:) = sum((G*testing_norm(testingIndex,:)' - G*training_norm').^2);
    [~,minIndex] = min(mahalDistance_cov1(testingIndex,:));
    mahalClass_cov1(testingIndex) = training_classes(minIndex);
    
    % Class 2 covariance matrix
    G = chol(cov_2Norm^-1);
    mahalDistance_cov2(testingIndex,:) = sum((G*testing_norm(testingIndex,:)' - G*training_norm').^2);
    [~,minIndex] = min(mahalDistance_cov2(testingIndex,:));
    mahalClass_cov2(testingIndex) = training_classes(minIndex);
    
    % Class 3 covariance matrix
    G = chol(cov_3Norm^-1);
    mahalDistance_cov3(testingIndex,:) = sum((G*testing_norm(testingIndex,:)' - G*training_norm').^2);
    [~,minIndex] = min(mahalDistance_cov3(testingIndex,:));
    mahalClass_cov3(testingIndex) = training_classes(minIndex);
    
    % Full covariance matrix
    G = chol(cov_allNorm^-1);
    mahalDistance_covAll(testingIndex,:) = sum((G*testing_norm(testingIndex,:)' - G*training_norm').^2);
    [~,minIndex] = min(mahalDistance_covAll(testingIndex,:));
    mahalClass_covAll(testingIndex) = training_classes(minIndex);
end

%% Evaluate results

% Find failure indices
mahalFailureIndices_cov1 = find(testing_classes ~= mahalClass_cov1);
mahalFailureIndices_cov2 = find(testing_classes ~= mahalClass_cov2);
mahalFailureIndices_cov3 = find(testing_classes ~= mahalClass_cov3);
mahalFailureIndices_covAll = find(testing_classes ~= mahalClass_covAll);

mahalSuccess_cov1 = ones(1,numTesting);
mahalSuccess_cov1(mahalFailureIndices_cov1) = 0;

mahalSuccess_cov2 = ones(1,numTesting);
mahalSuccess_cov2(mahalFailureIndices_cov2) = 0;

mahalSuccess_cov3 = ones(1,numTesting);
mahalSuccess_cov3(mahalFailureIndices_cov3) = 0;

mahalSuccess_covAll = ones(1,numTesting);
mahalSuccess_covAll(mahalFailureIndices_covAll) = 0;

mahalAcc_cov1 = sum(mahalSuccess_cov1)*100/numTesting;
mahalAcc_cov2 = sum(mahalSuccess_cov2)*100/numTesting;
mahalAcc_cov3 = sum(mahalSuccess_cov3)*100/numTesting;
mahalAcc_covAll = sum(mahalSuccess_covAll)*100/numTesting;

%% Plot graphs

figure
plot(mahalDistance_cov1(1,:))
ylabel('Mahalanobis Distance','interpreter','latex','fontsize',30)
xlabel('Training Data Index','interpreter','latex','fontsize',30)
line([40 40], ylim, 'Color', 'black')
line([87 87], ylim, 'Color', 'black')


