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
load wine.data.csv

partitionSizes = [ 118 20 40 ]; % training validation testing
classSizes = [ 59 71 48 ]; % 1 2 3
numVectors = size(wine_data,1);
numPartitions = 3;
numClasses = 3;

%% extract raw data
featureVectors_classes = wine_data(:,1);
featureVectors_raw = wine_data(:,2:14);
featureVectors_norm = featureVectors_raw/norm(featureVectors_raw,2);

%% partition data
training_classes = zeros(1, 118);
validation_classes = zeros(1, 20);
testing_classes= zeros(1, 40);

training_raw = zeros(118, 13);

classIndexes = [ 
%class: 1  2   3
        59 130 178 ];
                    
classPartitionIndex = [ 
%class: 1  2  3
        0  59  130;
        39 106 162;  % training
        46 114 167;  % validation
        59 130 178]; % testing
      
partitionVectorIndex = [
%class: trn val tst
        0   0   0 ;
        39  7   13;  % 1
        86  15  29;  % 2
        118 20  40]; % 3
    
    
    
for i = 1:numClasses
    training_classes( partitionVectorIndex(i, 1)+1 : partitionVectorIndex(i+1, 1) ) = featureVectors_classes( classPartitionIndex(1, i)+1 : classPartitionIndex(2, i) );
    validation_classes( partitionVectorIndex(i, 2)+1 : partitionVectorIndex(i+1, 2) ) = featureVectors_classes( classPartitionIndex(2, i)+1 : classPartitionIndex(3, i) );
    testing_classes( partitionVectorIndex(i, 3)+1 : partitionVectorIndex(i+1, 3) ) = featureVectors_classes( classPartitionIndex(3, i)+1 : classPartitionIndex(4, i) );
    
    training_raw( partitionVectorIndex(i, 1)+1 : partitionVectorIndex(i+1, 1), : ) = featureVectors_raw( classPartitionIndex(1, i)+1 : classPartitionIndex(2, i), : );
    validation_raw( partitionVectorIndex(i, 2)+1 : partitionVectorIndex(i+1, 2), : ) = featureVectors_raw( classPartitionIndex(2, i)+1 : classPartitionIndex(3, i), : );
    testing_raw( partitionVectorIndex(i, 3)+1 : partitionVectorIndex(i+1, 3), : ) = featureVectors_raw( classPartitionIndex(3, i)+1 : classPartitionIndex(4, i), : );
    
    training_norm( partitionVectorIndex(i, 1)+1 : partitionVectorIndex(i+1, 1), : ) = featureVectors_norm( classPartitionIndex(1, i)+1 : classPartitionIndex(2, i), : );
    validation_norm( partitionVectorIndex(i, 2)+1 : partitionVectorIndex(i+1, 2), : ) = featureVectors_norm( classPartitionIndex(2, i)+1 : classPartitionIndex(3, i), : );
    testing_norm( partitionVectorIndex(i, 3)+1 : partitionVectorIndex(i+1, 3), : ) = featureVectors_norm( classPartitionIndex(3, i)+1 : classPartitionIndex(4, i), : );
end
                    

if ~isempty(dataPath)
    save(char(strcat(dataPath, '/wine_featureVectors')),'featureVectors_classes','featureVectors_raw','featureVectors_norm')
    save(char(strcat(dataPath, '/wine_separatedData')),'training_classes','validation_classes','testing_classes','training_raw','validation_raw','testing_raw','training_norm','validation_norm','testing_norm')
else
    save('wine_featureVectors','featureVectors_classes','featureVectors_raw','featureVectors_norm')
    save('wine_separatedData','training_classes','validation_classes','testing_classes','training_raw','validation_raw','testing_raw','training_norm','validation_norm','testing_norm')
end