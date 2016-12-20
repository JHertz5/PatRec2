%%

clc
%clear all
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

%%

% analyse covariance matrices. -> Find dimensions along whicht here is most
% variance. Most dependable dimensions for ea h individual class and
% combined classes.
% Each class can have their own set of dimensions of most variance -> that
% dimension will enable identification
% Cov matrix for all classes will tell us the set of dimensions along which
% all three classes vary the most. They wont necessarily have to be the
% same as those obtained from individual classes.

%% Find dimensions along which there is most covariance
matrices = cat(3,cov_1Norm,cov_1Raw,cov_2Norm,cov_2Raw,cov_3Norm,cov_3Raw,cov_allNorm,cov_allRaw);
names = {'cov_1Norm','cov_1Raw','cov_2Norm','cov_2Raw','cov_3Norm','cov_3Raw','cov_allNorm','cov_allRaw'};
for j = 1:8
    for i = 1:5
        matrix = matrices(:,:,j); %which matrix to use
        pos = i; % find nth highest value
        [svals,idx] = sort(matrix(:),'descend'); % sort to vector
        [m,n] = ind2sub(size(matrix),idx(pos)); % position in the matrix
        fprintf('%s ::: %i pos. var. ::: dims. %i : %i \n\n',string(names(j)),pos, m, n);
        
        pos = 170-i;
        [m,n] = ind2sub(size(matrix),idx(pos)); % position in the matrix
        fprintf('%s ::: %i neg. var. ::: dims. %i : %i \n\n',string(names(j)),i, m, n);
    end
end
%%

%figure(1)
%scatter(training_raw(:,m),training_raw(:,n),'filled') %plots the data along the two dimensions found above

