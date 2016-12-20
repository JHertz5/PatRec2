%%

% run the script
% it allows you to choose dimensions for each case along which there is
% most variance/covariance (pos or neg).

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

%% User Variables
numMaxMin = 5; % select hom many sest of dimenisons (of max pos and max neg variance) we want
toPlot = 1; % choose which matrix' data to plot: 1-cov1Norm, 2-cov1Raw ... 7-cov_allNorm, 8-cov_allRaw
%% Find dimensions along which there is most covariance

% put all covs on a single matrix
matrices = cat(3,cov_1Norm,cov_1Raw,cov_2Norm,cov_2Raw,cov_3Norm,cov_3Raw,cov_allNorm,cov_allRaw);

% store names for printf
names = {'cov_1Norm','cov_1Raw','cov_2Norm','cov_2Raw','cov_3Norm','cov_3Raw','cov_allNorm','cov_allRaw'};

%initialise dimension matrix
dims = zeros(2*numMaxMin,2,8);
cov_vals = zeros(2*numMaxMin,8);

for j = 1:8 %iterate thorugh all matrices
    for i = 1:numMaxMin %get all sets of dims
        
        matrix = matrices(:,:,j); % which matrix to use
        pos = i; % find nth highest value
        [svals,idx] = sort(matrix(:),'descend'); % sort to vector
        [m,n] = ind2sub(size(matrix),idx(pos)); % position in the matrix
        fprintf('%s ::: %i pos. var. ::: dims. %i : %i \n\n',string(names(j)),pos, m, n);
        dims(i,:, j) = [m,n]; % store
        cov_vals(i,j) = svals(i);
        
        pos = 170-i; % find nth most negative value
        [m,n] = ind2sub(size(matrix),idx(pos)); % position in the matrix
        fprintf('%s ::: %i neg. var. ::: dims. %i : %i \n\n',string(names(j)),i, m, n);
        dims(2*numMaxMin+1-i,:, j) = [m,n]; % store
        cov_vals(2*numMaxMin+1-i,j) = svals(end+1-i);
    end
end
%%

[V,D] = eig(cov_1Norm);


figure(1)
for i = 1:2*numMaxMin
    subplot(2,5,i)
    hold all
    title(['Cov Val = ' num2str(cov_vals(i,toPlot))]);
    xlabel(num2str(dims(i,1,toPlot)))
    ylabel(num2str(dims(i,2,toPlot)))
    scatter(training_raw(class1Indexes,dims(i,1,toPlot)),training_raw(class1Indexes,dims(i,2,toPlot)),'filled') %plots the data along the two dimensions found above
end
