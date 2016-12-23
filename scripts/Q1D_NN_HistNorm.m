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

%% K-NN for Raw Feature Vectors

% k-NN, with k = 1
% from each testing vector sub every training vector and find min
w = zeros(1,length(training_raw));
NNclasses = zeros(1, length(testing_raw));
for i = 1:length(testing_raw)
    for j = 1:length(training_raw)
        w(j) = norm(testing_raw(i,:) - training_raw(j,:));
    end
    [minVal, idx] = min(w);
    NNclasses(i) = training_classes(idx);
end

%% Calculate accuracy
accVal1 = (length(testing_raw)-nnz(NNclasses - testing_classes))*100/length(testing_raw);

%% K-NN for Norm Feature Vectors

% k-NN, with k = 1
% from each testing vector sub every training vector and find min
w = zeros(1,length(training_norm));
NNclasses = zeros(1, length(testing_norm));
for i = 1:length(testing_norm)
    for j = 1:length(training_norm)
        w(j) = norm(testing_norm(i,:) - training_norm(j,:));
    end
    [minVal, idx] = min(w);
    NNclasses(i) = training_classes(idx);
end

%% Calculate accuracy
accVal2 = (length(testing_raw)-nnz(NNclasses - testing_classes))*100/length(testing_raw);


%% HISTOGRAM INTERSECTION FOR NORM DATA

% extract training data for each class and storem them as a vector (not
% matrix)

class1 = find(training_classes == 1);
[m,n] = size(training_raw(class1,:));
class1dataNorm = reshape(training_norm(class1,:),m*n,1);

class2 = find(training_classes == 2);
[m,n] = size(training_raw(class2,:));
class2dataNorm = reshape(training_norm(class2,:),m*n,1);

class3 = find(training_classes == 3);
[m,n] = size(training_raw(class3,:));
class3dataNorm = reshape(training_norm(class3,:),m*n,1);

for kk = 1:100
    numBins = kk; % +1, user variable.
    
    % one histogram per class
    
    %training histograms for norm data
    
    % bin width
    binW = 0.2/numBins;
    
    % setup bins
    bins = 0:binW:0.2;
    
    % setups bin heights for three classes
    s = zeros(numBins+1,3);
    
    % calculate bin heights for class 1 (norm)
    [nb1,xb1] = hist(class1dataNorm,bins);
    s(:,1) = nb1./(sum(nb1)*binW);
    
    % calculate bin heights for class 2 (norm)
    [nb2,xb2] = hist(class2dataNorm,bins);
    s(:,2) = nb2./(sum(nb2)*binW);
    
    % calculate bin heights for class 3 (norm)
    [nb3,xb3] = hist(class3dataNorm,bins);
    s(:,3) = nb3./(sum(nb3)*binW);
    
    %% testing data using one 13-element vector
    
    % test a single vector (norm)
    for jj = 1:length(testing_norm)
        
        [nbt,xbt] = hist(testing_norm(jj,:),bins);
        st = nbt./(sum(nbt)*binW);
        
        % calculate intersection
        for i = 1:numBins+1
            inter1(i) = min(s(i,1),st(i));
            inter2(i) = min(s(i,2),st(i));
            inter3(i) = min(s(i,3),st(i));
        end
        
        % calculate scores
        score1 = sum(inter1)*binW;
        score2 = sum(inter2)*binW;
        score3 = sum(inter3)*binW;
        
        switch max([score1 score2 score3])
            case score1
                assClass(jj) = 1;
            case score2
                assClass(jj) = 2;
            case score3
                assClass(jj) = 3;
        end
        
    end
    acc(kk) = (1-nnz(assClass - testing_classes)/length(testing_classes))*100;
end

filacc = filter(0.167*[1 1 1 1 1 1],[1],acc);
figure(1)
subplot(1,2,2)
plot(1:kk,acc,'linewidth',2)
hold all
plot(-2:(kk-3),filacc,'linewidth',2)
set(gca,'fontsize',15)
title('Accuracy of Histogram Union (NORM)','interpreter','latex','fontsize',30)
xlabel('Number of histogram bins','interpreter','latex','fontsize',30)
ylabel('Accuracy [\%]','interpreter','latex','fontsize',30)
xlim([0 kk])
grid on
grid minor
set(gca,'linewidth',1.5)

