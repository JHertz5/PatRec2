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

%% HISTOGRAM INTERSECTION FOR RAW DATA
maxRange = 1700;

% extract training data for each class and storem them as a vector (not
% matrix)
class1 = find(training_classes == 1);
[m,n] = size(training_raw(class1,:));
class1dataRaw = reshape(training_raw(class1,:),m*n,1);

class2 = find(training_classes == 2);
[m,n] = size(training_raw(class2,:));
class2dataRaw = reshape(training_raw(class2,:),m*n,1);

class3 = find(training_classes == 3);
[m,n] = size(training_raw(class3,:));
class3dataRaw = reshape(training_raw(class3,:),m*n,1);

for hh = 1:100
    numBins = hh; % +1, user variable.
    
    % one histogram per class
    
    %training histograms for norm data
    
    % bin width
    binW = maxRange/numBins;
    
    % setup bins
    bins = 0:binW:maxRange;
    
    % setups bin heights for three classes
    s = zeros(numBins+1,3);
    figure(1)
    % calculate bin heights for class 1 (norm)
    [nb1,xb1] = hist(class1dataRaw,bins);
    s(:,1) = nb1./(sum(nb1)*binW);
    
    % calculate bin heights for class 2 (norm)
    [nb2,xb2] = hist(class2dataRaw,bins);
    s(:,2) = nb2./(sum(nb2)*binW);
    
    % calculate bin heights for class 3 (norm)
    [nb3,xb3] = hist(class3dataRaw,bins);
    s(:,3) = nb3./(sum(nb3)*binW);
    
    %% testing data using one 13-element vector
    
    % test a single vector (norm)
    for ll = 1:length(testing_raw)
        
        [nbt,xbt] = hist(testing_raw(ll,:),bins);
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
                assClass(ll) = 1;
            case score2
                assClass(ll) = 2;
            case score3
                assClass(ll) = 3;
        end
        
    end
    acc(hh) = (1-nnz(assClass - testing_classes)/length(testing_classes))*100;
end

filacc = filter(0.167*[1 1 1 1 1 1],[1],acc);

figure(1)
plot(1:hh,acc,'linewidth',2)
hold all
plot(-2:(hh-3),filacc,'linewidth',2)
title('Accuracy of Histogram Intersection Classification Method (RAW)')
xlabel('Number of histogram bins')
ylabel('Accuracy [%]')
set(gca,'fontsize',20)
xlim([0 hh])
grid on
grid minor