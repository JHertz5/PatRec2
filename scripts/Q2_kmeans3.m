%% gotta run it 1000 times and take average (done)
% Mahal is wrong
% MUST INCLUDE MAHALANOBIS and repeat for Q1D.b features

%clc
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

%load norm data
load wine_separatedData.mat
load wine_covMatrix.mat

plotfigs = 1;

if plotfigs == 1
    maxIter = 1;
else
    maxIter = 1000;
end

for hh = 1:maxIter
    %% clustering using scityblock
    
    [idx,C] = kmeans(training_norm,3,'distance','cityblock');
    
    clus1 = find(idx == 1);
    clus2 = find(idx == 2);
    clus3 = find(idx == 3);
    
    % assign class labels to C
    bins = 0.5:1:3.5;
    
    class1 = find(testing_classes == 1);
    class2 = find(testing_classes == 2);
    class3 = find(testing_classes == 3);
    
    
    [hb,nb] = hist(training_classes(clus1),bins);
    [~, id] = max(hb);
    
    cl(1) = id;
    
    [hb,nb] = hist(training_classes(clus2),bins);
    [~, id] = max(hb);
    
    cl(2) = id;
    
    [hb,nb] = hist(training_classes(clus3),bins);
    [~, id] = max(hb);
    
    cl(3) = id;
    
    %
    %
    % L1 DIST
    %
    %
    
    IDX1 = knnsearch(C,testing_norm,'distance','cityblock');
    
    for i = 1:length(IDX1)
        switch IDX1(i)
            case 1
                IDX1(i) = cl(1);
            case 2
                IDX1(i) = cl(2);
            case 3
                IDX1(i) = cl(3);
        end
    end
    
    k3acc1(1,hh) = (1-nnz(IDX1' - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % L2 DIST
    %
    %
    
    IDX2 = knnsearch(C,testing_norm,'distance','euclidean');
    
    for i = 1:length(IDX2)
        switch IDX2(i)
            case 1
                IDX2(i) = cl(1);
            case 2
                IDX2(i) = cl(2);
            case 3
                IDX2(i) = cl(3);
        end
    end
    
    k3acc2(1,hh) = (1-nnz(IDX2' - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % CORR DIST
    %
    %
    
    IDX3 = knnsearch(C,testing_norm,'distance','correlation');
    
    for i = 1:length(IDX3)
        switch IDX3(i)
            case 1
                IDX3(i) = cl(1);
            case 2
                IDX3(i) = cl(2);
            case 3
                IDX3(i) = cl(3);
        end
    end
    
    k3acc3(1,hh) = (1-nnz(IDX3' - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % CHISQ DIST
    %
    %
    
    for i = 1:length(testing_norm)
        w = zeros(1,size(C,1));
        for j = 1:size(C,1)
            w(j) = 0.5*sum(((testing_norm(i,:) - C(j,:)).^2)./(testing_norm(i,:) + C(j,:)));
        end
        [minVal, IDX4(i)] = min(w);
    end
    
    for i = 1:length(IDX4)
        switch IDX4(i)
            case 1
                IDX4(i) = cl(1);
            case 2
                IDX4(i) = cl(2);
            case 3
                IDX4(i) = cl(3);
        end
    end
    
    k3acc4(1,hh) = (1-nnz(IDX4 - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % HIST DIST
    %
    %
    
    for i = 1:length(testing_norm)
        w = zeros(1,size(C,1));
        for j = 1:size(C,1)
            
            w(j) = sum(min(testing_norm(i,:), C(j,:)));
            
        end
        [maxVal, IDX5(i)] = max(w);
    end
    
    for i = 1:length(IDX5)
        switch IDX5(i)
            case 1
                IDX5(i) = cl(1);
            case 2
                IDX5(i) = cl(2);
            case 3
                IDX5(i) = cl(3);
        end
    end
    
    k3acc5(1,hh) = (1-nnz(IDX5 - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % MAHAL
    %
    %
    
    w = zeros(3,length(testing_norm));
    G = chol(cov_allNorm^-1); % Cholesky Decomposition
    for j = 1:3
        for i = 1:length(testing_norm)
            w(j,i) = sum(sum((G*testing_norm(i,:)' - G*C(j,:)').^2));
        end
    end
    
    for i = 1:length(testing_norm)
        switch min(w(:,i))
            case w(1,i)
                IDX6(i) = cl(1);
            case w(2,i)
                IDX6(i) = cl(2);
            case w(3,i)
                IDX6(i) = cl(3);
        end
    end
    
    k3acc6(1,hh) = (1-nnz(IDX6 - testing_classes)/length(testing_classes))*100;
    
    %% clustering using cityblock (L1)
    
    [idx,C] = kmeans(training_norm,3);
    
    clus1 = find(idx == 1);
    clus2 = find(idx == 2);
    clus3 = find(idx == 3);
    
    % assign class labels to C
    bins = 0.5:1:3.5;
    
    class1 = find(testing_classes == 1);
    class2 = find(testing_classes == 2);
    class3 = find(testing_classes == 3);
    
    
    [hb,nb] = hist(training_classes(clus1),bins);
    [~, id] = max(hb);
    
    cl(1) = id;
    
    [hb,nb] = hist(training_classes(clus2),bins);
    [~, id] = max(hb);
    
    cl(2) = id;
    
    [hb,nb] = hist(training_classes(clus3),bins);
    [~, id] = max(hb);
    
    if plotfigs == 1
        featx = 6;
        featy = 7;
        
        figure(1)
        subplot(1,2,1)
        plot(training_norm(find(training_classes == 1),featx),training_norm(find(training_classes == 1),featy),'r.','MarkerSize',20)
        hold all
        plot(training_norm(find(training_classes == 2),featx),training_norm(find(training_classes == 2),featy),'b.','MarkerSize',20)
        plot(training_norm(find(training_classes == 3),featx),training_norm(find(training_classes == 3),featy),'g.','MarkerSize',20)
        plot(C(:,featx),C(:,featy),'kx',...
            'MarkerSize',25,'LineWidth',3)
        grid on
        grid minor
        set(gca,'linewidth',1.5,'fontsize',15)
        title('Original Data Partition','fontsize',30,'interpreter','latex')
        xlabel('Feature 6','fontsize',30,'interpreter','latex')
        ylabel('Feature 7','fontsize',30,'interpreter','latex')
        
        legend('Class 1','Class 2','Class 3','Centroids',...
            'Location','NW')
        axis tight
        subplot(1,2,2)
        plot(training_norm(find(idx==1),featx),training_norm(find(idx==1),featy),'r.','MarkerSize',20)
        hold all
        plot(training_norm(find(idx==2),featx),training_norm(find(idx==2),featy),'b.','MarkerSize',20)
        plot(training_norm(find(idx==3),featx),training_norm(find(idx==3),featy),'g.','MarkerSize',20)
        plot(C(:,featx),C(:,featy),'kx',...
            'MarkerSize',25,'LineWidth',3)
        grid on
        grid minor
        set(gca,'linewidth',1.5,'fontsize',15)
        title('3 Means Clustering','fontsize',30,'interpreter','latex')
        xlabel('Feature 6','fontsize',30,'interpreter','latex')
        ylabel('Feature 7','fontsize',30,'interpreter','latex')
        axis tight
        legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
            'Location','NW')
    end
    
    %
    %
    % L1 DIST
    %
    %
    
    IDX1 = knnsearch(C,testing_norm,'distance','cityblock');
    
    for i = 1:length(IDX1)
        switch IDX1(i)
            case 1
                IDX1(i) = cl(1);
            case 2
                IDX1(i) = cl(2);
            case 3
                IDX1(i) = cl(3);
        end
    end
    
    k3acc1(2,hh) = (1-nnz(IDX1' - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % L2 DIST
    %
    %
    
    IDX2 = knnsearch(C,testing_norm,'distance','euclidean');
    
    for i = 1:length(IDX2)
        switch IDX2(i)
            case 1
                IDX2(i) = cl(1);
            case 2
                IDX2(i) = cl(2);
            case 3
                IDX2(i) = cl(3);
        end
    end
    
    k3acc2(2,hh) = (1-nnz(IDX2' - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % CORR DIST
    %
    %
    
    IDX3 = knnsearch(C,testing_norm,'distance','correlation');
    
    for i = 1:length(IDX3)
        switch IDX3(i)
            case 1
                IDX3(i) = cl(1);
            case 2
                IDX3(i) = cl(2);
            case 3
                IDX3(i) = cl(3);
        end
    end
    
    k3acc3(2,hh) = (1-nnz(IDX3' - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % CHISQ DIST
    %
    %
    
    for i = 1:length(testing_norm)
        w = zeros(1,size(C,1));
        for j = 1:size(C,1)
            w(j) = 0.5*sum(((testing_norm(i,:) - C(j,:)).^2)./(testing_norm(i,:) + C(j,:)));
        end
        [minVal, IDX4(i)] = min(w);
    end
    
    for i = 1:length(IDX4)
        switch IDX4(i)
            case 1
                IDX4(i) = cl(1);
            case 2
                IDX4(i) = cl(2);
            case 3
                IDX4(i) = cl(3);
        end
    end
    
    k3acc4(2,hh) = (1-nnz(IDX4 - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % HIST DIST
    %
    %
    
    for i = 1:length(testing_norm)
        w = zeros(1,size(C,1));
        for j = 1:size(C,1)
            
            w(j) = sum(min(testing_norm(i,:), C(j,:)));
            
        end
        [maxVal, IDX5(i)] = max(w);
    end
    
    for i = 1:length(IDX5)
        switch IDX5(i)
            case 1
                IDX5(i) = cl(1);
            case 2
                IDX5(i) = cl(2);
            case 3
                IDX5(i) = cl(3);
        end
    end
    
    k3acc5(2,hh) = (1-nnz(IDX5 - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % MAHAL
    %
    %
    
    w = zeros(3,length(testing_norm));
    G = chol(cov_allNorm^-1); % Cholesky Decomposition
    for j = 1:3
        for i = 1:length(testing_norm)
            w(j,i) = sum(sum((G*testing_norm(i,:)' - G*C(j,:)').^2));
        end
    end

    for i = 1:length(testing_norm)
        switch min(w(:,i))
            case w(1,i)
                IDX6(i) = cl(1);
            case w(2,i)
                IDX6(i) = cl(2);
            case w(3,i)
                IDX6(i) = cl(3);
        end
    end
    
    k3acc6(2,hh) = (1-nnz(IDX6 - testing_classes)/length(testing_classes))*100;
    
    %% clustering using cosine
    
    [idx,C] = kmeans(training_norm,3,'distance','cosine');
    
    clus1 = find(idx == 1);
    clus2 = find(idx == 2);
    clus3 = find(idx == 3);
    
    % assign class labels to C
    bins = 0.5:1:3.5;
    
    class1 = find(testing_classes == 1);
    class2 = find(testing_classes == 2);
    class3 = find(testing_classes == 3);
    
    
    [hb,nb] = hist(training_classes(clus1),bins);
    [~, id] = max(hb);
    
    cl(1) = id;
    
    [hb,nb] = hist(training_classes(clus2),bins);
    [~, id] = max(hb);
    
    cl(2) = id;
    
    [hb,nb] = hist(training_classes(clus3),bins);
    [~, id] = max(hb);
    
    cl(3) = id;
    
    %
    %
    % L1 DIST
    %
    %
    
    IDX1 = knnsearch(C,testing_norm,'distance','cityblock');
    
    for i = 1:length(IDX1)
        switch IDX1(i)
            case 1
                IDX1(i) = cl(1);
            case 2
                IDX1(i) = cl(2);
            case 3
                IDX1(i) = cl(3);
        end
    end
    
    k3acc1(3,hh) = (1-nnz(IDX1' - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % L2 DIST
    %
    %
    
    IDX2 = knnsearch(C,testing_norm,'distance','euclidean');
    
    for i = 1:length(IDX2)
        switch IDX2(i)
            case 1
                IDX2(i) = cl(1);
            case 2
                IDX2(i) = cl(2);
            case 3
                IDX2(i) = cl(3);
        end
    end
    
    k3acc2(3,hh) = (1-nnz(IDX2' - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % CORR DIST
    %
    %
    
    IDX3 = knnsearch(C,testing_norm,'distance','correlation');
    
    for i = 1:length(IDX3)
        switch IDX3(i)
            case 1
                IDX3(i) = cl(1);
            case 2
                IDX3(i) = cl(2);
            case 3
                IDX3(i) = cl(3);
        end
    end
    
    k3acc3(3,hh) = (1-nnz(IDX3' - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % CHISQ DIST
    %
    %
    
    for i = 1:length(testing_norm)
        w = zeros(1,size(C,1));
        for j = 1:size(C,1)
            w(j) = 0.5*sum(((testing_norm(i,:) - C(j,:)).^2)./(testing_norm(i,:) + C(j,:)));
        end
        [minVal, IDX4(i)] = min(w);
    end
    
    for i = 1:length(IDX4)
        switch IDX4(i)
            case 1
                IDX4(i) = cl(1);
            case 2
                IDX4(i) = cl(2);
            case 3
                IDX4(i) = cl(3);
        end
    end
    
    k3acc4(3,hh) = (1-nnz(IDX4 - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % HIST DIST
    %
    %
    
    for i = 1:length(testing_norm)
        w = zeros(1,size(C,1));
        for j = 1:size(C,1)
            
            w(j) = sum(min(testing_norm(i,:), C(j,:)));
            
        end
        [maxVal, IDX5(i)] = max(w);
    end
    
    for i = 1:length(IDX5)
        switch IDX5(i)
            case 1
                IDX5(i) = cl(1);
            case 2
                IDX5(i) = cl(2);
            case 3
                IDX5(i) = cl(3);
        end
    end
    
    k3acc5(3,hh) = (1-nnz(IDX5 - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % MAHAL
    %
    %
    
    w = zeros(3,length(testing_norm));
    G = chol(cov_allNorm^-1); % Cholesky Decomposition
    for j = 1:3
        for i = 1:length(testing_norm)
            w(j,i) = sum(sum((G*testing_norm(i,:)' - G*C(j,:)').^2));
        end
    end
    
    for i = 1:length(testing_norm)
        switch min(w(:,i))
            case w(1,i)
                IDX6(i) = cl(1);
            case w(2,i)
                IDX6(i) = cl(2);
            case w(3,i)
                IDX6(i) = cl(3);
        end
    end
    
    k3acc6(3,hh) = (1-nnz(IDX6 - testing_classes)/length(testing_classes))*100;
    
    %% clustering using correlation
    
    [idx,C] = kmeans(training_norm,3,'distance','correlation');
    
    clus1 = find(idx == 1);
    clus2 = find(idx == 2);
    clus3 = find(idx == 3);
    
    % assign class labels to C
    bins = 0.5:1:3.5;
    
    class1 = find(testing_classes == 1);
    class2 = find(testing_classes == 2);
    class3 = find(testing_classes == 3);
    
    
    [hb,nb] = hist(training_classes(clus1),bins);
    [~, id] = max(hb);
    
    cl(1) = id;
    
    [hb,nb] = hist(training_classes(clus2),bins);
    [~, id] = max(hb);
    
    cl(2) = id;
    
    [hb,nb] = hist(training_classes(clus3),bins);
    [~, id] = max(hb);
    
    cl(3) = id;
    
    %
    %
    % L1 DIST
    %
    %
    
    IDX1 = knnsearch(C,testing_norm,'distance','cityblock');
    
    for i = 1:length(IDX1)
        switch IDX1(i)
            case 1
                IDX1(i) = cl(1);
            case 2
                IDX1(i) = cl(2);
            case 3
                IDX1(i) = cl(3);
        end
    end
    
    k3acc1(4,hh) = (1-nnz(IDX1' - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % L2 DIST
    %
    %
    
    IDX2 = knnsearch(C,testing_norm,'distance','euclidean');
    
    for i = 1:length(IDX2)
        switch IDX2(i)
            case 1
                IDX2(i) = cl(1);
            case 2
                IDX2(i) = cl(2);
            case 3
                IDX2(i) = cl(3);
        end
    end
    
    k3acc2(4,hh) = (1-nnz(IDX2' - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % CORR DIST
    %
    %
    
    IDX3 = knnsearch(C,testing_norm,'distance','correlation');
    
    for i = 1:length(IDX3)
        switch IDX3(i)
            case 1
                IDX3(i) = cl(1);
            case 2
                IDX3(i) = cl(2);
            case 3
                IDX3(i) = cl(3);
        end
    end
    
    k3acc3(4,hh) = (1-nnz(IDX3' - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % CHISQ DIST
    %
    %
    
    for i = 1:length(testing_norm)
        w = zeros(1,size(C,1));
        for j = 1:size(C,1)
            w(j) = 0.5*sum(((testing_norm(i,:) - C(j,:)).^2)./(testing_norm(i,:) + C(j,:)));
        end
        [minVal, IDX4(i)] = min(w);
    end
    
    for i = 1:length(IDX4)
        switch IDX4(i)
            case 1
                IDX4(i) = cl(1);
            case 2
                IDX4(i) = cl(2);
            case 3
                IDX4(i) = cl(3);
        end
    end
    
    k3acc4(4,hh) = (1-nnz(IDX4 - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % HIST DIST
    %
    %
    
    for i = 1:length(testing_norm)
        w = zeros(1,size(C,1));
        for j = 1:size(C,1)
            
            w(j) = sum((min(testing_norm(i,:), C(j,:))).^2);
            
        end
        [maxVal, IDX5(i)] = max(w);
    end
    
    for i = 1:length(IDX5)
        switch IDX5(i)
            case 1
                IDX5(i) = cl(1);
            case 2
                IDX5(i) = cl(2);
            case 3
                IDX5(i) = cl(3);
        end
    end
    
    k3acc5(4,hh) = (1-nnz(IDX5 - testing_classes)/length(testing_classes))*100;
    
    %
    %
    % MAHAL
    %
    %
    
    w = zeros(3,length(testing_norm));
    G = chol(cov_allNorm^-1); % Cholesky Decomposition
    for j = 1:3
        for i = 1:length(testing_norm)
            w(j,i) = sum(sum((G*testing_norm(i,:)' - G*C(j,:)').^2));
        end
    end
    
    for i = 1:length(testing_norm)
        switch min(w(:,i))
            case w(1,i)
                IDX6(i) = cl(1);
            case w(2,i)
                IDX6(i) = cl(2);
            case w(3,i)
                IDX6(i) = cl(3);
        end
    end
    
    k3acc6(4,hh) = (1-nnz(IDX6 - testing_classes)/length(testing_classes))*100;
    
end

% k3accX -> Rows: L1, L2, COS, CORR / Cols: Number of runs
% k3accX -> X=1 L1, X=2 L2, X=3 Corr, X=4 ChiSq, X=5 Hist



% statistical properties (mean vaue for each, max value for each, min
% values for each)

mean_acc(1,:) = mean(k3acc1');
mean_acc(2,:) = mean(k3acc2');
mean_acc(3,:) = mean(k3acc3');
mean_acc(4,:) = mean(k3acc4');
mean_acc(5,:) = mean(k3acc5');
mean_acc(6,:) = mean(k3acc6');

max_acc(1,:) = max(k3acc1');
max_acc(2,:) = max(k3acc2');
max_acc(3,:) = max(k3acc3');
max_acc(4,:) = max(k3acc4');
max_acc(5,:) = max(k3acc5');
max_acc(6,:) = max(k3acc6');

min_acc(1,:) = min(k3acc1');
min_acc(2,:) = min(k3acc2');
min_acc(3,:) = min(k3acc3');
min_acc(4,:) = min(k3acc4');
min_acc(5,:) = min(k3acc5');
min_acc(6,:) = min(k3acc6');