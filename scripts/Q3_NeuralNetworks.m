%% Neural Netoworks. Code Stolen from Matlab since they own the wine_data and use it to exmplain NN.

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

% load norm data
% load wine_separatedData.mat

[x,t] = wine_dataset;

idcs = randperm(178);

tr = 118;
tes = 40;
val = 178-118-40;
training_raw = x(:,idcs(1:tr));
training_cl = t(:,idcs(1:tr));
testing_raw = x(:,idcs(tr+1:tr+tes));
testing_cl = t(:,idcs(tr+1:tr+tes));

for i = 1:length(testing_raw)
   [~,testing_classes(i)] = max(testing_cl(:,i));
end
%% Vary hidden layer size

% 
% training_cl = zeros(3,length(training_classes));
% for i = 1:length(training_classes)
%     training_cl(training_classes(i),i) = 1;
% end
% 
maxNum = 15;

for p = 1:10
    for k = 1:maxNum
        net.divideParam.trainRatio = 100/100;     
        net.divideParam.valRatio = 0/100;      
        net.divideParam.testRatio = 0/100;
        net = patternnet(k);
        net = train(net,training_raw,training_cl);
        % view(net)
        y = net(testing_raw);
        
        for i = 1:length(testing_raw)
            [~, NeurClass(i)] = max(y(:,i));
        end
        
        acc1(p,k) = (length(testing_raw)-nnz(NeurClass - testing_classes))*100/length(testing_raw);
        
        clear net y NeurClass
    end
end

figure(1)
plot(1:maxNum,mean(acc1),'linewidth',4)
grid on
grid minor
set(gca,'fontsize',15,'linewidth',1.5)
xlabel('Hidden Layer Size','fontsize',30,'interpreter','latex')
ylabel('Accuracy [\%]','fontsize',30,'interpreter','latex')
title('Effect of Hidden Layer Size on Performance','fontsize',30,'interpreter','latex')

%% Vary number of hidden layers
% 
% 
% training_cl = zeros(3,length(training_classes));
% for i = 1:length(training_classes)
%     training_cl(training_classes(i),i) = 1;
% end

hidSize = 20;
maxHid = 20;

for p = 1:3
  for s = 1:hidSize  
    for k = 1:maxHid
        
        
        net = patternnet(s*ones(1,k));
        net.divideParam.trainRatio = 100/100;     
        net.divideParam.valRatio = 0/100;      
        net.divideParam.testRatio = 0/100; 
        net = train(net,training_raw,training_cl);
        % view(net)
        y = net(testing_raw);
        
        for i = 1:length(testing_raw)
            [~, NeurClass(i)] = max(y(:,i));
        end
        
        acc2(s,k,p) = (length(testing_raw)-nnz(NeurClass - testing_classes))*100/length(testing_raw);
        
        clear net y NeurClass
    end
  end
end


% plot(1:maxHid,mean(acc),'linewidth',4)
% grid on
% grid minor
% set(gca,'fontsize',15,'linewidth',1.5)
% shading interp
% xlabel('Number of Hidden Layers','fontsize',30,'interpreter','latex')
% %ylabel('Hidden Layers Size','fontsize',30,'interpreter','latex')
% ylabel('Accuracy [\%]','fontsize',30,'interpreter','latex')
% title('Effect of Number and Size of Hidden Layer on Performance','fontsize',30,'interpreter','latex')
% % 
surf(1:maxHid,1:hidSize,mean(acc2,3))
grid on
grid minor
set(gca,'fontsize',15,'linewidth',1.5)
shading interp
xlabel('Number of Hidden Layers','fontsize',30,'interpreter','latex')
ylabel('Hidden Layers Size','fontsize',30,'interpreter','latex')
zlabel('Accuracy [\%]','fontsize',30,'interpreter','latex')
title('Effect of Number and Size of Hidden Layer on Performance','fontsize',30,'interpreter','latex')

