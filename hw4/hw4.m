%% clear
clear;close all; clc

%% load data
[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte'); %Traning data
[images2,labels2] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte'); %Test data

%% Reshape & Reduced SVD
data =im2double(reshape(images,784,60000));
data2 = im2double(reshape(images2,784,10000));
[m,n]=size(data); % compute data size
mn=mean(data,2); % compute mean for each row
data=data-repmat(mn,1,n); % subtract mean
data2 = data2 - repmat(mn,1,10000);
[U,S,V] = svd(data,'econ'); % reduced SVD
%% Singular value spectrum
plot(diag(S),'ko','Linewidth',2);
set(gca,'Fontsize',16,'Xlim',[0 100]);
title('Singular value spectrum(first 100 values)');
ylabel('\sigma')
%% energy
% plot(cumsum(diag(S).^2)/sum(diag(S).^2),'ko','Linewidth',2);
% set(gca,'Fontsize',16,'Xlim',[0 200]);
% title('Cumulative energy(first 200 values)');
%% low-rank approximation
% k = 20;
% data_approx = U(:,1:k)*S(1:k,1:k)*V(:,1:k)'; %'// rank-15 approximation 
% image_approx = reshape(data_approx(:,5),28,28);
% imshow(image_approx);

%% 3d plot of three selected principal components
Y_3mode = U(:,[2,4,5])'*data;

for k = 0:9
    data_3mode = Y_3mode(:,find(labels == k));
    plot3(data_3mode(1,:),data_3mode(2,:),data_3mode(3,:),'o');hold on 
end
xlabel('2nd pc'); ylabel('4th pc'); zlabel('5th pc')
legend('0','1','2','3','4','5','6','7','8','9');

%% Create low-dimension matrix for each digit in training/test set in PC
%Traning set
feature = 20;
Y = U(:,1:feature)'*data;
data_0 = Y(:,find(labels == 0));
data_1 = Y(:,find(labels == 1));
data_2 = Y(:,find(labels == 2));
data_3 = Y(:,find(labels == 3));
data_4 = Y(:,find(labels == 4));
data_5 = Y(:,find(labels == 5));
data_6 = Y(:,find(labels == 6));
data_7 = Y(:,find(labels == 7));
data_8 = Y(:,find(labels == 8));
data_9 = Y(:,find(labels == 9));
data_cell= {data_0,data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9};

%Test set
Y2 = U(:,1:feature)'*data2;
data2_0 = Y2(:,find(labels2 == 0));
data2_1 = Y2(:,find(labels2 == 1));
data2_2 = Y2(:,find(labels2 == 2));
data2_3 = Y2(:,find(labels2 == 3));
data2_4 = Y2(:,find(labels2 == 4));
data2_5 = Y2(:,find(labels2 == 5));
data2_6 = Y2(:,find(labels2 == 6));
data2_7 = Y2(:,find(labels2 == 7));
data2_8 = Y2(:,find(labels2 == 8));
data2_9 = Y2(:,find(labels2 == 9));
data2_cell= {data2_0,data2_1,data2_2,data2_3,data2_4,data2_5,data2_6,data2_7,data2_8,data2_9};

%% the two digits we choose in traning set
digit_1 = 1;
digit_2 = 0;
%the data of the chosen 2 digits in training set.
var_1 = data_cell{digit_1+1};
var_2 = data_cell{digit_2+1}; 

var2_1 = data2_cell{digit_1+1};
var2_2 = data2_cell{digit_2+1}; 


%% Calculate scatter matrices
m1 = mean(var_1,2);
m2 = mean(var_2,2);

Sw = 0; % within class variances
for k = 1:size(var_1,2);
    Sw = Sw + (var_1(:,k) - m1)*(var_1(:,k) - m1)';
end
for k = 1:size(var_2,2);
   Sw =  Sw + (var_2(:,k) - m2)*(var_2(:,k) - m2)';
end

Sb = (m1-m2)*(m1-m2)'; % between class
%% Find the best projection line
[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

%% Project onto w
v_var1 = w'*var_1;
v_var2 = w'*var_2;

%for test data
v2_var1 = w'*var2_1;
v2_var2 = w'*var2_2;


%% Make var2 above the threshold

if mean(v_var1) > mean(v_var2)
    w = -w;
    v_var1 = -v_var1;
    v_var2 = -v_var2;
    v2_var1 = -v2_var1;
    v2_var2 = -v2_var2;
    
end

%% Find the threshold value

sort1 = sort(v_var1);
sort2 = sort(v_var2);

t1 = length(sort1);
t2 = 1;
while sort1(t1) > sort2(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sort1(t1) + sort2(t2))/2;

%% Plot histogram of results
hold off
subplot(1,2,1)
histogram(sort1,30); hold on, plot([threshold threshold], [0 800],'r')
set(gca,'Xlim',[-3 4],'Ylim',[0 800],'Fontsize',14)
title('digit 5')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 800],'r')
set(gca,'Xlim',[-3 4],'Ylim',[0 800],'Fontsize',14)
title('digit 3')
%% Test accuracy between 2 digits in training data.
ResVec_1 = (v_var1 < threshold);
success_1 = sum(ResVec_1);
err_1 = length(v_var1) - success_1;
sucRate_1 = success_1 / length(v_var1)

ResVec_2 = (v_var2 > threshold);
success_2 = sum(ResVec_2);
err_2 = length(v_var2) - success_2;
sucRate_2 = success_2 / length(v_var2)

% Accuracy between 2 digits in test set

ResVec2_1 = (v2_var1 < threshold);
success2_1 = sum(ResVec2_1);
err2_1 = length(v2_var1) - success2_1;
sucRate2_1 = success2_1 / length(v2_var1)

ResVec2_2 = (v2_var2 > threshold);
success2_2 = sum(ResVec2_2);
err2_2 = length(v2_var2) - success2_2;
sucRate2_2 = success2_2 / length(v2_var2);

%% SVM
data_00 = var_1./max(var_1(:));
data_11 = var_2./max(var_2(:));
ones1 = ones(length(sort1),1);
ones2 = ones(length(sort2),1);
label0 = digit_1.*ones1;
label1 = digit_2.*ones2;
xtrain = [data_00 data_11];
xtrain = xtrain.';
xlabel = [label0;label1];
Mdl = fitcecoc(xtrain,xlabel);
error = resubLoss(Mdl) 

%% SVM test
xtest = Y2./max(Y2(:));
xtest = xtest';
test_labels = predict(Mdl,xtest);
CVSVMModel = crossval(Mdl);
%%
classLoss = kfoldLoss(CVSVMModel)
%% fitctree
MdlDefault = fitctree(xtrain,xlabel);

%% fitctree error
result = predict(MdlDefault, xtest);


%%
xtrain = U(:,feature)'*vec;
xtrain = xtrain / max(xtrain(:));
xtrain = xtrain';
xlabel = labels;
Mdl = fitcecoc(xtrain,xlabel);
error = resubLoss(Mdl);


