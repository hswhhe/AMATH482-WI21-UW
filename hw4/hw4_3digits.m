% LDA for three digits
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
plot(cumsum(diag(S).^2)/sum(diag(S).^2),'ko','Linewidth',2);
set(gca,'Fontsize',16,'Xlim',[0 200]);
title('Cumulative energy(first 200 values)');
%% low-rank approximation
k = 15;
data_approx = U(:,1:k)*S(1:k,1:k)*V(:,1:k)'; %'// rank-15 approximation 
aaa = reshape(data_approx(:,5),28,28);
imshow(aaa);

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
feature = 100;
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
data_cell= {data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9};

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
data2_cell= {data2_1,data2_2,data2_3,data2_4,data2_5,data2_6,data2_7,data2_8,data2_9};

%% the three digits we choose in traning set
digit_1 = 4;
digit_2 = 0;
digit_3 = 3;
%the data of the chosen 2 digits in training set.
var_1 = data_cell{digit_1+1};
var_2 = data_cell{digit_2+1}; 
var_3 = data_cell{digit_3+1}; 

var2_1 = data2_cell{digit_1+1};
var2_2 = data2_cell{digit_2+1}; 
var2_3 = data2_cell{digit_3+1}; 



%% Calculate scatter matrices
m1 = mean(var_1,2);
m2 = mean(var_2,2);
m3 = mean(var_3,2);
m_all = (m1+m2+m3)./3;

Sw = 0; % within class variances
for k = 1:size(var_1,2);
    Sw = Sw + (var_1(:,k) - m1)*(var_1(:,k) - m1)';
end
for k = 1:size(var_2,2);
   Sw =  Sw + (var_2(:,k) - m2)*(var_2(:,k) - m2)';
end
for k = 1:size(var_3,2);
   Sw =  Sw + (var_3(:,k) - m3)*(var_3(:,k) - m3)';
end

Sb = (m1-m_all)*(m1-m_all)'+(m2-m_all)*(m2-m_all)'+(m3-m_all)*(m3-m_all)'; % between class
%% Find the best projection line
[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

%% Project onto w
v_var1 = w'*var_1;
v_var2 = w'*var_2;
v_var3 = w'*var_3;

%for test data
v2_var1 = w'*var2_1;
v2_var2 = w'*var2_2;
v2_var3 = w'*var2_3;



%% compare the mean
a1 = mean(v_var1);
a2 = mean(v_var2);a3 = mean(v_var3);


%% Find the threshold value

sort1 = sort(v_var1);
sort2 = sort(v_var2);
sort3 = sort(v_var3);

t3 = length(sort3);
t1 = 1;
while sort3(t3) > sort1(t1)
    t3 = t3 - 1;
    t1 = t1 + 1;
end
threshold1 = (sort3(t3) + sort1(t1))/2;

t1 = length(sort1);
t2 = 1;
while sort1(t1) > sort2(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold2 = (sort2(t2) + sort1(t1))/2;
%% Plot histogram of results
hold off
subplot(1,3,1)
histogram(sort1,30); hold on, plot([threshold1 threshold1], [0 1800],'r')
plot([threshold2 threshold2], [0 1800],'r')
set(gca,'Xlim',[-3 4],'Ylim',[0,1800],'Fontsize',14)
title('digit 4')
subplot(1,3,2)
histogram(sort2,30); hold on, 
plot([threshold1 threshold1], [0 1800],'r')
plot([threshold2 threshold2], [0 1800],'r')
set(gca,'Xlim',[-3 4],'Ylim',[0,1800],'Fontsize',14)
title('digit 0')
subplot(1,3,3)
histogram(sort3,30); hold on, 
plot([threshold1 threshold1], [0 1800],'r')
plot([threshold2 threshold2], [0 1800],'r')
set(gca,'Xlim',[-3 4],'Ylim',[0,1800],'Fontsize',14)
title('digit 3')
%% Test accuracy between 3 digits in training data.
ResVec_1 = (v_var1 < threshold2 & v_var1 >threshold1);
success_1 = sum(ResVec_1);
err_1 = length(v_var1) - success_1;
sucRate_1 = success_1 / length(v_var1)

ResVec_2 = (v_var2 > threshold2);
success_2 = sum(ResVec_2);
err_2 = length(v_var2) - success_2;
sucRate_2 = success_2 / length(v_var2)

ResVec_3 = (v_var3 < threshold1);
success_3 = sum(ResVec_3);
err_3 = length(v_var3) - success_3;
sucRate_3 = success_3 / length(v_var3)
%% Accuracy between 3 digits in test set
ResVec2_1 = (v2_var1 < threshold2 &v2_var1 > threshold1);
success2_1 = sum(ResVec2_1);
err2_1 = length(v2_var1) - success2_1;
sucRate2_1 = success2_1 / length(v2_var1)

ResVec2_2 = (v2_var2 > threshold2);
success2_2 = sum(ResVec2_2);
err2_2 = length(v2_var2) - success2_2;
sucRate2_2 = success2_2 / length(v2_var2)

ResVec2_3 = (v2_var3 < threshold1);
success2_3 = sum(ResVec2_3);
err2_3 = length(v2_var3) - success2_3;
sucRate2_3 = success2_3 / length(v2_var3)
