clear; close all; clc;



%% First camera
%Load Movie
load('cam1_3.mat');
[height,width, rgb, num_frames] = size(vidFrames1_3);
%Watch Movie
x1_coord = zeros(1,num_frames);
y1_coord = zeros(1,num_frames);
for j=1:num_frames
%Turn the fram into grayscale
X=rgb2gray(vidFrames1_3(:,:,:,j));
X=im2double(X);
brightest = 0.971;
%Returns all the positions where the frame is the brightest
[y,x] = find(X > brightest);
%Returns the positions of the birght spot;
%Limit y value to be larger than 300;
y_position = y>300;
y_max = max(y(y_position));
y1_coord(j) = y_max;
%Limit x value to be between 400 and 280;
 x_high = x<400;
 x = x(x_high);
 x_low = x > 280;
 x_min = min(x(x_low));
 x1_coord(j) = x_min;
% imshow(X); drawnow
end
plot(1:num_frames, y1_coord);

%% Second camera

 load('cam2_3.mat');
 [height2,width2, rgb2, num_frames2] = size(vidFrames2_3);
%Watch Movie
x2_coord = zeros(1,num_frames2);
y2_coord = zeros(1,num_frames2);
for j=1:num_frames2
%Turn the fram into grayscale
X=rgb2gray(vidFrames2_3(:,:,:,j));
X=im2double(X);
brightest = 0.988;
%Returns all the positions where the frame is the brightest
[y,x] = find(X > brightest);
%Returns the positions of the birght spot;
%Limit y value to be between 200 and 380;
 y_low = y>200;
 y = y(y_low);
 y_high = y<380;
 y_max = max(y(y_high));
 y2_coord(j) = y_max;
 
 %Limit x value to be between 180 and 380
 x_high = x<380;
 x = x(x_high);
 x_low = x > 180;
 x_min = min(x(x_low));
 x2_coord(j) = x_min;
 %imshow(X); drawnow
end
plot(1:num_frames2, y2_coord);




%% Third camera

load('cam3_3.mat');
[height2,width2, rgb2, num_frames3] = size(vidFrames3_3);
%Watch Movie
x3_coord = zeros(1,num_frames3);
y3_coord = zeros(1,num_frames3);
for j=1:num_frames3
%Turn the fram into grayscale
X=rgb2gray(vidFrames3_3(:,:,:,j));
X=im2double(X);
brightest = 0.9;
%Returns all the positions where the frame is the brightest
[y,x] = find(X > brightest);
%Returns the positions of the birght spot;
%Limit x value to be between 279 and 400 a
 x_low = x>270;
 x = x(x_low);
 x_high = x<400;
 x_max = min(x(x_high));
 x3_coord(j) = x_max;

 %Limit y value to be between 180 and 300
 y_low = y>180;
 y = y(y_low);
 y_high = y<300;
 y_max = min(y(y_high));
 y3_coord(j) = y_max;
%imshow(X); drawnow
end
plot(1:num_frames3, x3_coord);

%% Align all the data from different cameras in time

x3_min = 291;
x3_min_position = find(x3_coord == x3_min);
x3_aligned = x3_coord(x3_min_position(1):end);
y3_aligned = y3_coord(x3_min_position(1):end);

num_newFrames = length(y3_aligned);
y2_min = 277;
y2_min_position = find(y2_coord == y2_min);
y2_aligned = y2_coord(y2_min_position(1):y2_min_position(1)+length(x3_aligned)-1);
x2_aligned = x2_coord(y2_min_position(1):y2_min_position(1)+length(x3_aligned)-1);

y1_min = 352;
y1_min_position = find(y1_coord == y1_min);
y1_aligned = y1_coord(y1_min_position(1):y1_min_position(1)+length(x3_aligned)-1);
x1_aligned = x1_coord(y1_min_position(1):y1_min_position(1)+length(x3_aligned)-1);

plot(1:num_newFrames,y1_aligned,1:num_newFrames,y2_aligned,1:num_newFrames,x3_aligned);
%% Reduce redundancy(PCA)
%Put all the data into a single matrix;
X = [x1_aligned;y1_aligned;x2_aligned;y2_aligned;x3_aligned;y3_aligned];
%Covariance of X

[m,n]=size(X); % compute data size
mn=mean(X,2); % compute mean for each row
X=X-repmat(mn,1,n); % subtract mean

cov_x = 1/(n-1) * X* X';


[u,s,v]=svd(X/sqrt(n-1),'econ'); % perform the SVD
lambda=diag(s).^2; % produce diagonal variances
Y=u'*X; % produce the principal components projection
cov_y = 1/(n-1) * Y* Y'; %Covariance matrix of Y;

%% Plot:Third case; original basis
figure(1)
sgtitle('Can Position in the Horizontal disp Case');
subplot(3,2,1);
plot(1:num_newFrames,X(1,:));
ylim([-100,100]);
xlabel('time frame')
ylabel('x disp(cam1)');
hold on
subplot(3,2,2);
plot(1:num_newFrames,X(2,:));
ylim([-100,100]);
xlabel('time frame')
ylabel('y disp(cam1)');
subplot(3,2,3);
plot(1:num_newFrames,X(3,:));
ylim([-100,100]);
xlabel('time frame')
ylabel('x disp(cam2)');
subplot(3,2,4);
plot(1:num_newFrames,X(4,:));
ylim([-100,100]);
xlabel('time frame')
ylabel('y disp(cam2)');
subplot(3,2,5);
plot(1:num_newFrames,X(5,:));
ylim([-100,100]);
xlabel('time frame')
ylabel('x disp(cam3)');
subplot(3,2,6);
plot(1:num_newFrames,X(6,:));
ylim([-100,100]);
xlabel('time frame')
ylabel('y disp(cam3)');

%% Graph of Principle components
hold off

plot(1:num_newFrames,Y(1,:));
hold on
plot(1:num_newFrames,Y(2,:));
plot(1:num_newFrames,Y(3,:));
plot(1:num_newFrames,Y(4,:));
plot(1:num_newFrames,Y(5,:));
title('Can Position in the Basis of the Principal Components(Case 3) ');
legend('1st pc','2nd pc', '3rd pc','4th pc','5th pc','6th pc');


%% Energy
sgtitle('Energy for Horizontal disp Case');
subplot(1,2,1)
sig = diag(s);
energy1 = sig(1)^2/sum(sig.^2)
energy2 = sum(sig(1:2).^2)/sum(sig.^2)
energy3 = sum(sig(1:3).^2)/sum(sig.^2)

hold off
plot(sig,'ko','Linewidth',2)
title('Energies vs.Singular Values');
ylabel('\sigma')
subplot(1,2,2)
plot(cumsum(sig.^2)/sum(sig.^2),'ko','Linewidth',2);
axis([0 6 10^-(18) 1])
title('Cumulative Energy');
