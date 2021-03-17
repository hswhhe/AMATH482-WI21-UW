clear; close all; clc;



%% First camera
%Load Movie
load('cam1_4.mat');
[height,width, rgb, num_frames] = size(vidFrames1_4);
%Watch Movie
x1_coord = zeros(1,num_frames);
y1_coord = zeros(1,num_frames);
for j=1:num_frames
%Turn the fram into grayscale
X=rgb2gray(vidFrames1_4(:,:,:,j));
X=im2double(X);
brightest = 0.92;
%Returns all the positions where the frame is the brightest
[y,x] = find(X > brightest);
x_position = x>320& x <400;
x = x(x_position);
y = y(x_position);
%Returns the positions of the birght spot;
%Limit y value to be larger than 250 and x value to be smaller than 400 ;

y_position = y>250;
y_max = max(y(y_position));

y1_coord(j) = y_max;
 x_high = x<400;
 x = x(x_high);
 x_min = min(x);
 x1_coord(j) = x_min;
 %imshow(X); drawnow
end
plot(1:num_frames, y1_coord);

%% Second camera

 load('cam2_4.mat');
 [height2,width2, rgb2, num_frames2] = size(vidFrames2_4);
%Watch Movie
x2_coord = zeros(1,num_frames2);
y2_coord = zeros(1,num_frames2);
for j=1:num_frames2
%Turn the fram into grayscale
X=rgb2gray(vidFrames2_4(:,:,:,j));
X=im2double(X);
brightest = 0.97;
%Returns all the positions where the frame is the brightest
[y,x] = find(X > brightest);
%Only looking at the part where x is between 246 and 360;
x_position = x>230 & x < 360;
x = x(x_position);
y = y(x_position);
%Returns the positions of the birght spot;
%Limit y value to be between 180 and 400;

 y_low = y>180;
 y = y(y_low);
 y_high = y<400;
 y_max = max(y(y_high));
 y2_coord(j) = y_max;

 x_min = min(x);
 x2_coord(j) = x_min;
 %imshow(X); drawnow
end
figure();
  plot(1:num_frames2, y2_coord);
% X=rgb2gray(vidFrames2_4(:,:,:,56));
% imshow(X); drawnow



%% Third camera

load('cam3_4.mat');
[height2,width2, rgb2, num_frames3] = size(vidFrames3_4);
%Watch Movie
x3_coord = zeros(1,num_frames3);
y3_coord = zeros(1,num_frames3);
for j=1:num_frames3
%Turn the fram into grayscale
X=rgb2gray(vidFrames3_4(:,:,:,j));
X=im2double(X);
brightest = 0.89;
%Returns all the positions where the frame is the brightest
%Only looking at the part where y is between 140 and 240;
[y,x] = find(X > brightest);
y_position = y>140 & y < 240;
x = x(y_position);
%Returns the positions of the birght spot;
%Limit x value to be between 200 and 600
 x_low = x>200;
 x = x(x_low);
 x_high = x<600;
 x_max = max(x(x_high));
 x3_coord(j) = x_max;
 
 %Limit y value to be between 140 and 240

x_position = x > 200 & x < 600;
y = y(x_position);
 y_low = y>140;
 y = y(y_low);
 y_high = y<240;
 y_max = min(y(y_high));
 y3_coord(j) = y_max;
%imshow(X); drawnow
end
plot(1:num_frames3, x3_coord);
% 
% X=rgb2gray(vidFrames3_4(:,:,:,7));
% imshow(X); drawnow

%% Align all the data from different cameras in time

y1_min_position = y1_coord == 315;
y1_small = y1_coord((y1_min_position));
y1_min_position = find(y1_coord == y1_small(1));
y1_aligned = y1_coord(y1_min_position(1):end);
x1_aligned = x1_coord(y1_min_position(1):end);

num_newFrames = length(y1_aligned);
y2_min = 219;
y2_min_position = find(y2_coord == y2_min);
y2_aligned = y2_coord(y2_min_position(1):y2_min_position(1)+length(y1_aligned)-1);
x2_aligned = x2_coord(y2_min_position(1):y2_min_position(1)+length(y1_aligned)-1);

x3_min = 363;
x3_min_position = find(x3_coord == x3_min);
x3_aligned = x3_coord(x3_min_position(1):x3_min_position(1)+length(y1_aligned)-1);
y3_aligned = y3_coord(x3_min_position(1):x3_min_position(1)+length(y1_aligned)-1);

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

%% Plot:Horizontal Displacement and Rotation; original basis
figure(1)
sgtitle('Can Position in the Horizontal disp & Rotation Case');
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
title('Can Position in the Basis of the Principal Components(Case 4) ');
legend('1st pc','2nd pc', '3rd pc','4th pc','5th pc');


%% Energy
sgtitle('Energy for Horizontal disp &Rotation Case');
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
