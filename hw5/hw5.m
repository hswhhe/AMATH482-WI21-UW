clear; close all; clc

%% Read video
v = VideoReader('monte_carlo_low_Trim.mp4');
frames = read(v,[1 Inf]);
[height,width, rgb, num_frames] = size(frames);
time = v.CurrentTime;
dt = time / (num_frames - 1);
t = 0:dt:time;

%% Turn video into grayscale, make DMD Matrices
pixel = height*width;
X = zeros(pixel, num_frames);
for j=1:num_frames
    X_frame = frames(:,:,:,j);
    X_frame = im2double(rgb2gray(X_frame));
    X_frame_rs = reshape(X_frame,pixel,1);
    X(:,j) = X_frame_rs;
end
   X_play=frames(:,:,:,40);
   X_play = rgb2gray(X_play);
    %X_play = abs(X_play);
    imshow(X_play); drawnow


%%
X1 = X(:,1:end-1);
X2 = X(:,2:end);

%% SVD of X1

[U, Sigma, V] = svd(X1,'econ');
%% Low-rank approximation
mode = 1;
U_low = U(:,1:mode);
Sigma_low = Sigma(1:mode,1:mode);
V_low = V(:,1:mode);
diag_sigma = diag(Sigma);
plot(diag_sigma(1:50),'ko','Linewidth',2);
title('Singular value spectrum(first 50 values) of Ski drop');
ylabel('\sigma')
%% Computation of ~S
S = U_low'*X2*V_low*diag(1./diag(Sigma_low));
[eV, D] = eig(S); % compute eigenvalues + eigenvectors
mu = diag(D); % extract eigenvalues
%% phi & omega
omega = log(mu)/dt;
Phi = U_low*eV;

%% Create DMD Solution

y0 = Phi\X1(:,1); % pseudoinverse to get initial conditions

u_modes = zeros(length(y0),num_frames);
for iter = 1:num_frames
   u_modes(:,iter) = y0.*exp(omega*t(iter)); 
end
X_dmd_low = Phi*u_modes;

%% Plotting Eigenvalues (omega)

% make axis lines
line = -15:15;

plot(zeros(length(line),1),line,'k','Linewidth',2) % imaginary axis
hold on
plot(line,zeros(length(line),1),'k','Linewidth',2) % real axis
plot(real(omega)*dt,imag(omega)*dt,'r.','Markersize',15)
xlabel('Re(\omega)')
ylabel('Im(\omega)')
set(gca,'FontSize',16,'Xlim',[-1.5 0.5],'Ylim',[-3 3])

%% video of low-rank DMD;
X_dmd_low_video = reshape(X_dmd_low,height,width,num_frames);
for j=1:num_frames
    X_play=X_dmd_low_video(:,:,j);
    %X_play = abs(X_play);
    imshow(X_play); drawnow
end

%% Sparse DMD
x_dmd_low_abs = abs(X_dmd_low);
X_sparse = X - x_dmd_low_abs;

%% Foreground
X_fore_video = reshape(X_sparse,height,width,num_frames);
for j=1:num_frames
    X_play=X_fore_video(:,:,j);
    X_play = X_play + 0.4;
    %X_play = abs(X_play);
    imshow(X_play); drawnow
end


%% find residual matrix R
X_sparse_negative_logical = X_sparse < 0;
R = X_sparse.*X_sparse_negative_logical;
%% Find X_sparse and X_lowRank with R;
X_sparse_R = X_sparse - R ;
X_low_R = R + abs(X_dmd_low);
%% foreground with R
X_sparse_R_video = reshape(X_sparse_R,height,width,num_frames);
for j=1:num_frames
    X_play=X_sparse_R_video(:,:,j);
%     X_play = abs(X_play);
    imshow(X_play); drawnow
end

%% background with R
X_low_R_video = reshape(X_low_R,height,width,num_frames);
for j=1:num_frames
    X_play=X_low_R_video(:,:,j);
%     X_play = abs(X_play);
    imshow(X_play); drawnow
end

