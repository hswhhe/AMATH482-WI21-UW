% Clean workspace
clear all; close all; clc

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

%Plot the original noised signal
ave = zeros(n,n,n);
for j=1:49
Un(:,:,:)=reshape(subdata(:,j),n,n,n);
M = max(abs(Un),[],'all');
Unt = fftn(Un);
ave = ave+Unt;
close all;
isosurface(X,Y,Z,abs(Un)/M,0.7)
axis([-20 20 -20 20 -20 20]), grid on, drawnow
pause(0.1);
end

%%
%Averaging the spectrum 
aved = abs(fftshift(ave))/49;
M = max(abs(aved),[],'all');
close all
isosurface(Kx,Ky,Kz,abs(aved)/M,0.7)
axis([-20 20 -20 20 -20 20]), grid on, drawnow
K0_x = 5;
K0_y = -7;
K0_z = 2;
tau = 0.2;
%%
%Define the filter
filter_x = exp(-tau*(Kx - K0_x).^2); 
filter_y = exp(-tau*(Ky - K0_y).^2); 
filter_z = exp(-tau*(Kz - K0_z).^2); 
filter = filter_x.*filter_y.*filter_z;

%%
%Find the path
close all;
coord = zeros(49,3); %giving the coordinates of positions at each time
for k = 1:49
    Un(:,:,:)=reshape(subdata(:,k),n,n,n);
    Unt = fftn(Un);
    Untf = fftshift(Unt).*filter;
    %Untf = ifftshift(Untf);
    Unf = ifftn(Untf);
    Unf = Unf/max(abs(Unf(:)));

    [M_Unf,I] = max(abs(Unf(:)));
    [x_coord,y_coord,z_coord] = ind2sub(size(Unf),I);
    coord(k,1) = X(x_coord,y_coord,z_coord);
    coord(k,2) = Y(x_coord,y_coord,z_coord);
    coord(k,3) = Z(x_coord,y_coord,z_coord);
end
%%
%Plot the path
figure(3)
plot3(coord(:,1),coord(:,2),coord(:,3));
hold on;
plot3(coord(49,1),coord(49,2),coord(49,3),'ro');%final position
legend('Path of the submarine','Final location');

%%
%Table of final x&y coordinates of the submarine
T = [coord(49,1),coord(49,2),coord(49,3)]

