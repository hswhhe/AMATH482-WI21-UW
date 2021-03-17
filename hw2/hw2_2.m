clear; close all; clc


[y, Fs] = audioread('Floyd-section1.m4a');
L = length(y)/Fs; % record time in seconds
n = length(y);
t2 = linspace(0,L,n+1); t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);%frequency domain

a = 100;
tau = (0:0.1:L);
ygt_spec = zeros(length(y),length(tau));
%% Gabor Transform and Spectrum
for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2); %window function
    yg = g.*y';
    ygt = fft(yg);
    ygt_spec(:,j) = fftshift(abs(ygt));
end

figure(1)
a = pcolor(tau,ks,ygt_spec(1:length(ks),:));
shading interp
set(gca,'ylim',[200 800],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')

