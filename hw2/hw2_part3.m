clear; close all; clc


[y, Fs] = audioread('Floyd.m4a');
L = length(y)/Fs; % record time in seconds
n = length(y);
t2 = linspace(0,L,n+1); t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);%frequency domain

a = 60;
tau = (0:0.1:L);
ygt_spec = zeros(length(y),length(tau));

%% Gabor Transform and filtered Spectrum
yf = zeros(1,length(y));
for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2); %window function
    yg = g.*y';
    ygt = fft(yg);
    ygt_spec(:,j) = fftshift(abs(ygt));
    temp = ygt_spec(:,j);
    
    %looking for the bass frequency(>200Hz) for centering the filter.
    %looking for the maximum value of ygt_spec of any frequency over 200 Hz 
    ks_abs = abs(ks);
    ks_bass_position = ks_abs > 200; %position of bass notes frequency in the frequency vector
    ks_bass = temp(ks_bass_position);
    M = max(ks_bass);
    I = find (temp == M);
    center1 = ks(I(1));%central freuqncy of the filter
    center2 = ks(I(2));
    %Define the frequency filter
    b = 0.3;
    filter1 = exp(-b*(ks-center1).^2);
    filter2 = exp(-b*(ks-center2).^2);
    ygt_spec(1:length(ks),j) = temp(1:length(ks)).'.*filter1+temp(1:length(ks)).'.*filter2;  
end



figure(1)
a = pcolor(tau,ks,ygt_spec(1:length(ks),:));
shading interp
set(gca,'ylim',[200 800],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')

