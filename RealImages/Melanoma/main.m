clear
clc
close all

%% Load data // Image and coordinates
img = imread('170216.jpg');
imgBW = rgb2gray(img) + 50;

load mel1_x.mat
load mel1_y.mat

xn = xn(1:1:end);
yn = yn(1:1:end);

%% Parameter estimation
K = 22;

T = FRIFD_T(xn,yn,K);
[pa,pb] = FRIFD_C(xn,yn,K,T);

%% Reconstruction
Nr = ceil(2*pi/T); nr = 1:Nr;
xr = pa(K); yr = pb(K);
for m = 1:K-1
   xr = xr + pa(K+m)*exp(1j*m*nr*T) + pa(K-m)*exp(-1j*m*nr*T);
   yr = yr + pb(K+m)*exp(1j*m*nr*T) + pb(K-m)*exp(-1j*m*nr*T);
end

%% Figures
figure
imshow(imgBW)
hold on
plot(yr,xr,'-r', "LineWidth",5)
% print(gcf, 'mel1', '-depsc', '-r300')

%% Functions
function T = FRIFD_T(x,y,K)
% Estimation of sampling iterval
% using block annihilating filter
%
% INPUT:  Coordinate functions, (x,y)
%         Model order, K  
% OUTPUT: Estimated sampling interval, T

    % Preprocessing // select windowed signal
    N = length(x); n = 1:N;
    M1 = floor(N/2);
    M2 = floor(N/3);
    
    % Block annihilation to find T and its integer multiples
    try
        kT = block_ann(x(M1-M2:M1+M2), y(M1-M2:M1+M2), 2*K);
    catch
        kT = block_ann(x(M1-M2:M1+M2), y(M1-M2:M1+M2), 2*K);
    end
    
    % Find the 1*T
    kT = sort(kT+2*pi);
    idxT = find(kT>0.006, 1, 'first');
    T = kT(idxT);
end

function [pa,pb] = FRIFD_C(x,y,K,T)
% Estimation of Fourier Descrtiptors
% using least-squares fit
%
% INPUT:  Coordinate functions, (x,y)
%         Model order, K
%         Estimated sampling interval, T
% OUTPUT: Real ang imag. parts of Fourier descriptors, (pa,pb)

    N = length(x); n = 1:N;

    % Least squares
    ExM  = ones(N,1);
    for m = 1:K-1
        ExM = [transpose(exp(-1j*m*T*n)) ExM transpose(exp(1j*m*T*n))];
    end

    try
        pa = pinv(ExM)*x(:);
        pb = pinv(ExM)*y(:);
    catch
        pa = pinv(ExM(1:end-1,:))*x(:);
        pb = pinv(ExM(1:end-1,:))*y(:);
    end
end