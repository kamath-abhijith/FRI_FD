clear
clc
close all

%% Load data // SMU Dataset, China
load brainTumorDataPublic_1766/3.mat %loads cjdata
img = cjdata.image;
edge = cjdata.tumorBorder;
mask = cjdata.tumorMask;

xn = edge(1:2:end-1);
yn = edge(2:2:end);

%% Image to display // overlay with mask
disp_img = adapthisteq(img) + 10^4*(int16(mask));

%% Parameter estimation
K = 7;

T = FRIFD_T(xn,yn,K);
[pa,pb] = FRIFD_C(xn,yn,K,T);

%% Curve reconstruction
Nr = ceil(2*pi/T); nr = 1:Nr;
xr = pa(K); yr = pb(K);
for m = 1:K-1
   xr = xr + pa(K+m)*exp(1j*m*nr*T) + pa(K-m)*exp(-1j*m*nr*T);
   yr = yr + pb(K+m)*exp(1j*m*nr*T) + pb(K-m)*exp(-1j*m*nr*T);
end

%% Figures
figure
imshow(disp_img)
hold on
% scatter(xn, yn, 'b*')
plot(xr, yr, "Color", [1 0 0], "LineWidth", 3)
print(gcf, 'MRI', '-depsc', '-r300')

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