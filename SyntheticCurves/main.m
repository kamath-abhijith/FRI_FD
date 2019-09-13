clear
clc
close all

%% Support
T = 0.01; var = 0.1;
n = 1:ceil(2*pi/T); N = length(n);
nT = n*T +- (T/2)+T*rand(1,N);

%% Initialise shape
% a = [0 7]; b = [0 8*exp(j*pi/10)];                      % Ellipse
% a = [0 4 2*exp(j*pi/3)]; b = [0 5 2.5];                 % Boomerang
b = [1 4 1*exp(j*pi/6)]; a = [5 6 .2*exp(j*pi/4)];      % Oval
% a = [2 5 2.5]; b = [-j*4 5 2.5];                        % Cardiod
% a = [0 3 10 2 1]; b = [0 10 5 4 2];                     % Pacman
% a = [0 30 1 .2 .1]; b = [0 10 5 4 2];                   % Prayer
% a = [0 6 .9 .4]; b = [0 6 -.8 1];

K = length(a);
x = a(1); y = b(1);
for m = 1:K-1
   x = x + a(m+1)*exp(1j*m*nT) + conj(a(m+1))*exp(-1j*m*nT);
   y = y + b(m+1)*exp(1j*m*nT) - conj(b(m+1))*exp(-1j*m*nT);
end

%% Add noise to coordinate functions
xn = x + sqrt(var)*randn(1,N) + j*sqrt(var)*randn(1,N);
yn = y + sqrt(var)*randn(1,N) + j*sqrt(var)*randn(1,N);

% Create partial samples
Mp = ceil(1*N);
xn = xn(1:Mp); yn = yn(1:Mp);

%% Denoising
M = floor(2*Mp/3);
win = tukeywin(M+1, .99);
hlp = fir1(M, .157, win);

xL = cconv(xn, hlp);
yL = cconv(yn, hlp);

%% Estimation of parameters
T_est = FRIFD_T(xL,yL,K);
[pa, pb] = FRIFD_C(xL,yL,K,T_est);

%% Curve reconstruction
Nr = ceil(2*pi/T_est); nr = 1:Nr;
xr = pa(K); yr = pb(K);
for m = 1:K-1
   xr = xr + pa(K+m)*exp(1j*m*nr*T_est) + pa(K-m)*exp(-1j*m*nr*T_est);
   yr = yr + pb(K+m)*exp(1j*m*nr*T_est) + pb(K-m)*exp(-1j*m*nr*T_est);
end

%% Plots
figure
plot(real(x), imag(y), '-r', "LineWidth", 4)
grid on, hold on
plot(real(xr), imag(yr), '-b', "LineWidth", 2)
scatter(real(xn), imag(yn), '.g', "LineWidth", 4)

set(gca, 'FontSize', 14)
set(gca,'color','none','xcol','w','ycol','w','GridColor','w')

set(gcf, 'Color', 'k');
set(gcf, 'InvertHardCopy', 'off');

%% Functions
function a = cconv(b,c)
% Performs circular convolution with
% the delay removed
% 
% INPUT: Vectors to be convolved, b,c
% 
% OUTPUT: Convolution output

    % Support vectors
    nb = length(b);
    nc = length(c);
  
    % Zero pad smaller vector
    if (nb<nc)
        b = [b zeros(1,nc-nb)];
    end

    if (nc<nb)
        c = [c zeros(1,nb-nc)];
    end

    % Convolve using fft and ifft
    a = ifft(fft(b).*fft(c));

    % Force a to be real
    if (all(b == real(b)) && all(c == real(c)))
        a = real(a);
    end

    % Remove the delay
    a = [a(ceil(nc/2):end) a(1:ceil(nc/2)-1)];
end

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