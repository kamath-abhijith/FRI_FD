function a = cconv(b,c)
% Performs circular convolution with
% the delay removed
% 
% INPUT: Vectors to be convolved, b,c
% 
% OUTPUT: Convolution output
% 
% Written by: Abijith J Kamath
% kamath-abhijith.github.io


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