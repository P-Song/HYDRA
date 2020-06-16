function [ val ] = calc_mse( x,ref )
%MSE calculates the mean square error in the region of interest
v = x(:)-ref(:);
N = (ref~=0); N = sum(N(:));
val = (1/N)*(v'*v);
end

