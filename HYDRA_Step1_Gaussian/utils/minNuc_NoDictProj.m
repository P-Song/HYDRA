function [ U ] = minNuc( Y,r,X_real,th )
% 27/10/2018 No dictionary, thus no projection operation.
%%solver using iterations
X = zeros(size(Y));
U = X;
mu = 1;
L = size(Y,2);
N = sqrt(size(Y,1));
number_of_iterations = 100;
t = 1;

Y = reshape(Y,N,N*L);
figure;
% h = waitbar(0,'Please wait...');
i = 1;
% pinv_D = pinv(D);
% pinv_DD = pinv_D*D;
mseX = zeros(1,number_of_iterations-1);
while i<number_of_iterations %&& diff_X>error;
%     waitbar(i/number_of_iterations,h,['Iteration ',num2str(i),' of ',num2str(number_of_iterations)]); 
    %mu = 1/sqrt(i);
    X_prev = X;
    X = U-(mu)*reshape(fft_mats((Y~=0).*(fft_mats(reshape(U,N,N*L),1)-Y),2),N*N,L);
%     X = X*pinv_DD;
    X = proj_rank( X,r,th );
%Not Accelerated:
%     U = X ;
%Accelerated:
   t_prev = t;
   t = 0.5*(1+sqrt(1+4*t^2));
   U = X + ((t_prev-1)/t)*(X-X_prev);
%
   X_estimated_cs = reshape(X,N,N,L);
   subplot 121;
   imagesc(abs(X_estimated_cs(:,:,50)));title(num2str(i));drawnow;
   mseX(i) = calc_mse(X,X_real);
   subplot 122;plot(mseX);
   i = i + 1;
end
% close(h);

end

