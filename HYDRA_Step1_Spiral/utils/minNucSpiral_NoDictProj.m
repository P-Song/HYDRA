function [ X ] = minNucSpiral( N,Y,sampling_locations,cell_dimensions,X_real,th )
% 13/04/2019 No dictionary, thus no projection operation. But the
% convergence speed decreases. 

%%FLOR solver using iterations
L = size(Y,2);
X = zeros(N*N,L);
U = X;
number_of_iterations = 100;
mu=1;
t = 1;
r = L; 
% pinv_D = pinv(D); % comment to avoid projection on dictionary;
% pinv_DD = pinv_D*D; % comment to avoid projection on dictionary;
sampling_locations_spiral_amir = cellfun(@(x) x*128,sampling_locations,'un',0);
mseX = zeros(1,number_of_iterations-1);
figure;
h = waitbar(0,'Please wait...');
i=1;
while i<number_of_iterations 
    waitbar(i/number_of_iterations,h,['Iteration ',num2str(i),' of ',num2str(number_of_iterations)]);
    spiralX = sample_k_spaces(sampling_locations,reshape(U,N,N,L));
    % NUFFT:  
%     subXY =cellfun(@minus,spiralX,Y,'UniformOutput',false);
%     X_prev = X;
%     X=U-(mu)*single(ifft_mats_non_uniform_fessler(cell2mat(subXY'),sampling_locations,cell_dimensions,N));
    % SPURS:
    spiralX = cellfun(@(x) x/128/128,spiralX,'un',0);
    subXY = cellfun(@minus,spiralX,Y,'UniformOutput',false);
    X_prev = X;
    X = U-(mu)*single(ifft_mats_non_uniform_amir(cell2mat(subXY'),sampling_locations_spiral_amir,cell_dimensions,N,evalin('base','SPURS_settings'))); 
    
% 	X = X*pinv_DD; % comment to avoid projection on dictionary;
    X = proj_rank( X,r,th );
    
    %Acceleration:
    t_prev = t;
    t = 0.5*(1+sqrt(1+4*t^2));
    U = X + ((t_prev-1)/t)*(X-X_prev);

    X_estimated_cs=reshape(U,N,N,L);
    mseX(i) = calc_mse(X_estimated_cs,X_real);
    subplot 121;imagesc(abs(X_estimated_cs(:,:,50)));drawnow;title(num2str(i));%[0 20]
    subplot 122;plot(mseX);
    i = i + 1;
 end
close(h);

end

