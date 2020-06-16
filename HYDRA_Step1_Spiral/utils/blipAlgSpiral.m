function [ X_estimated_blip,E_matched_filter_blip ] = blipAlgSpiral( Y_spiral_noised,sampling_locations_spiral,cell_dimensions,D,N,L,X )
% This function calculates BLIP algorithm

X_estimated_blip = zeros(N*N,L);
num_of_iterations = 30;
muInit = 1;
i = 1;
mu = muInit;
figure;
sampling_locations_spiral_amir = cellfun(@(x) x*128,sampling_locations_spiral,'un',0);
Y_spiral_noised_amir = cellfun(@(x) x/128/128,Y_spiral_noised,'un',0);
mse_blip = zeros(1,num_of_iterations);
figure;
h = waitbar(0,'Please wait...');
while i<=num_of_iterations
    waitbar(i/num_of_iterations,h,['Iteration ',num2str(i),' of ',num2str(num_of_iterations)]);
    prev_X = X_estimated_blip; 
    spiralX = sample_k_spaces(sampling_locations_spiral,reshape(X_estimated_blip,N,N,L));
% NUFFT:   
%     subXY =cellfun(@minus,spiralX,Y_spiral_noised,'UniformOutput',false);
%     X_estimated_blip=X_estimated_blip-(mu)*single(ifft_mats_non_uniform_fessler(cell2mat(subXY'),sampling_locations_spiral,cell_dimensions,N));
% SPURS:
    spiralX = cellfun(@(x) x/128/128,spiralX,'un',0);
    subXY = cellfun(@minus,spiralX,Y_spiral_noised_amir,'UniformOutput',false);
    X_estimated_blip=X_estimated_blip-(mu)*single(ifft_mats_non_uniform_amir(cell2mat(subXY'),sampling_locations_spiral_amir,cell_dimensions,N,evalin('base','SPURS_settings')));    
    
    X_estimated_blip=reshape(X_estimated_blip,N,N,L);
    E_matched_filter_blip=find_E_fast(X_estimated_blip,D);
    E_matched_filter_blip = (E_matched_filter_blip>0).*E_matched_filter_blip;
    X_estimated_blip = E_matched_filter_blip*D;
    subplot 121;imagesc(abs(reshape(X_estimated_blip(:,50),N,N)));title(num2str(i));drawnow;
    mse_blip(i) = calc_mse(X_estimated_blip,X);
    if i>1
        if mse_blip(i)<= mse_blip(i-1);
            mu = muInit;
        elseif mu>0.01
            X_estimated_blip = prev_X;
            mse_blip(i) = mse_blip(i-1);
            mu = mu*0.9;
        else
                break;
        end
    end
    subplot 122;plot(mse_blip);
    i = i+1;
end
close(h);

