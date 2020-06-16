function [ X_estimated_blip,E_matched_filter_blip ] = blipAlg( Y,D,N,L,X, Iter )

% save memory by using find_E_SaveMem.m

X_estimated_blip=zeros(N,N*L);
muInit = 1;%1 %1/0.15 0.15 is the sampling ratio
% tol = 10;
% diff_X = 101;
i = 1;
mu = muInit;
mse_blip = [];
while i<=Iter%diff_X>tol
    prev_X = X_estimated_blip;
    X_estimated_blip = X_estimated_blip -mu*((fft_mats((Y~=0).*(fft_mats(X_estimated_blip,1)-Y),2)));
    X_estimated_blip=reshape(X_estimated_blip,N,N,L);
 
    E_matched_filter_blip=find_E_SaveMem(X_estimated_blip,D, 1);
    
%     E_matched_filter_blip=full(find_E_fast(X_estimated_blip,D));% very memory-consuming
    E_matched_filter_blip = (E_matched_filter_blip>0).*E_matched_filter_blip;
    X_estimated_blip = full(E_matched_filter_blip)*D; % very memory-consuming

%     E_matched_filter_blip = (E_matched_filter_blip>0).*E_matched_filter_blip;
%     % slow, but save memory
%     tmp = zeros(N*N,L);
%     for j = 1: (N*N)
%         tmp(j,:) = full(E_matched_filter_blip(j,:))*D;
%     end
%     X_estimated_blip = tmp;

    X_estimated_blip=reshape(X_estimated_blip,N,N*L);
    subplot 121;
    imagesc(abs(X_estimated_blip(:,N*50+1:N*51)));title(num2str(i));drawnow;
    %diff_X = norm(prev_X(:)-X_estimated_blip(:))
%     if mu>0.9*diff_X/(norm(sampling_matrix.*fft_mats(prev_X-X_estimated_blip,1),'fro'))
%         mu = mu*0.5
%         X_estimated_blip = prev_X;
%     else
%         mu = muInit;
%     end
    current_mse = mse(X_estimated_blip,X);
    if i>1
        if current_mse<= mse_blip(end);
            mse_blip = [mse_blip, current_mse];
%             mse_blip = current_mse;
            mu = muInit;
        elseif mu>0.01
            %break;
            X_estimated_blip = prev_X;
%             mse_blip = [mse_blip, mse_blip(end)];
            mu = mu*0.9;
        else
                break;
        end
    else
         mse_blip = [mse_blip,  mse(X_estimated_blip,X)];
%         mse_blip = mse(X_estimated_blip,X);
    end
    subplot 122; plot(mse_blip);drawnow;
    i = i+1;
end


end

