function [ E ] = find_E_fast( X,D, varargin )
% 06/12/2018 add quiet option for not displaying waitbar.
% 18/10/2018 Perform inner product one sample by one sample to save memory

%This function returns 1 row sparse matrix E that matches for each pixel in X a
%signature from D using matched filter


if ~isempty(varargin)
    Quiet = varargin{1};
else
    Quiet = 0;
end


X=reshape(X,size(X,1)*size(X,2),size(X,3));
X=X.';
%normalize
norms_D = sqrt(sum(abs(D).^2,2));
norms_D_mat=repmat(norms_D,1,size(D,2));
D_normed=D./norms_D_mat;
clear norms_D_mat

norms_X = sqrt(sum(abs(X).^2,1));
norms_X_mat=repmat(norms_X,size(X,1),1);
X_normed=(X)./norms_X_mat;

clear norms_X_mat
X_normed(isnan(X_normed))=0;

ind=1:length(norms_X);
clear norms_X
E=sparse(size(X,2),size(D,1));
val_mats = zeros(1,size(E,1));
max_ind = zeros(1,size(E,1));


% [val_mats(ind),max_ind]=max(real(D_normed*X_normed(:,ind))); % may consume a large amount of memory if D is large

if Quiet > 0
    
    % Perform inner product one sample by one sample to save memory
    for i = 1 : size(X,2)
    % 	E=sparse(1 ,size(D,1));
        [val_mats(i),max_ind(i)]=max(real(D_normed*X_normed(:,i)));

        if abs(val_mats(i))>0.9
    % 		T1(i)=LUT(max_ind(i),1);
    % 		T2(i)=LUT(max_ind(i),2);
            InnerProd = real(D_normed(max_ind(i),:)*X(:,i));
            E(i,max_ind(i)) = InnerProd/(norms_D(max_ind(i))) ;
        end
    end

else

    % Perform inner product one sample by one sample to save memory
    hwait=waitbar(0,'Please wait>>>>>>>>');
    for i = 1 : size(X,2)

        if mod(i, 1000) ==0
            str=['running ',num2str(i/ size(X,2) * 100),'%'];
            waitbar(i/size(X,2),hwait,str);
            pause(0.02);
        end

    % 	E=sparse(1 ,size(D,1));
        [val_mats(i),max_ind(i)]=max(real(D_normed*X_normed(:,i)));

        if abs(val_mats(i))>0.9
    % 		T1(i)=LUT(max_ind(i),1);
    % 		T2(i)=LUT(max_ind(i),2);
            InnerProd = real(D_normed(max_ind(i),:)*X(:,i));
            E(i,max_ind(i)) = InnerProd/(norms_D(max_ind(i))) ;
        end

    end
    close(hwait);

end



end

