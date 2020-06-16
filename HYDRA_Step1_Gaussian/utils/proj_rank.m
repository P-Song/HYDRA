function [ X ] = proj_rank( Y,r,t )
%Returns X, which is the composition of [U S_t V], where [U S V] are the
%singular values decomposition of Y, and S_t contains  the r largest singular
%values of S soft-thresholded by operator t.
if nargin<3
    t = 0;
end

[U, S, V] = svd(Y,'econ');
S_t = wthresh(S(1:r,1:r),'s',t);
V = V';
X = U(:,1:r)*S_t*V(1:r,:);

end

