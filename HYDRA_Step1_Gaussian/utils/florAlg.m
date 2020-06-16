function [ X_nuc, E_matched_filter_cs] = florAlg( Y,D,N,L,X,r,th )
%This function calculates FLOR algorithm
Y_sticks = reshape(Y,N*N,L);
base = orth(D.');
X_nuc = minNuc( Y_sticks,base.' ,r,X,th);
X_nuc = reshape(X_nuc,N,N,L);
E_matched_filter_cs=full(find_E_fast(X_nuc,D));             %results is one-sparse E
X_nuc = reshape(X_nuc,N,N*L);

end

