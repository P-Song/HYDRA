function [ X_nuc ] = florAlg( Y,D,N,L,X,r,th )
% 18/10/2018 Perform inner product one sample by one sample to save memory

%This function calculates FLOR algorithm
Y_sticks = reshape(Y,N*N,L);
base = orth(D.');
X_nuc = minNuc( Y_sticks,base.' ,r,X,th);
X_nuc = reshape(X_nuc,N,N,L);
X_nuc = reshape(X_nuc,N,N*L);

% % Do not compute and return E_matched_filter_cs anymore. It is moved to
% % build_maps_SaveMem.m function.
% E_matched_filter_cs = find_E_SaveMem(X_nuc,D); %results is one-sparse E
% % E_matched_filter_cs=full(E_matched_filter_cs); % may cost a huge memory           
end

