function [ X_nuc_spiral] = florAlgSpiral( Y,sampling_locations_spiral,cell_dimensions,N,L,X,th )
% 13/04/2019 No dictionary, thus no projection operation.
% 13/04/2019 Perform inner product one sample by one sample to save memory

%This function calculates FLOR algorithm
Y_spiral_noised_amir = cellfun(@(x) x/128/128,Y,'un',0);
% base = orth(D.'); % comment to avoid projection on dictionary;

%NUFFT:
% X_nuc_spiral = minNucSpiral( N,Y_spiral_noised,sampling_locations_spiral,cell_dimensions,base.' );
%SPURS:
X_nuc_spiral = minNucSpiral_NoDictProj( N,Y_spiral_noised_amir,sampling_locations_spiral,cell_dimensions,X,th);

X_nuc_spiral = reshape(X_nuc_spiral,N,N,L);

% % Do not compute and return E_matched_filter_cs anymore. It is moved to build_maps_SaveMem.m function.
% E_matched_filter_cs=find_E_fast(X_nuc_spiral,D);             %results is one-sparse E
% E_matched_filter_cs = find_E_SaveMem(X_nuc,D); %results is one-sparse E

X_nuc_spiral = reshape(X_nuc_spiral,N*N,L);

end

