function [ X_nuc_spiral, varargout] = florAlgSpiral( Y,sampling_locations_spiral,cell_dimensions,D,N,L,X,th )

% 13/04/2019 Perform inner product one sample by one sample to save memory

%This function calculates FLOR algorithm
Y_spiral_noised_amir = cellfun(@(x) x/128/128,Y,'un',0);
base = orth(D.');
%NUFFT:
% X_nuc_spiral = minNucSpiral( N,Y_spiral_noised,sampling_locations_spiral,cell_dimensions,base.' );
%SPURS:
[X_nuc_spiral, mseX] = minNucSpiral( N,Y_spiral_noised_amir,sampling_locations_spiral,cell_dimensions,base.' ,X,th);

X_nuc_spiral = reshape(X_nuc_spiral,N,N,L);

% % Do not compute and return E_matched_filter_cs anymore. It is moved to build_maps_SaveMem.m function.
% E_matched_filter_cs=find_E_fast(X_nuc_spiral,D);             %results is one-sparse E
% E_matched_filter_cs = find_E_SaveMem(X_nuc,D); %results is one-sparse E

X_nuc_spiral = reshape(X_nuc_spiral,N*N,L);

varargout{1} = mseX;
end

