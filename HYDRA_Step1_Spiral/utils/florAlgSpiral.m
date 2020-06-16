function [ X_nuc_spiral, E_matched_filter_cs] = florAlgSpiral( Y,sampling_locations_spiral,cell_dimensions,D,N,L,X,th )
%This function calculates FLOR algorithm
Y_spiral_noised_amir = cellfun(@(x) x/128/128,Y,'un',0);
base = orth(D.');
%NUFFT:
% X_nuc_spiral = minNucSpiral( N,Y_spiral_noised,sampling_locations_spiral,cell_dimensions,base.' );
%SPURS:
X_nuc_spiral = minNucSpiral( N,Y_spiral_noised_amir,sampling_locations_spiral,cell_dimensions,base.' ,X,th);

X_nuc_spiral = reshape(X_nuc_spiral,N,N,L);
E_matched_filter_cs=find_E_fast(X_nuc_spiral,D);             %results is one-sparse E
X_nuc_spiral = reshape(X_nuc_spiral,N*N,L);

end

