function res=ifft_mats_non_uniform_amir(values,sampling_locations,cell_dimensions,N,SPURS_settings)
L=length(cell_dimensions);
res=zeros(N*N,L);
sampling_locations=cell2mat(sampling_locations');
loc_in_cell_dimensions = 1:cell_dimensions(1):(L*(cell_dimensions(1))+1);
values = double(values);
% h = waitbar(0,'Please wait...');
for i=1:L
%     waitbar(i/L,h,['Iteration ',num2str(i),' of ',num2str(L)]);
[tmp_res,~]=SPURS(values(loc_in_cell_dimensions(i):(loc_in_cell_dimensions(i+1)-1)),sampling_locations(loc_in_cell_dimensions(i):(loc_in_cell_dimensions(i+1)-1),:),SPURS_settings);
res(:,i) = reshape(tmp_res(:,:,end),N*N,1);
end

i=1;