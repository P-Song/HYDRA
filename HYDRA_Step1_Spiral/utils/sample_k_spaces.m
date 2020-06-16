function sampled_k_space_data=sample_k_spaces(sampling_locations,fully_sampled_contrasts)
sampled_k_space_data=cell(1,length(sampling_locations));
% h = waitbar(0,'Please wait...');

for i=1:length(sampling_locations)
% parfor i=1:length(sampling_locations)
%     waitbar(i/length(sampling_locations),h,['Iteration ',num2str(i),' of ',num2str(length(sampling_locations))]);
   % sampled_k_space_data{i}=zeros(size(sampling_locations,1),2);
  %  sampled_k_space_data{i}(:,1)=sampling_locations{i};
    im=squeeze(fully_sampled_contrasts(:,:,i));
  
   %--------NUFFT-------------
    
    sampled_k_space_data{i}=single(non_uniform_fessler_fft2c(im,sampling_locations{i})); %the samples of the FFT in the locations of "file"
end
% close(h);

a=1;