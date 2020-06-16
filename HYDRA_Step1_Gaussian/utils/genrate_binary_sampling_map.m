function sampling_mat=genrate_binary_sampling_map(N,undersampling_ratio,L)
% This function generates the binary under-sampling  masks for each image, low
% frequencies have higher probabilty to be chosen
sampling_mat= false(N,N*L);
pdf_vardens_cut=genPDF([N,N],9,undersampling_ratio,2,0,0);
for i=1:L
    r_mat=rand(N);  
    pdf_vardens2=(r_mat.*pdf_vardens_cut);
    pdf_vardens3=pdf_vardens2(:);
    [~,b]=sort(pdf_vardens3);
    b=flipud(b);
    threshold_for_sampling=pdf_vardens3(b(round(undersampling_ratio*length(b))));
    pdf_vardens4=zeros(N);
    pdf_vardens4(pdf_vardens2>=threshold_for_sampling)=1;
    sampling_mat(1:N,(i-1)*N+1:i*N)=logical(fftshift(pdf_vardens4));
end
