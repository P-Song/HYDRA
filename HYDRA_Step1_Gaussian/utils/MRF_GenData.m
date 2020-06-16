% 29/05/2018 generate subsampled signatures for input and fully sampled signatures for output.

% X: 128x128x500 , fully_sampled_contrasts with 500 time points
% Y: 


%% Initialize:
clear;
addpath('Bloch\');
N=128;                                                               %The image size is NxN
L = 500;                                                             %The sequence length
k_space_undersampling_ratio=0.15;             %undersampling ratio in k-space

load input_to_fisp_experiment                        %load the quantitative maps
[RFpulses, TR]=generate_RF_TR(L);            %Load slowly changing RF and TR values
RFpulses = RFpulses*1i;                                %to avoid complex values of X and D
TE = 10;%2

%% build dictionary using MRF FISP sequence, omit entries where T2>T1
% if exist('D.mat')~=0                                                                                                         % you can save X, D, and LUT to D.mat to save time
% 	load D.mat;

if exist('MRF_Signatures.mat')~=0                                                                                                         % you can save X, D, and LUT to D.mat to save time
	load MRF_Signatures.mat;
else
    disp('Building the dictionary...');
    [FISP_dictionary,LUT] = build_dictionary_fisp(L,RFpulses,TR,TE);                     %build the MRF FISP dictionary and its matching look up table
    D = single(FISP_dictionary);
    clear FISP_dictionary;
    LUT = LUT*1000;                                                                                                        %change units
    disp('The dictionary is ready, building the temporal images...');
    X = build_fully_sampled_contrasts(RFpulses ,TR ,TE, T1_128,T2_128,PD_128);  %build the fully sampled temporal contrasts
    disp('The images are ready');
% 	save('D_X_LUT.mat', 'D', 'X', 'LUT')
end
	%% undersample the data in Fourier domain 
	X = reshape(X,N,N*L);
	sampling_matrix = genrate_binary_sampling_map(N,k_space_undersampling_ratio,L); %generates binary sampling masks
	Y_full = fft_mats(X,1);
	kSpaceNoise =  reshape([1 1i]*0.5*randn(2,L*N^2),N,L*N);                        
	disp('Adding noise to the data in Fourier domain');
	Y = Y_full + kSpaceNoise;                                                                                                         %add noise to the data, complex white noise with sigma=0.5
	disp('Under-sampling the noisy data in Fourier domain');
	Y = sampling_matrix.*Y;                                                                                                             %uder-sampling the noised data

	%% Signature reconstruction from subsampled measurements using conventional MRF
	disp('Reconstruction from subsampled measurements using conventional MRF');
	X_subsamp = reshape(fft_mats(Y,2),N,N,L); 

	%% Signature reconstruction from fully-sampled measurements using conventional MRF
	disp('Signature reconstruction from fully-sampled measurements using conventional MRF');
	X_fullysamp = reshape(fft_mats(Y_full,2),N,N,L);                                                              %Fully sampled non-noised data, used as reference

% 	save('MRF_Signatures.mat', 'D', 'X', 'LUT', 'Y')
	


%%
row = 50; col = 50;
figure; 
plot(squeeze(abs(X_subsamp(row, col, :))), 'r'); hold on;
plot(squeeze(abs(X_fullysamp(row, col, :))), 'b')

figure; 
subplot(1,2,1)
imagesc(abs(X_subsamp(:, :, 250))); colormap gray;
subplot(1,2,2)
imagesc(abs(X_fullysamp(:, :, 250)));  colormap gray;

%%
% estimate maps using matched filter for conventional MRF solution
% E_matched_filter_old_mrf: 16384x3336£» X_estimated_old_mrf: 128x128x500
E_matched_filter_old_mrf = full(find_E_fast(X_estimated_old_mrf,D));                                         %results is one-sparse E
[T1_old_mrf,T2_old_mrf,PD_old_mrf] = build_maps_from_E(E_matched_filter_old_mrf,LUT);   %restore the quatitative maps from E and LUT

%% check whether X can give original contrasts. 
% It seems the original X cannot guarantee successful finding of the exact filter for each tissue. As this is a sparse coding problem, failure implies that D does not satisfy RIP condition well.
E_matched_filter_orig = full(find_E_fast( reshape(X,N,N,L) ,D));                                         %results is one-sparse E
[T1_orig2,T2_orig2,PD_orig2] = build_maps_from_E(E_matched_filter_orig,LUT);   %restore the quatitative maps from E and LUT

%% FLOR algorithm
disp('Calculating the parameter maps using FLOR');
r = L;
th = 5;
[ X_estimated_flor, E_estimated_flor] = florAlg( Y,D,N,L,X,r,th );
[T1_flor,T2_flor,PD_flor] = build_maps_from_E(E_estimated_flor,LUT);
%% Show Results in the region of interest
T1.orig = T1_128;                                                T2.orig = T2_128;                                                      PD.orig = PD_128;
T1.old_mrf = T1_old_mrf.*(T1.orig~=0);            T2.old_mrf = T2_old_mrf.*(T2.orig~=0);                 PD.old_mrf = PD_old_mrf.*(PD.orig~=0);
T1.flor = T1_flor.*(T1.orig~=0);                           T2.flor = T2_flor.*(T2.orig~=0);                                PD.flor = PD_flor.*(PD.orig~=0);
showResults( k_space_undersampling_ratio,T1,T2,PD );

%% show PSNR and SSIM
PSNR = [];  SNR = [];
[PSNR.T1.old_mrf SNR.T1.old_mrf] = psnr( T1.old_mrf , T1.orig ) ;
[PSNR.T2.old_mrf SNR.T2.old_mrf] = psnr( T2.old_mrf , T2.orig ) ;

[PSNR.T1.flor SNR.T1.flor] = psnr( T1.flor , T1.orig ) ;
[PSNR.T2.flor SNR.T2.flor] = psnr( T2.flor , T2.orig ) ;

T1.orig2 = T1_orig2.*(T1.orig~=0);            T2.orig2 = T2_orig2.*(T2.orig~=0);   
[PSNR.T1.orig2 SNR.T1.orig2] = psnr( T1.orig2 ,  T1.orig ) ;
[PSNR.T2.orig2 SNR.T2.orig2] = psnr( T2.orig2 , T2.orig ) ;


PSNR.T1, PSNR.T2, SNR.T1, SNR.T2


