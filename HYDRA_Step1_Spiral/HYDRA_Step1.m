% This code is to perform the first step of HYDRA: 
% reconstruct temporal signatures using adapted FLOR
% algorithm.

% After the temporal signatures are reconstructed, 
% they are input into the trained neural network to 
% restore the parameter maps, i.e. the second step of HYDRA. 
% Please refer to 'MRF_FullNL_ResCNN_T1T2_L1000_Test.py' 
% in the upper folder for the second step.

% Reference:
% ----------------------------
% [1] Pingfan Song, Yonina C. Eldar, Gal Mazor, Miguel R. D. Rodrigues, "HYDRA: Hybrid Deep Magnetic Resonance Fingerprinting", Medical Physics, 2019, doi: 10.1002/mp.13727. 
% [2] Pingfan Song, Yonina C. Eldar, Gal Mazor, Miguel R. D. Rodrigues, "Magnetic Resonance Fingerprinting Using a Residual Convolutional Neural Network", ICASSP, pp. 1040-1044. IEEE, 2019.
% [3] Gal Mazor, Lior Weizman, Assaf Tal, Yonina C. Eldar. "Low‐rank magnetic resonance fingerprinting." Medical physics 45, no. 9 (2018): 4066-4084.


% 14/05/2019 improve spiral trajectory design, by setting the number of interleaves
% N_inner and N_outer as 4 and 10, instead of 1 and 24, i.e. reducing number
% of inner spirals and increasing bumber of outer spirals.

function [Output] = HYDRA_Step1(flag)

%% Initialize:
clear;
addpath(genpath('utils') );
% add path to fessler's Michigan Image Reconstruction Toolbox (MIRT)
addpath('../fessler/irt')
setup;
addpath(genpath('../SPURS'))

if nargin < 1
    flag = 0 ;
end
Output.flag = flag;

N=128;                                                               %The image size is NxN
L = 200; % 1000;  % 200                                                           %The sequence length


[RFpulses, TR]=generate_RF_TR(1000);            %Load slowly changing RF and TR values
RFpulses = RFpulses*1i;                                %to avoid complex values of X and D
TE = 10;%2
RFpulses = RFpulses(1:round(1000/L):1000); % undersampling in time dimension.
TR = TR(1:round(1000/L):1000);

T1_values = [1:10:5000];
T2_values = [1:10:2000];
T1_labels = {'1:10:5000 ms'};
T2_labels = {'1:10:2000 ms'};

load('SPURS_settings.mat');                         %For SPURS algorithm
% setenv('SPURS_DIR', 'spurs_directory_path\SPURS_DEMO');    %set spurs_directory_path to SPURS_DEMO directory on your computer, like: 'c:\...'
setenv('SPURS_DIR', '../SPURS/SPURS_DEMO');    %set spurs_directory_path to SPURS_DEMO directory on your computer, like: 'c:\...'


Params.N = N;
Params.L = L;
Params.RFpulses = RFpulses;
Params.TR = TR;
Params.TE = TE;
Params.T1_values = T1_values;
Params.T2_values = T2_values;
Params.T1_labels = T1_labels;
Params.T2_labels = T2_labels;

load Groundtruth_T1_T2                        %load the quantitative maps

T1max = 4500 ;
T2max = 2500;
PDmax = 117 ;

%% build dictionary using MRF FISP sequence, omit entries where T2>T1
	

% % If there is no available D and LUT, produce them.
%{
InputDataFile = ['D_X_LUT', Size, Len, TEtime, '.mat'];

if exist( InputDataFile )~=0                                                                                                         % you can save X, D, and LUT to D.mat to save time
	load(InputDataFile);
else
    disp('Building the dictionary...');
    [FISP_dictionary,LUT] = build_dictionary_fisp(L,RFpulses,TR,TE, T1_values, T2_values);                     %build the MRF FISP dictionary and its matching look up table
    D = single(FISP_dictionary);
    clear FISP_dictionary;
    LUT = LUT*1000;                                                                                                        %change units
    disp('The dictionary is ready, building the temporal images...');

    disp('Building the temporal images...');
    X_fullysamp = build_fully_sampled_contrasts(RFpulses ,TR ,TE, T1_128,T2_128,PD_128);  %build the fully sampled temporal contrasts
    disp('The images are ready');

	save(InputDataFile, 'D', 'X_fullysamp', 'LUT', 'Params')
end                                                                                                      %uder-sampling the noised data
%}

% % If there is available D and LUT, load them.
load('../D_LUT_L1000_TE10_Train.mat');
load('../X_fullysamp.mat', 'X_fullysamp')

D = D(1:1:end,1:round(1000/L):end) ; % sub-sampled from 1000 time points;
LUT = LUT(1:1:end,:) ; 
X = X_fullysamp(:,:,1:round(1000/L):end) ; % sub-sampled from 1000 time points;

% % fast test
% D = D(1:100:end,1:round(1000/L):end) ; % sub-sampled from 1000 time points;
% LUT = LUT(1:100:end,:) ; 
% X = X_fullysamp(:,:,1:round(1000/L):end) ; % sub-sampled from 1000 time points;

%% undersample the data in Fourier domain 

% Generate spiral sampling trajectories
num_of_TRs = L;
%Magnet parameters
smax=17000;                                 %(=170 mT/m/ms in Siemens)
gmax=4;                                         %(=40 mT/m)
T= 0.00001;
%for inner region (10x10 central k-space)
inner_region_size=40; % related to undersampling ratio. 20: 5%, 40: 11%, 60:15%, 120: 70%
N_inner=4; % 2; 4;                                      %Number of interleaves
N_outer=10; % 8; 12; % 24;
N_pixels=N;                                    %128x128 slice is obtained
FOV=24;
[sampling_locations_spiral, num_samples_inner, num_samples_outer]=...
    spiral_trajectories_fisp_Angle(num_of_TRs,smax,gmax,T,inner_region_size,N_inner,N_outer,N_pixels,FOV);
[sampling_locations_spiral, samples_locations_complex]=normalize_sampling_locations(sampling_locations_spiral,N_pixels);
% Sample the data with spiral trajectories
Y_spiral = sample_k_spaces(sampling_locations_spiral,reshape(X,N,N,L));

%%
% i=9
for i = 1:10
    TempSamp = sampling_locations_spiral{i};
    figure(321); 
    plot(TempSamp(:,1), TempSamp(:,2), '-b', 'LineWidth', 2)
    xlim([-0.5,0.5]); ylim([-0.5,0.5])
    % grid on; grid minor
    % set(gcf, 'position', [100,100,400,400])
    pause(0.01)
    % set(gcf, 'position', [100,100,400,400])
%     print(['spiral_', num2str(i),'.png'], '-dpng');
end

%%
% Add noise to the sampled data
% kSpaceNoise =  reshape([1 1i]*0.5*randn(2,L*N^2),N,L*N); % noise
kSpaceNoise =  0*reshape([1 1i]*0.5*randn(2,L*N^2),N,L*N); % no noise
spiral_noise = sample_k_spaces(sampling_locations_spiral,reshape(kSpaceNoise,N,N,L));
Y_spiral_noised = cellfun(@plus,Y_spiral,spiral_noise,'UniformOutput',false);

% sampled_k_space_data has num_of_TRs celss, and each cell holds the
% sampling locations and the k_space values for each TR
cell_dimensions=zeros(1,L);
for i=1:length(Y_spiral_noised)
    cell_dimensions(i)=size(Y_spiral_noised{i},1);
end 

k_space_undersampling_ratio = (num_samples_inner+num_samples_outer)/(N^2);
Params.k_space_undersampling_ratio = k_space_undersampling_ratio;
Params.num_of_TRs = num_of_TRs;
Params.smax = smax;
Params.gmax = gmax;
Params.T = T;
Params.inner_region_size = inner_region_size;
Params.N_inner = N_inner;
Params.N_outer = N_outer;
Params.N_pixels = N_pixels;
Params.FOV = FOV;
Params.sampling_locations_spiral = sampling_locations_spiral;
Params.num_samples_inner = num_samples_inner;
Params.num_samples_outer = num_samples_outer;
Params.samples_locations_complex = samples_locations_complex;

% Params.kSpaceNoise = kSpaceNoise;
% Params.spiral_noise = spiral_noise;

Size = ['_N', num2str(N)];
Len = ['_L', num2str(L) ];
TEtime = ['_TE', num2str(TE) ];
Ratio = ['_Ratio_', num2str(round(k_space_undersampling_ratio*100))];
FileName = [ 'FLOR_Spiral', Size, Len, TEtime, Ratio, '.mat'];


%% adapted FLOR algorithm for reconstructing temporal signatures.
disp('Calculating the parameter maps using FLOR');
th = 5; 

% % fast, but consume large memory
% [ X_estimated_flor, E_estimated_flor] = florAlgSpiral( Y_spiral_noised,sampling_locations_spiral,cell_dimensions,D,N,L,X,th );
% [T1_flor,T2_flor,PD_flor] = build_maps_from_E(E_estimated_flor,double(LUT));                             %restore the quatitative maps from E and LUT

% % slow, but save memory
tic;
[X_estimated_flor, mseX ] = florAlgSpiral_SaveMem( Y_spiral_noised,sampling_locations_spiral,cell_dimensions,D,N,L,X,th );
Threshold = 8; % 20;
X_estimated_flor = reshape(X_estimated_flor,N*N,L); 
NormX = sum(real(X_estimated_flor).^2,2); 
X_estimated_flor(NormX < Threshold, :) = 0; % too small signatures will be set to 0;

X_estimated_flor = reshape(X_estimated_flor,N,N,L); 
TimeCost_flor = toc;

XFileName = [ 'X_Est_FLOR_Spiral', Size, Len, TEtime, Ratio, '.mat'];
save(XFileName, 'X_estimated_flor');

% After the temporal signatures 'X_estimated_flor' are reconstructed, 
% they are input into the trained neural network to 
% restore the parameter maps, i.e. the second step of HYDRA. 
% Please refer to 'MRF_FullNL_ResCNN_T1T2_L1000_Test.py' 
% in the upper folder for the second step.






%% dictionary matching for reconstructing parameter maps. 
% !!! Note, this part is only used for a comparison purpose. The reconstructed
% parameter maps are the final results of adapted FLOR, not HYDRA. 
% The final results of HYDRA come from a deep neural network with the
% reconstructed temporal signatures as input.

[T1_flor,T2_flor,PD_flor] = build_maps_SaveMem(X_estimated_flor, D, LUT);  %restore the quatitative maps from D and LUT

%% Show Results in the region of interest
T1.orig = T1_128;                                                T2.orig = T2_128;                                                      PD.orig = PD_128;
T1.old_mrf = T1_old_mrf.*(T1.orig~=0);            T2.old_mrf = T2_old_mrf.*(T2.orig~=0);                 PD.old_mrf = PD_old_mrf.*(PD.orig~=0);
T1.flor = T1_flor.*(T1.orig~=0);                           T2.flor = T2_flor.*(T2.orig~=0);                                PD.flor = PD_flor.*(PD.orig~=0);
% showResults( (num_samples_inner+num_samples_outer)/(N^2),T1,T2,PD );


%% show PSNR and SSIM
PSNR = [];  SNR = [];
[PSNR.T1.old_mrf SNR.T1.old_mrf] = psnr( T1.old_mrf , T1.orig, T1max) ;
[PSNR.T2.old_mrf SNR.T2.old_mrf] = psnr( T2.old_mrf , T2.orig, T2max ) ;
[PSNR.PD.old_mrf SNR.PD.old_mrf] = psnr( PD.old_mrf , PD.orig, PDmax ) ;

[PSNR.T1.flor SNR.T1.flor] = psnr( T1.flor , T1.orig, T1max) ;
[PSNR.T2.flor SNR.T2.flor] = psnr( T2.flor , T2.orig, T2max ) ;
[PSNR.PD.flor SNR.PD.flor] = psnr( PD.flor , PD.orig, PDmax ) ;


RMSE.T1.old_mrf = sqrt(mean((T1.old_mrf(:) - T1.orig(:)).^2));  % Root Mean Squared Error
RMSE.T1.flor = sqrt(mean((T1.flor(:) - T1.orig(:)).^2));  % Root Mean Squared Error
%
RMSE.T2.old_mrf = sqrt(mean((T2.old_mrf(:) - T2.orig(:)).^2));  % Root Mean Squared Error
RMSE.T2.flor = sqrt(mean((T2.flor(:) - T2.orig(:)).^2));  % Root Mean Squared Error
%
RMSE.PD.old_mrf = sqrt(mean((PD.old_mrf(:) - PD.orig(:)).^2));  % Root Mean Squared Error
RMSE.PD.flor = sqrt(mean((PD.flor(:) - PD.orig(:)).^2));  % Root Mean Squared Error



% compute correlation coefficients
coeff.T1.old_mrf = corrcoef(T1.orig(:),T1.old_mrf(:)) ;
coeff.T1.old_mrf = coeff.T1.old_mrf(1,2);
coeff.T1.flor = corrcoef(T1.orig(:),T1.flor(:)) ;
coeff.T1.flor = coeff.T1.flor(1,2);
%
coeff.T2.old_mrf = corrcoef(T2.orig(:),T2.old_mrf(:)) ;
coeff.T2.old_mrf = coeff.T2.old_mrf(1,2);
coeff.T2.flor = corrcoef(T2.orig(:),T2.flor(:)) ;
coeff.T2.flor = coeff.T2.flor(1,2);
%
coeff.PD.old_mrf = corrcoef(PD.orig(:),PD.old_mrf(:)) ;
coeff.PD.old_mrf = coeff.PD.old_mrf(1,2);
coeff.PD.flor = corrcoef(PD.orig(:),PD.flor(:)) ;
coeff.PD.flor = coeff.PD.flor(1,2);

%% save output
Output.PSNR = PSNR;
Output.SNR = SNR;
Output.RMSE = RMSE;
Output.coeff = coeff;
Output.T1 = T1;
Output.T2 = T2;
Output.PD = PD;
Output.TimeCost_old_mrf = TimeCost_old_mrf;
Output.TimeCost_flor = TimeCost_flor;
Output.mseX = mseX ;

save(FileName, 'Params', 'Output')

%% show figures;
%{
figure; 
imagesc([Output.T1.orig, Output.T1.flor]); 
colormap jet

figure; 
imagesc([Output.T2.orig, Output.T2.flor]); 
colormap jet

figure; 
imagesc([Output.PD.orig, Output.PD.flor]); 
colormap jet

figure; 
plot(Output.T1.orig(:),Output.T1.old_mrf(:),'r.', 'MarkerSize', 15); hold on
plot(Output.T1.orig(:),Output.T1.orig(:),'b-')
% plt.title('T1_Corr')
grid on
xlim([0, 5000])
ylim([0, 5000])
xlabel('Reference T1 (ms)')
ylabel('Estimated T1 (ms)')
legend('Estimation', 'Reference', 'location', 'best')

figure; 
plot(Output.T2.orig(:),Output.T2.old_mrf(:),'r.', 'MarkerSize', 15); hold on
plot(Output.T2.orig(:),Output.T2.orig(:),'b-')
% plt.title('T2_Corr')
grid on
xlim([0, 2000])
ylim([0, 2000])
xlabel('Reference T2 (ms)')
ylabel('Estimated T2 (ms)')
legend('Estimation', 'Reference', 'location', 'best')
%}
%% show SNR and PSNR
% 
% oracle = [Output.PSNR.T1.old_mrf, Output.PSNR.T2.old_mrf;...
%             Output.SNR.T1.old_mrf, Output.SNR.T2.old_mrf; ...
%             Output.RMSE.T1.old_mrf, Output.RMSE.T2.old_mrf; ...
%             Output.coeff.T1.old_mrf, Output.coeff.T2.old_mrf;...
% ]
% 
% FLOR = [Output.PSNR.T1.flor, Output.PSNR.T2.flor;...
%             Output.SNR.T1.flor, Output.SNR.T2.flor; ...
%             Output.RMSE.T1.flor, Output.RMSE.T2.flor; ...
%             Output.coeff.T1.flor, Output.coeff.T2.flor;...
% ]







