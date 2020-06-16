% This code is to perform the first step of HYDRA: 
% reconstruct temporal signatures using low-rank prior and adapted FLOR
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


% 19/10/2018 Save memory by performing inner product one sample by one
% sample in build_maps_SaveMem.m, florAlg_SaveMem, find_E_SaveMem.
% Make sure that one can also use Python matlab engine to run the code.
% 22/06/2018 build_maps directly from temporal MRF data X, referring D and
% LUT. Perform inner product one sample by one sample to save memory


function [Output] = HYDRA_Step1(flag)

%% Initialize:

addpath(genpath('utils') );

if nargin < 1
    flag = 0 ;
end
Output.flag = flag;


N= 128; %128;                                                               %The image size is NxN
L = 200; % 1000;                                                             %The sequence length
k_space_undersampling_ratio=0.15;             %undersampling ratio in k-space

[RFpulses, TR]=generate_RF_TR(1000);            %Load slowly changing RF and TR values
RFpulses = RFpulses*1i;                                %to avoid complex values of X and D
TE = 10;%2

RFpulses = RFpulses(1:round(1000/L):1000); % undersampling in time dimension.
TR = TR(1:round(1000/L):1000);

T1_values = [1:10:5000];
T2_values = [1:10:2000];
T1_labels = {'1:10:5000 ms'};
T2_labels = {'1:10:2000 ms'};

Params.N = N;
Params.L = L;
Params.k_space_undersampling_ratio = k_space_undersampling_ratio;
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
Size = ['_N', num2str(N)];
Len = ['_L', num2str(L) ];
TEtime = ['_TE', num2str(TE) ];
FileName = [ 'FLOR', Size, Len, TEtime, '_NoDictProj', '.mat'];

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

%     X_fullysamp = reshape(X_fullysamp,N,N*L);
%     sampling_matrix = genrate_binary_sampling_map(N,k_space_undersampling_ratio,L); %generates binary sampling masks
%     Y_fullysamp = fft_mats(X_fullysamp,1);
%     % kSpaceNoise = reshape([1 1i]*0.5*randn(2,L*N^2),N,L*N); 
%     kSpaceNoise = reshape([1 1i]*0*randn(2,L*N^2),N,L*N); % no noise
%     disp('Adding noise to the data in Fourier domain');
%     Y_subsamp = Y_fullysamp + kSpaceNoise;                                                                                                         %add noise to the data, complex white noise with sigma=0.5
%     disp('Under-sampling the noisy data in Fourier domain');
%     Y_subsamp = sampling_matrix.*Y_subsamp;  

	save(InputDataFile, 'D', 'X_fullysamp', 'LUT', 'Params')
end                                                                                                      %uder-sampling the noised data
%}

% % If there is available D and LUT, load them.
load('../D_LUT_L1000_TE10_Train.mat');
load('../MRF_ImageStack_N128_L1000_TE10_Ratio0.15.mat', 'X_fullysamp')

%%
D = D(1:1:end,1:round(1000/L):end) ; % sub-sampled from 1000 time points;
LUT = LUT(1:1:end,:) ; 
X = X_fullysamp(:,:,1:round(1000/L):end) ; % sub-sampled from 1000 time points;

% Y_full = Y_fullysamp(:,:,2:5:end) ; % Note: bug! Y_fullysamp does not correspond to X.

X = reshape(X,N,N*L);

sampling_matrix = genrate_binary_sampling_map(N,k_space_undersampling_ratio,L); %generates binary sampling masks
Y_full = fft_mats(X,1);
% kSpaceNoise =  reshape([1 1i]*0.5*randn(2,L*N^2),N,L*N);     
kSpaceNoise = 0;
disp('Adding noise to the data in Fourier domain');
Y = Y_full + kSpaceNoise;                                                                                                         %add noise to the data, complex white noise with sigma=0.5
disp('Under-sampling the noisy data in Fourier domain');
Y = sampling_matrix.*Y;    


%% adapted FLOR algorithm for reconstructing temporal signatures.
disp('Calculating the parameter maps using adapted FLOR');
r = L;
th = 5;

% % fast, but consume large memory
% [ X_estimated_flor, E_estimated_flor] = florAlg( Y,D,N,L,X,r,th );
% [T1_flor,T2_flor,PD_flor] = build_maps_from_E(E_estimated_flor,LUT); % fast, large memory

% % slow, but save memory
tic;
% X_estimated_flor = florAlg_SaveMem( Y,D,N,L,X,r,th );
X_estimated_flor = florAlg_SaveMem_NoDictProj( Y,N,L,X,r,th );
Threshold = 8; % 20;
X_estimated_flor = reshape(X_estimated_flor,N*N,L); 
NormX = sum(real(X_estimated_flor).^2,2); 
X_estimated_flor(NormX < Threshold, :) = 0; % too small signatures will be set to 0;

X_estimated_flor = reshape(X_estimated_flor,N,N,L); 
TimeCost_flor = toc;

XFileName = [ 'X_Est_FLOR_Gauss', Size, Len, TEtime, Ratio, '.mat'];
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

[T1_flor,T2_flor,PD_flor] = build_maps_SaveMem(X_estimated_flor, D, LUT); % slow, save memory

%% Show Results in the region of interest
T1.orig = T1_128;                                                
T2.orig = T2_128;                                                      

T1.flor = T1_flor.*(T1.orig~=0);                           
T2.flor = T2_flor.*(T2.orig~=0);                                

% showResults( k_space_undersampling_ratio,T1,T2,PD );

%% show PSNR and SSIM
PSNR = [];  SNR = [];
[PSNR.T1.flor SNR.T1.flor] = psnr( T1.flor , T1.orig, T1max) ;
[PSNR.T2.flor SNR.T2.flor] = psnr( T2.flor , T2.orig, T2max ) ;

RMSE.T1.flor = sqrt(mean((T1.flor(:) - T1.orig(:)).^2));  % Root Mean Squared Error
RMSE.T2.flor = sqrt(mean((T2.flor(:) - T2.orig(:)).^2));  % Root Mean Squared Error

coeff.T1.flor = corrcoef(T1.orig(:),T1.flor(:)) ;
coeff.T1.flor = coeff.T1.flor(1,2);
coeff.T2.flor = corrcoef(T2.orig(:),T2.flor(:)) ;
coeff.T2.flor = coeff.T2.flor(1,2);

%% save output
Output.PSNR = PSNR;
Output.SNR = SNR;
Output.RMSE = RMSE;
Output.coeff = coeff;
Output.T1 = T1;
Output.T2 = T2;
Output.TimeCost_flor = TimeCost_flor;
Output.X_estimated_flor = X_estimated_flor;
save(FileName, 'Params', 'Output')
disp('done!')

%% show figures;
%{
figure; 
imagesc([Output.T1.orig, Output.T1.basic_mrf, Output.T1.flor]); 
colormap jet

figure; 
imagesc([Output.T2.orig, Output.T2.basic_mrf, Output.T2.flor]); 
colormap jet

figure; 
imagesc([Output.PD.orig, Output.PD.basic_mrf, Output.PD.flor]); 
colormap jet

%% show reconstructed T1, T2 parameter maps
FigFormat = ['.png']; % ['.png'] ; % ['.jpg'] ; % ['.eps'];
save_figure = 1;

cmin = 0; cmax = T1max; % min and max color value threshold;
figure; 
imagesc(Output.T1.old_mrf); 
set(gcf, 'position', [100,100,256,256]); axis off
set(gca, 'Position', [0,0,1,1]);
colormap jet; caxis([cmin, cmax]) ;
% c = colorbar ; c.LineWidth = 0.5; c.FontSize = 11; c.Position = [1/1.2, 0.02, 0.05, 0.96]; % [left, bottom, width, height]
if save_figure
    FileName = ['T1_MRF_DictMatch'];
    export_fig(gcf, [FileName, FigFormat], '-nocrop', '-transparent')  % export_fig(figure_handle, filename);		
%     savefig(gcf, FileName)
end
pause(0.2)

cmin = 0; cmax = T2max; % min and max color value threshold;
figure; 
imagesc(Output.T2.old_mrf); 
set(gcf, 'position', [300,100,256,256]); axis off
set(gca, 'Position', [0,0,1,1]);
colormap jet; caxis([cmin, cmax]) ;
% c = colorbar ; c.LineWidth = 0.5; c.FontSize = 11; c.Position = [1/1.2, 0.02, 0.05, 0.96]; % [left, bottom, width, height]
if save_figure
    FileName = ['T2_MRF_DictMatch'];
    export_fig(gcf, [FileName, FigFormat], '-nocrop', '-transparent')  % export_fig(figure_handle, filename);		
%     savefig(gcf, FileName)
end
pause(0.2)

%% residual
cmin = 0; cmax = 20; % min and max color value threshold;
figure; 
imagesc(abs(Output.T2.old_mrf - Output.T2.orig)); 
set(gcf, 'position', [100,300,256*1.2,256]); axis off
set(gca, 'Position', [0,0,1/1.2,1]);
colormap jet; caxis([cmin, cmax]) ;
c = colorbar ; c.LineWidth = 0.5; c.FontSize = 8; c.Position = [1/1.15, 0.02, 0.05, 0.96]; % [left, bottom, width, height]
if save_figure
    FileName = ['T1_MRF_DictMatch_Res'];
    export_fig(gcf, [FileName, FigFormat], '-nocrop', '-transparent')  % export_fig(figure_handle, filename);		
%     savefig(gcf, FileName)
end
pause(0.2)

cmin = 0; cmax = 10; % min and max color value threshold;
figure; 
imagesc(abs(Output.T2.old_mrf - Output.T2.orig)); 
set(gcf, 'position', [300,300,256*1.2,256]); axis off
set(gca, 'Position', [0,0,1/1.2,1]);
colormap jet; caxis([cmin, cmax]) ;
c = colorbar ; c.LineWidth = 0.5; c.FontSize = 8; c.Position = [1/1.15, 0.02, 0.05, 0.96]; % [left, bottom, width, height]
if save_figure
    FileName = ['T2_MRF_DictMatch_Res'];
    export_fig(gcf, [FileName, FigFormat], '-nocrop', '-transparent')  % export_fig(figure_handle, filename);		
%     savefig(gcf, FileName)
end
pause(0.2)

%% show correlation curve
figure; 
plot(Output.T1.orig(:),Output.T1.old_mrf(:),'r.', 'MarkerSize', 15); hold on
plot(Output.T1.orig(:),Output.T1.orig(:),'b-')
% plt.title('T1_Corr')
grid on; 
xlim([0, 5000])
ylim([0, 5000])
xlabel('Reference T1 (ms)')
ylabel('Estimated T1 (ms)')
legend('Estimation', 'Reference', 'location', 'best')
set(gcf, 'position', [100,300,500,300]);
if save_figure
    FileName = ['T1_Corr_DictMatch'];
    export_fig(gcf, [FileName, FigFormat], '-nocrop', '-transparent')  % export_fig(figure_handle, filename);		
%     savefig(gcf, FileName)
end
pause(0.2)

figure; 
plot(Output.T2.orig(:),Output.T2.old_mrf(:),'r.', 'MarkerSize', 15); hold on
plot(Output.T2.orig(:),Output.T2.orig(:),'b-')
% plt.title('T2_Corr')
grid on;
xlim([0, 2000])
ylim([0, 2000])
xlabel('Reference T2 (ms)')
ylabel('Estimated T2 (ms)')
legend('Estimation', 'Reference', 'location', 'best')
set(gcf, 'position', [300,300,500,300]);
if save_figure
    FileName = ['T2_Corr_DictMatch'];
    export_fig(gcf, [FileName, FigFormat], '-nocrop', '-transparent')  % export_fig(figure_handle, filename);		
%     savefig(gcf, FileName)
end
pause(0.2)
%}


%% show SNR and PSNR
% PSNR_T1 = [Output.PSNR.T1.old_mrf, Output.PSNR.T1.basic_mrf, Output.PSNR.T1.flor]' ;
% PSNR_T2 = [Output.PSNR.T2.old_mrf, Output.PSNR.T2.basic_mrf, Output.PSNR.T2.flor]' ;
% PSNR_PD = [Output.PSNR.PD.old_mrf, Output.PSNR.PD.basic_mrf, Output.PSNR.PD.flor]' ;
% 
% SNR_T1 = [Output.SNR.T1.old_mrf, Output.SNR.T1.basic_mrf, Output.SNR.T1.flor]' ;
% SNR_T2 = [Output.SNR.T2.old_mrf, Output.SNR.T2.basic_mrf, Output.SNR.T2.flor]' ;
% SNR_PD = [Output.SNR.PD.old_mrf, Output.SNR.PD.basic_mrf, Output.SNR.PD.flor]' ;
% 
% RMSE_T1 = [Output.RMSE.T1.old_mrf, Output.RMSE.T1.basic_mrf, Output.RMSE.T1.flor]' ;
% RMSE_T2 = [Output.RMSE.T2.old_mrf, Output.RMSE.T2.basic_mrf, Output.RMSE.T2.flor]' ;
% RMSE_PD = [Output.RMSE.PD.old_mrf, Output.RMSE.PD.basic_mrf, Output.RMSE.PD.flor]' ;
% 
% coeff_T1 = [Output.coeff.T1.old_mrf, Output.coeff.T1.basic_mrf, Output.coeff.T1.flor]' ;
% coeff_T2 = [Output.coeff.T2.old_mrf, Output.coeff.T2.basic_mrf, Output.coeff.T2.flor]' ;
% coeff_PD = [Output.coeff.PD.old_mrf, Output.coeff.PD.basic_mrf, Output.coeff.PD.flor]' ;
% 
% Results = [SNR_T1, SNR_T2, SNR_PD, PSNR_T1, PSNR_T2, PSNR_PD, ...
%     RMSE_T1, RMSE_T2, RMSE_PD, coeff_T1, coeff_T2, coeff_PD]




