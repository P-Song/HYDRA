
% 01/02/2019 show Gaussian sampling patterns.

% Reference:
% ----------------------------
% [1] Pingfan Song, Yonina C. Eldar, Gal Mazor, Miguel R. D. Rodrigues, "HYDRA: Hybrid Deep Magnetic Resonance Fingerprinting", Medical Physics, 2019, doi: 10.1002/mp.13727. 
% [2] Pingfan Song, Yonina C. Eldar, Gal Mazor, Miguel R. D. Rodrigues, "Magnetic Resonance Fingerprinting Using a Residual Convolutional Neural Network", ICASSP, pp. 1040-1044. IEEE, 2019.
% [3] Gal Mazor, Lior Weizman, Assaf Tal, Yonina C. Eldar. "Low‐rank magnetic resonance fingerprinting." Medical physics 45, no. 9 (2018): 4066-4084.

%% Initialize:
clear;
% close all;
% addpath('Bloch\');
addpath(genpath('utils') );
N= 128; %128;                                                               %The image size is NxN
L = 200; % 500;                                                             %The sequence length
k_space_undersampling_ratio=0.15;             %undersampling ratio in k-space
% k_space_undersampling_ratio=0.7;             %undersampling ratio in k-space

Params.N = N;
Params.L = L;
Params.k_space_undersampling_ratio = k_space_undersampling_ratio;

%% build dictionary using MRF FISP sequence, omit entries where T2>T1

sampling_matrix = genrate_binary_sampling_map(N,k_space_undersampling_ratio,L); %generates binary sampling masks
sampling_matrix = reshape(sampling_matrix,N,N,L);

for j=1:size(sampling_matrix,3)
    figure(121)
    imagesc(ifftshift(sampling_matrix(:,:,j)))
    colormap gray
    axis off
    set(gcf,'Position', [10,10,256,256]);
    set(gca,'Position', [0,0,1,1]);
%     export_fig(gcf, ['GausPattern_', num2str(j), '.png'], '-nocrop', '-transparent')  
    pause(0.01)
end


disp('done!')















