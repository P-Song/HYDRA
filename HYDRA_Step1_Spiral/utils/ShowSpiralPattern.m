
% draw spiral sampling patterns

addpath(genpath('utils') );

%% undersample the data in Fourier domain 

N = 128; %The image size is NxN

% Generate spiral sampling trajectories
num_of_TRs = 200;
%Magnet parameters
smax=17000;                                 %(=170 mT/m/ms in Siemens)
gmax=4;                                         %(=40 mT/m)
T=0.00001;
%for inner region (10x10 central k-space)
inner_region_size= 20; % 20; % 40
N_inner=1;                                      %Number of interleaves
N_outer=48; % 25;
N_pixels=N;                                    %128x128 slice is obtained
FOV= 24; % 24;
[sampling_locations_spiral, num_samples_inner, num_samples_outer]=spiral_trajectories_fisp(num_of_TRs,smax,gmax,T,inner_region_size,N_inner,N_outer,N_pixels,FOV);
[sampling_locations_spiral, samples_locations_complex]=normalize_sampling_locations(sampling_locations_spiral,N_pixels);

% Sample the data with spiral trajectories
Y_spiral = sample_k_spaces(sampling_locations_spiral,reshape(X,N,N,L));

%%
for i = 1 % 1: 24
    TempSamp = sampling_locations_spiral{i};
    figure(110); 
    plot(TempSamp(:,1), TempSamp(:,2), '-b', 'LineWidth', 1)
    xlim([-0.5,0.5]); ylim([-0.5,0.5])
    grid on; grid minor
    set(gcf, 'position', [100,100,400,400])
    pause(0.1)
    set(gcf, 'position', [100,100,400,400])
    print(['spiral_', num2str(i),'.png'], '-dpng');
end


disp('done!')

