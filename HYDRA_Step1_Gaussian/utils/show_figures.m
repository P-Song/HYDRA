% 07/11/2018 show figures

% Reference:
% ----------------------------
% [1] Pingfan Song, Yonina C. Eldar, Gal Mazor, Miguel R. D. Rodrigues, "HYDRA: Hybrid Deep Magnetic Resonance Fingerprinting", Medical Physics, 2019, doi: 10.1002/mp.13727. 
% [2] Pingfan Song, Yonina C. Eldar, Gal Mazor, Miguel R. D. Rodrigues, "Magnetic Resonance Fingerprinting Using a Residual Convolutional Neural Network", ICASSP, pp. 1040-1044. IEEE, 2019.
% [3] Gal Mazor, Lior Weizman, Assaf Tal, Yonina C. Eldar. "Low‐rank magnetic resonance fingerprinting." Medical physics 45, no. 9 (2018): 4066-4084.


addpath(genpath('utils') );
N= 128; %128;                                                               %The image size is NxN
L = 200; % 500;                                                             %The sequence length
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


T1max = 4500 ;
T2max = 2500;
PDmax = 117 ;


%% show reconstructed T1, T2 parameter maps with subsampling
FigFormat = ['.png']; % ['.png'] ; % ['.jpg'] ; % ['.eps'];
save_figure = 0;

cmin = 0; cmax = T1max; % min and max color value threshold;
figure; 
imagesc(Output.T1.basic_mrf); 
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
imagesc(Output.T2.basic_mrf); 
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
cmin = 0; cmax = 200; % min and max color value threshold;
figure; 
imagesc(abs(Output.T2.basic_mrf - Output.T2.orig)); 
set(gcf, 'position', [100,300,256*1.2,256]); axis off
set(gca, 'Position', [0,0,1/1.2,1]);
colormap jet; caxis([cmin, cmax]) ;
c = colorbar ; c.LineWidth = 0.5; c.FontSize = 8; c.Position = [1/1.15, 0.02, 0.05, 0.96]; % [left, bottom, width, height]
if save_figure
    FileName = ['T1_res_MRF_DictMatch'];
    export_fig(gcf, [FileName, FigFormat], '-nocrop', '-transparent')  % export_fig(figure_handle, filename);		
%     savefig(gcf, FileName)
end
pause(0.2)

cmin = 0; cmax = 100; % min and max color value threshold;
figure; 
imagesc(abs(Output.T2.basic_mrf - Output.T2.orig)); 
set(gcf, 'position', [300,300,256*1.2,256]); axis off
set(gca, 'Position', [0,0,1/1.2,1]);
colormap jet; caxis([cmin, cmax]) ;
c = colorbar ; c.LineWidth = 0.5; c.FontSize = 8; c.Position = [1/1.15, 0.02, 0.05, 0.96]; % [left, bottom, width, height]
if save_figure
    FileName = ['T2_res_MRF_DictMatch'];
    export_fig(gcf, [FileName, FigFormat], '-nocrop', '-transparent')  % export_fig(figure_handle, filename);		
%     savefig(gcf, FileName)
end
pause(0.2)

%% show correlation curve
figure; 
plot(Output.T1.orig(:),Output.T1.basic_mrf(:),'r.', 'MarkerSize', 15); hold on
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
    FileName = ['T1_Corr_MRF_DictMatch'];
    export_fig(gcf, [FileName, FigFormat], '-nocrop', '-transparent')  % export_fig(figure_handle, filename);		
%     savefig(gcf, FileName)
end
pause(0.2)

figure; 
plot(Output.T2.orig(:),Output.T2.basic_mrf(:),'r.', 'MarkerSize', 15); hold on
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
    FileName = ['T2_Corr_MRF_DictMatch'];
    export_fig(gcf, [FileName, FigFormat], '-nocrop', '-transparent')  % export_fig(figure_handle, filename);		
%     savefig(gcf, FileName)
end
pause(0.2)




%% show reconstructed T1, T2 parameter maps without subsampling
FigFormat = ['.png']; % ['.png'] ; % ['.jpg'] ; % ['.eps'];
save_figure = 0;

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
    FileName = ['T1_res_MRF_DictMatch'];
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
    FileName = ['T2_res_MRF_DictMatch'];
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
    FileName = ['T1_Corr_MRF_DictMatch'];
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
    FileName = ['T2_Corr_MRF_DictMatch'];
    export_fig(gcf, [FileName, FigFormat], '-nocrop', '-transparent')  % export_fig(figure_handle, filename);		
%     savefig(gcf, FileName)
end
pause(0.2)

%%
% T1_basic_mrf = Output.T1.basic_mrf; 
% T1_flor = Output.T1.flor ;
% T1_old_mrf = Output.T1.old_mrf ;
% 
% T2_basic_mrf = Output.T2.basic_mrf; 
% T2_flor = Output.T2.flor ;
% T2_old_mrf = Output.T2.old_mrf ;



