
% This code generate simulated dictionary and look-up-table as training, validation, and testing data.
% T1, 1: 10: 5000 ms, T2, 1:10: 2000 ms,

% Reference:
% ----------------------------
% [1] Pingfan Song, Yonina C. Eldar, Gal Mazor, Miguel R. D. Rodrigues, "HYDRA: Hybrid Deep Magnetic Resonance Fingerprinting", Medical Physics, 2019, doi: 10.1002/mp.13727. 
% [2] Pingfan Song, Yonina C. Eldar, Gal Mazor, Miguel R. D. Rodrigues, "Magnetic Resonance Fingerprinting Using a Residual Convolutional Neural Network", ICASSP, pp. 1040-1044. IEEE, 2019.
% [3] Gal Mazor, Lior Weizman, Assaf Tal, Yonina C. Eldar. "Low‐rank magnetic resonance fingerprinting." Medical physics 45, no. 9 (2018): 4066-4084.


%% Initialize:
clear;
addpath(genpath('utils') );
L = 1000;        %The sequence length

% % training dataset
% T1_values = [1:10:5000];
% T2_values = [1:10:2000];
% T1_labels = {'1:10:5000 ms'};
% T2_labels = {'1:10:2000 ms'};

% % validation dataset
% T1_values=[5:10:5000];
% T2_values=[5:10:2000];
% T1_labels = {'5:10:5000 ms'};
% T2_labels = {'5:10:2000 ms'};

% % testing dataset
% T1_values=[3:10:5000];
% T2_values=[3:10:2000];
% T1_labels = {'3:10:5000 ms'};
% T2_labels = {'3:10:2000 ms'};

% random testing dataset
T1_values = randperm(5000,500);
T1_values = sort(T1_values,'ascend');
T2_values = randperm(2000,200);
T2_values = sort(T2_values,'ascend');
T1_labels = {'random 1:5000 ms'};
T2_labels = {'random 1:2000 ms'};

% % demo dataset
% T1_values = [400, 600, 800, 1000]; % 800;
% T1_labels = {'400 ms', '600 ms', '800 ms', '1000 ms'};
% T2_values = 80 ; % [40, 60, 80, 100] % 80;
% T2_labels = {'40 ms', '60 ms' ,'80 ms', '100 ms'};

[RFpulses, TR]=generate_RF_TR(L);            %Load slowly changing RF and TR values
RFpulses = RFpulses*1i;                                %to avoid complex values of X and D
TE = 10;%2
RFpulses = RFpulses(1:round(1000/L):1000); % undersampling in time dimension.
TR = TR(1:round(1000/L):1000);

Params.L = L;
Params.RFpulses = RFpulses;
Params.TR = TR;
Params.TE = TE;
Params.T1_values = T1_values;
Params.T2_values = T2_values;
Params.T1_labels = T1_labels;
Params.T2_labels = T2_labels;

%% build dictionary using MRF FISP sequence, omit entries where T2>T1
Len = ['_L', num2str(L) ];
TEtime = ['_TE', num2str(TE) ];
InputDataFile = ['D_LUT', Len, TEtime, '.mat'];

disp('Building the dictionary...');
[FISP_dictionary,LUT] = build_dictionary_fisp(L,RFpulses,TR,TE, T1_values, T2_values);                     %build the MRF FISP dictionary and its matching look up table
D = single(FISP_dictionary);
clear FISP_dictionary;
LUT = LUT*1000;                                                                                                        %change units
disp('The dictionary is ready.');
save(InputDataFile, 'D', 'LUT', 'Params')


%% T1 varies, T2 fixed
figure; 
plot(NaN, NaN, 'w.'); hold on % legend space
plot(abs(D'))
plot(NaN, NaN, 'w.'); hold on cl% % legend space
legend({'T1:',T1_labels{:}, 'T2: 80 ms'}, 'Orientation','Horizontal')
xlabel('Time Evolution (ms)')
ylabel('Signal Intensity')
set(gcf, 'position', [100,100,800,300])
% savefig(gcf, 'T2_80ms_Frames10000')
print(gcf,'T2_80ms_Frames10000','-dpng','-r300'); % 300 dpi


%% T2 varies, T1 fixed
figure; 
plot(NaN, NaN, 'w.'); hold on % legend space
plot(abs(D'))
plot(NaN, NaN, 'w.'); hold on % % legend space
legend({'T2:',T2_labels{:}, 'T1: 800 ms'}, 'Orientation','Horizontal')
xlabel('Time Evolution (ms)')
ylabel('Signal Intensity')
set(gcf, 'position', [100,100,800,300])
% savefig(gcf, 'T1_800ms_Frames10000')
print(gcf,'T1_800ms_Frames10000','-dpng','-r300'); % 300 dpi


