%% show colorbar only
addpath(genpath('utils') );

% horizontal colorbar
figure;
imagesc(255*ones(100,100))
pause(0.2)
colormap(jet) ; 
cmin = 0; cmax = 20;
caxis([cmin, cmax]) ;
axis off
set(gcf,'Position', [10,10,256,40]);
set(gca,'Position', [0.1,0.1,0,0]);
c = colorbar ;
% c.Limits = [0 10];
c.Location = 'south' ;
c.Position=[0.01 0. 0.96 0.4] ;
c.FontSize = 9;
export_fig(gcf, ['H_colorbar_20', '.fig'], '-nocrop', '-transparent')  
export_fig(gcf, ['H_colorbar_20', '.png'], '-nocrop', '-transparent')

%% vertical colorbar
figure;
imagesc(255*ones(100,100))
pause(0.2)
colormap(jet) ; 
cmin = 0; cmax = 20;
caxis([cmin, cmax]) ;
axis off
set(gcf,'Position', [10,10,40,256]);
set(gca,'Position', [0,0.1,0,0]);
c = colorbar ;
% c.Limits = [0 10];
c.Location = 'east'; % 'south' ;
c.Position=[0.0 0.01 0.4 0.96] ;
c.FontSize = 9;
export_fig(gcf, ['V_colorbar_20', '.fig'], '-nocrop', '-transparent')  
export_fig(gcf, ['V_colorbar_20', '.png'], '-nocrop', '-transparent')
