function [  ] = showResults( k_space_undersampling_ratio,T1,T2,PD )
%Compare results between methods: T1, T2 and PD maps

%T1
figure('Position',[0 0 1000 700]);
h = subplot(3,3,1);ax=get(h,'Position');ax(1) = ax(1)+0.00;set(h,'position',ax); imagesc(T1.orig,[0 4000]);title('T1 Gold standard','FontSize',12.5);axis off;
h = subplot(3,3,2);ax=get(h,'Position');ax(1) = ax(1)-0.01; set(h,'position',ax); imagesc(T1.old_mrf,[0 4000]);axis off;title(['Fully sampled MRF'],'FontSize',12.5);
h = subplot(3,3,3);ax=get(h,'Position');ax(3) = ax(3)+0.075;ax(1) = ax(1)-0.073; set(h,'position',ax);imagesc(T1.flor,[0 4000]);axis off;colorbar;set(gca,'fontsize',12.5);title(['FLOR ' ,num2str(k_space_undersampling_ratio*100), '% sampled'],'FontSize',12.5);

%T2
h = subplot(3,3,4); ax=get(h,'Position');ax(1) = ax(1)+0.00;set(h,'position',ax);imagesc(T2.orig,[0 1500]);title('T2 Gold standard','FontSize',12.5);axis off;
h = subplot(3,3,5);ax=get(h,'Position');ax(1) = ax(1)-0.01; set(h,'position',ax); imagesc(T2.old_mrf,[0 1500]);axis off;
h = subplot(3,3,6);ax=get(h,'Position');ax(3) = ax(3)+0.075;ax(1) = ax(1)-0.073; set(h,'position',ax); imagesc(T2.flor,[0 1500]);axis off;colorbar;set(gca,'fontsize',12.5);

%PD
h = subplot(3,3,7); ax=get(h,'Position');ax(1) = ax(1)+0.00;set(h,'position',ax); imagesc(PD.orig,[0 120]);title('PD Gold standard','FontSize',12.5);axis off;
h = subplot(3,3,8);ax=get(h,'Position');ax(1) = ax(1)-0.01; set(h,'position',ax); imagesc(PD.old_mrf,[0 120]);axis off;
h = subplot(3,3,9);ax=get(h,'Position');ax(3) = ax(3)+0.075;ax(1) = ax(1)-0.073; set(h,'position',ax); imagesc(PD.flor,[0 120]);axis off;colorbar;set(gca,'fontsize',12.5);

figure('Position',[700 0 700 700]);
mse_old_mrf = 1 - goodnessOfFit(T1.old_mrf(:),T1.orig(:),'NMSE');
h = subplot(3,2,1);ax=get(h,'Position');ax(1) = ax(1); set(h,'position',ax);imagesc(abs(T1.old_mrf-T1.orig),[0 500]);axis off;title('Fully Sampled MRF','FontSize',12.5);
text(-20,55,'T1','FontSize',15);text(20,140,['NMSE=',num2str(mse_old_mrf)],'FontSize',12.5);

mse_cs = 1 - goodnessOfFit(T1.flor(:),T1.orig(:),'NMSE');
h = subplot(3,2,2);ax=get(h,'Position');ax(3) = ax(3)+0.1;ax(1) = ax(1)-0.098; set(h,'position',ax); imagesc(abs(T1.flor-T1.orig),[0 500]);axis off;colorbar;set(gca,'fontsize',12.5);
title('FLOR','FontSize',20);text(20,140,['NMSE=',num2str(mse_cs)],'FontSize',12.5);

%Error T2
mse_old_mrf = 1 - goodnessOfFit(T2.old_mrf(:),T2.orig(:),'NMSE');
h = subplot(3,2,3);ax=get(h,'Position');ax(1) = ax(1); set(h,'position',ax);imagesc(abs(T2.old_mrf-T2.orig),[0 300]);axis off;
text(-20,55,'T2','FontSize',15);text(20,140,['NMSE=',num2str(mse_old_mrf)],'FontSize',12.5);

mse_cs = 1 - goodnessOfFit(T2.flor(:),T2.orig(:),'NMSE');
h = subplot(3,2,4);ax=get(h,'Position');ax(3) = ax(3)+0.1;ax(1) = ax(1)-0.098; set(h,'position',ax);  imagesc(abs(T2.flor-T2.orig),[0 300]);axis off;colorbar;set(gca,'fontsize',12.5);
text(20,140,['NMSE=',num2str(mse_cs)],'FontSize',12.5);

%Error PD
mse_old_mrf = 1 - goodnessOfFit(PD.old_mrf(:),PD.orig(:),'NMSE');
h = subplot(3,2,5);ax=get(h,'Position');ax(1) = ax(1); set(h,'position',ax);imagesc(abs(PD.old_mrf-PD.orig),[0 30]);axis off;
text(-22,55,'PD','FontSize',15);text(20,140,['NMSE=',num2str(mse_old_mrf)],'FontSize',12.5);

% mse_cs=mse(PD.flor.*(PD.orig~=0),PD.orig);
mse_cs = 1 - goodnessOfFit(PD.flor(:),PD.orig(:),'NMSE');
h = subplot(3,2,6);ax=get(h,'Position');ax(3) = ax(3)+0.1;ax(1) = ax(1)-0.098; set(h,'position',ax);  imagesc(abs(PD.flor-PD.orig),[0 30]);axis off;colorbar;set(gca,'fontsize',12.5);
text(20,140,['NMSE=',num2str(mse_cs)],'FontSize',12.5);
suptitle('Error Maps');
end

