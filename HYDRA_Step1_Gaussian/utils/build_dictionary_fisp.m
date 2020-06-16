function [FISP_dictionary,LUT]=build_dictionary_fisp(L,RFpulses,TR,TE,T1_values,T2_values)
%build the MRF FISP dictionary and its matching look up table (LUT)
rng(1);
if nargin<6
    T2_values=[20:5:100,110:10:200,300:200:1900];
end
if nargin<5
    T1_values=[100:20:2000,2300:300:5000];
end
if nargin<4
    TE=2;
end
if nargin<3
    TR=rand(1,1000)*4+10;
end
if nargin<2
    RFpulses=10*pi./180*randn(1,1000);
end
if nargin<2
    L=1000; %Number of random imaging contrasts
end

FISP_dictionary=zeros(length(T1_values)*length(T2_values),L,'single');
k=0;
LUT=zeros(length(T1_values)*length(T2_values),2,'single');
T1_values=T1_values/1000;
T2_values=T2_values/1000;
TR=TR/1000;
TE=TE/1000;
for i=1:length(T1_values)  
    for j=1:length(T2_values)
        if T1_values(i)<T2_values(j)
            continue
        end
        k=k+1;
        LUT(k,:)=[T1_values(i),T2_values(j)];
        FISP_dictionary(k,:)=(epg_fisp_mrf(RFpulses ,TR ,TE,T1_values(i),T2_values(j)));
    end
end
FISP_dictionary=FISP_dictionary(1:k,:);
LUT=LUT(1:k,:);


