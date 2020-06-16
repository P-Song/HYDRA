function [samples_locations, samples_locations_complex]=normalize_sampling_locations(samples_locations,N)

samples_locations_complex=cell(1,length(samples_locations));
for i=1:length(samples_locations)
    x=real(samples_locations{i});
    y=imag(samples_locations{i});
    maxx=N;%max(abs(x));
    maxy=N;%max(abs(y));
    x=(x-N/2)/(maxx);
    y=(y-N/2)/(maxy);
    file=zeros(length(x),2);
    file(:,1)=x;
    file(:,2)=y;
    file2=file(:,1)+1i*file(:,2);
    file2=unique(file2,'stable');
    file=zeros(length(file2),2);
    file(:,1)=real(file2);
    file(:,2)=imag(file2);
    samples_locations{i}=file;
    samples_locations_complex{i}=file(:,1)+1i*file(:,2);
end

