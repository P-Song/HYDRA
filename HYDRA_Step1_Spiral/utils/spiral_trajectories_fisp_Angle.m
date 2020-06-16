function [all_sampling_locations, num_samples_inner, num_samples_outer]=spiral_trajectories_fisp(num_of_TRs,smax,gmax,T,inner_region_size,N_inner,N_outer,N_pixels,FOV)

radius_ratio=inner_region_size/N_pixels;
rmax1=0.5*(N_pixels/FOV)*radius_ratio;
F0=FOV; %(FOV in cm);
F1=0*FOV;
F2=0;
k_inner=vds(smax,gmax,T,N_inner,[F0,F1,F2],0,rmax1);
num_samples_inner=length(k_inner);

%for outer region
F0=FOV; %(FOV in cm);
F1=0*FOV;
F2=0;
rmax=0.5*(N_pixels/FOV);
[k_outer_single_interleave g,s,time,r]=vds(smax,gmax,T,N_outer,[F0,F1,F2],rmax1,rmax);
num_samples_outer=length(k_outer_single_interleave);
% rot_angle=2*pi/N_outer;

N_outer2 = 96; % set the period by spf
rot_angle=2*pi/N_outer2; % better angle by spf
angles_to_rotate=0:rot_angle:(rot_angle*(N_outer2-1));
vec_to_mul_angles=exp(1i*angles_to_rotate)';

single_interleave=[k_inner k_outer_single_interleave];

all_outer_trajectories_values=repmat(vec_to_mul_angles,1,length(single_interleave)).*repmat(single_interleave,N_outer2,1);

all_outer_trajectories_values(:,1:size(k_inner,2))=((all_outer_trajectories_values(:,1:size(k_inner,2))*F0))+(N_pixels+1)/2+((N_pixels+1)/2)*1i;
all_outer_trajectories_values(:,size(k_inner,2)+1:end)=((all_outer_trajectories_values(:,size(k_inner,2)+1:end)/rmax)+(1+1i))*((N_pixels-1)/2)+(1+1i);


num_of_reps=ceil(num_of_TRs/N_outer2);

all_sampling_locations=repmat(all_outer_trajectories_values,num_of_reps,1);
all_sampling_locations=mat2cell(all_sampling_locations(1:num_of_TRs,:),ones(1,num_of_TRs),size(all_sampling_locations,2))';

% i=2; figure; plot(real(all_sampling_locations(i,:)),imag(all_sampling_locations(i,:)))






