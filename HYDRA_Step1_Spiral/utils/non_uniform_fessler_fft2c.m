function [ vec_of_samples]=non_uniform_fessler_fft2c(image,locations,mask)
J = [6 6];
N=size(image,1);
N2=[N N];
ig=image_geom('nx',N,'ny',N,'dx',1,'offsets','dsp');
if ~isvar('mask')
    mask=ig.circ(1+ig.nx/2, 1+ig.ny/2) > 0;
end
ig.mask=mask;
nufft_args = {N2, J, 2*N2, N2/2, 'table', 2^10, 'minmax:kb'};
omega=fliplr(locations)*2*pi;
Gn = Gnufft(ig.mask, {omega, nufft_args{:}});
vec_of_samples=Gn*image;