function [ image ] = non_uniform_fessler_ifft2c(fft_values,N,mask,mask_locs)
ig=image_geom('nx',N,'ny',N,'dx',1,'offsets','dsp');
if ~isvar('mask')
    mask=ig.circ(1+ig.nx/2, 1+ig.ny/2) > 0;
end
ig.mask=mask;
if ~isvar('mask_locs')
    mask_locs=find(mask);
end
J = [6 6];
N2=[N N];
nufft_args = {N2, J, 2*N2, N2/2, 'table', 2^10, 'minmax:kb'};
Gm = Gmri(fliplr(fft_values(:,1:2)), ig.mask, ...
		'fov', ig.fov, 'basis', {'rect'}, 'nufft', nufft_args);
 beta = 2^-7 * size(fft_values,1);	   
Rn = Robject(ig.mask, 'type_denom', 'matlab', 'potential', 'hyper3', 'beta', 2^2*beta, 'delta', 0.3);
	%xh = pwls_pcg1(xpcg(ig.mask), Gm, 1, yi(:), Rn, 'niter', 2*niter);
    niter = 40;
xh = pwls_pcg1(zeros(Gm.dim(2),1), Gm, 1, fft_values(:,3), Rn, 'niter', niter);
image=zeros(N);
image(mask_locs)=xh;
