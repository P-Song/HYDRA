function res=fft_mats(X,forward_backward)
N=size(X,1);
L=size(X,2)/N;
res=single(zeros(size(X)));
if forward_backward==1
    for i=1:L
        res(:,(i-1)*N+1:i*N)=fft2(X(:,(i-1)*N+1:i*N));
    end
else
    for i=1:L
        res(:,(i-1)*N+1:i*N)=ifft2(X(:,(i-1)*N+1:i*N));
    end
end