

function mse_val = mse(X,ref)

    X = reshape(X, size(ref,1), size(ref,2), []);
    err = abs(X(:) - ref(:));
    mse_val = mean(err.^2);

end