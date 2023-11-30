import torch


def calculate_kernel_matrix_as_tensor(
    X, Y=None, oa=False, se_kernel=None, normalize=True
) -> torch.Tensor:
    """
    Same as calculate kernel matrix, but in pytorch framework and uses autodiff to compute the gradient of
    the kernel function with respect to the feature vector.

    This function is taken out of the class to facilitate derivative computation.

    One difference is that to prevent the un-differentiable point at the min operation if optimal assignment
    kernel is used, we replace the hard-min with a soft-min differentiable approximation that uses the x-norm
    approximation.

    Parameters
    ----------
    X, Y: the feature vectors (X: train, Y: test). When Y is not supplied, the kernel matrix is computed with
        respect to itself.

    oa: bool: whether the optimal assignment kernel should be used.

    se_kernel: Defines any successive embedding kernel to be applied over the inner produce of X and Y. If none,
        a simple

    normalize: bool: Whether to normalize the GP covariance matrix to the range of [0, 1]. Default is True.

    Returns
    -------
    K: pytorch tensor, shape = [n_targets, n_inputs]
    dK_dY: pytorch tensor, of the same shape of K. The derivative of the value of the kernel function with
    respect to each of the X. If Y is None, the derivative is instead taken at the *training point* (i.e. X).
    """

    if Y is None:
        if se_kernel is not None:
            K = se_kernel.forward(X, X)
        else:
            K = X @ X.t()
        if normalize:
            K_diag = torch.sqrt(torch.diag(K))
            K_diag_outer = torch.ger(K_diag, K_diag)
            return K / K_diag_outer
    else:
        assert Y.shape[1] == X.shape[1], (
            "got Y shape " + str(Y.shape[1]) + " but X shape " + str(X.shape[1])
        )
        if se_kernel is not None:
            K = se_kernel.forward(X, Y)
        else:
            K = Y @ X.t()
        if normalize:
            Kxx = calculate_kernel_matrix_as_tensor(
                X, X, oa=oa, se_kernel=se_kernel, normalize=False
            )
            Kyy = calculate_kernel_matrix_as_tensor(
                Y, Y, oa=oa, se_kernel=se_kernel, normalize=False
            )
            K_diag_outer = torch.ger(
                torch.sqrt(torch.diag(Kyy)), torch.sqrt(torch.diag(Kxx))
            )
            return K / K_diag_outer
    return K
