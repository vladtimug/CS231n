from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    inp = x.reshape(x.shape[0], -1)
    out = np.dot(inp, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    inp = x.reshape((x.shape[0], -1))
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(inp.T, dout)
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.copy(dout)
    dx[x<=0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = x.shape[0]
    
    x -= np.max(x, axis=1, keepdims=True)
    probabilities = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    logits = -np.log(probabilities[range(num_train), y])
    loss = np.sum(logits) / num_train

    dx = probabilities.copy()
    dx[range(num_train), y] -= 1
    dx /= num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute minibatch stats feature-wise
        minibatch_mean = np.mean(x, axis=0)
        minibatch_var = np.var(x, axis=0)
        minibatch_std = np.sqrt(minibatch_var + eps)

        # Center and normalize the minibatch features
        centered_input = x - minibatch_mean
        inv_std = 1 / minibatch_std
        normalized_input = centered_input * inv_std
        out = gamma * normalized_input + beta

        # Update running average of the minibatch mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * minibatch_mean
        running_var = momentum * running_var + (1 - momentum) * minibatch_var

        # Register cache
        cache = (
            normalized_input, gamma, centered_input, inv_std, minibatch_std
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        normalized_input = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * normalized_input + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # https://www.adityaagrawal.net/blog/deep_learning/bprop_batch_norm

    normalized_input, gamma, centered_input, inv_std, minibatch_std = cache

    N, D = dout.shape

    dbeta = np.sum(dout, axis=0)
    
    dgamma = np.sum(dout * normalized_input, axis=0)
    
    dxhat = dout * gamma

    divar = np.sum(dxhat * centered_input, axis=0)
    dxmu1 = dxhat * inv_std

    dsqrtvar = -1. / (minibatch_std ** 2) * divar

    dvar = 0.5 * 1. / minibatch_std * dsqrtvar

    dsq = 1. / N * np.ones((N, D)) * dvar

    dxmu2 = 2 * centered_input * dsq

    dx1 = dxmu1 + dxmu2
    dmu = -1 * np.sum(dx1, axis=0)

    dx2 = 1. / N * np.ones((N, D)) * dmu

    dx = dx1 + dx2

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = dout.shape
    
    normalized_input, gamma, _, inv_var, _ = cache

    # intermediate partial derivatives
    dxhat = dout * gamma

    # final partial derivatives
    dx = (1. / N) * inv_var * (N * dxhat - np.sum(dxhat, axis=0)
      - normalized_input * np.sum(dxhat * normalized_input, axis=0))
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(normalized_input * dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute stats sample-wise
    mean = np.mean(x, axis=1).reshape(-1, 1)
    var = np.var(x, axis=1).reshape(-1, 1)
    std = np.sqrt(var + eps)

    # Center and normalize the samples
    centered_input = x - mean
    inv_std = 1 / std
    normalized_input = centered_input * inv_std
    out = gamma * normalized_input + beta

    # Register cache
    cache = (
        normalized_input, gamma, centered_input, inv_std, std
    )
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    normalized_input, gamma, centered_input, inv_std, std = cache

    gamma = gamma.T

    N, D = dout.shape

    dbeta = np.sum(dout, axis=0)
    
    dgamma = np.sum(dout * normalized_input, axis=0)
    
    dxhat = dout * gamma

    divar = np.sum(dxhat * centered_input, axis=1).reshape(-1, 1)
    
    dxmu1 = dxhat * inv_std

    dsqrtvar = -1. / (std ** 2) * divar

    dvar = 0.5 * 1. / std * dsqrtvar

    dsq = 1. / D * np.ones((N, D)) * dvar

    dxmu2 = 2 * centered_input * dsq

    dx1 = dxmu1 + dxmu2
    
    dmu = -1 * np.sum(dx1, axis=1).reshape(-1, 1)

    dx2 = 1. / D * np.ones((N, D)) * dmu

    dx = dx1 + dx2

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    F, CC, HH, WW = w.shape

    padded_input = np.pad(
        array=x,
        pad_width=((0,0), (0,0), (conv_param["pad"], conv_param["pad"]), (conv_param["pad"], conv_param["pad"])),
        mode="constant",
        constant_values=0
    )

    result_height = int(1 + (H + 2 * conv_param["pad"] - HH) / conv_param["stride"])
    result_width = int(1 + (W + 2 * conv_param["pad"] - WW) / conv_param["stride"])

    out = np.zeros(shape=(N, F, result_height, result_width))

    for sample_idx in range(N):
        for filter_idx in range(F):
            filtr = w[filter_idx, :, :, :]
            for out_vertical_idx, inp_vertical_idx in enumerate(range(0, H, conv_param["stride"])):
                for out_horizontal_idx, inp_horizontal_idx in enumerate(range(0, W, conv_param["stride"])):
                  sample_region = padded_input[sample_idx, :, inp_vertical_idx:inp_vertical_idx + HH, inp_horizontal_idx:inp_horizontal_idx + WW]
                  out[sample_idx, filter_idx, out_vertical_idx, out_horizontal_idx] = np.sum(sample_region * filtr) + b[filter_idx]
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
    # https://deeplearning.cs.cmu.edu/F20/document/slides/lec11.CNN.pdf - slide 30 onwards

    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    samples_number, sample_channels, _, _ = x.shape
    filters_number, _, filter_height, filter_width = w.shape
    _, _, upstream_gradient_height, upstream_gradient_width = dout.shape
    
    # db computed analogous to the bias gradient computed for the FCN case
    for f in range(filters_number):
        db[f] = np.sum(dout[:, f, :, :])
    
    xp = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
    dxp = np.pad(dx, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')

    # compute dx and dw
    for sample_idx in range(samples_number):
        for filter_idx in range(filters_number):
            for hw in range(upstream_gradient_height):
                start_h = stride * hw
                end_h = start_h + filter_height
                for ww in range(upstream_gradient_width):
                    start_w = stride * ww
                    end_w = start_w + filter_width
                    for channel_idx in range(sample_channels):
                        upstream_gradient_value = dout[sample_idx, filter_idx, hw, ww]
                        dxp[
                            sample_idx, channel_idx, start_h:end_h, start_w:end_w
                          ] += upstream_gradient_value * w[filter_idx, channel_idx, :, :]
                        dw[
                            filter_idx, channel_idx, :, :
                          ] += upstream_gradient_value * xp[sample_idx, channel_idx, start_h:end_h, start_w:end_w]

    dx = dxp[:,:,pad:-pad,pad:-pad]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    stride = pool_param["stride"]
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]

    H_ = int(1 + (H - pool_height) / stride)
    W_ = int(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_, W_))

    for sample_idx in range(N):
        for channel_idx in range(C):
            for out_vertical_idx, inp_vertical_idx in enumerate(range(0, H, stride)):
                for out_horizontal_idx, inp_horizontal_idx in enumerate(range(0, W, stride)):
                    out[sample_idx, channel_idx, out_vertical_idx, out_horizontal_idx] = np.max(
                        x[sample_idx, channel_idx,
                          inp_vertical_idx:inp_vertical_idx + pool_height,
                          inp_horizontal_idx : inp_horizontal_idx + pool_width
                          ])
                    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache

    N, C, H, W = x.shape
    stride = pool_param["stride"]
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]

    dx = np.zeros_like(x)

    for sample_idx in range(N):
        for channel_idx in range(C):
            for out_vertical_idx, inp_vertical_idx in enumerate(range(0, H, stride)):
                for out_horizontal_idx, inp_horizontal_idx in enumerate(range(0, W, stride)):
                    receptive_field = x[sample_idx, channel_idx,
                          inp_vertical_idx : inp_vertical_idx + pool_height,
                          inp_horizontal_idx : inp_horizontal_idx + pool_width
                        ]
                    max_vertical_idx, max_horizontal_idx = np.unravel_index(
                        np.argmax(receptive_field), receptive_field.shape
                      )
                    max_vertical_idx += inp_vertical_idx
                    max_horizontal_idx += inp_horizontal_idx
                    dx[sample_idx, channel_idx, max_vertical_idx, max_horizontal_idx] = dout[
                        sample_idx, channel_idx, out_vertical_idx, out_horizontal_idx
                      ]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape

    out = np.zeros_like(x)
    cache = [0] * C
    for channel_idx in range(C):
        stacked_channels = np.vstack(x[:, channel_idx, :, :])
        out_ch, cache[channel_idx] = batchnorm_forward(x=stacked_channels, gamma=gamma[channel_idx], beta=beta[channel_idx], bn_param=bn_param)
        out_ch = out_ch.reshape(N, H, W)
        for sample_idx in range(N):
            out[sample_idx, channel_idx, :, :] = out_ch[sample_idx]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dx = np.zeros_like(dout)
    dgamma = np.zeros(shape=C)
    dbeta = np.zeros(shape=C)

    for channel_idx in range(C):
        stacked_channels = np.vstack(dout[:, channel_idx, :, :])
        dx_ch, dgamma_ch, dbeta_ch = batchnorm_backward(stacked_channels, cache[channel_idx])
        dx_ch = dx_ch.reshape(N, H, W)
        dgamma[channel_idx] = np.sum(dgamma_ch)
        dbeta[channel_idx] = np.sum(dbeta_ch)
        for sample_idx in range(N):
            dx[sample_idx, channel_idx, :, :] = dx_ch[sample_idx]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape

    x = np.reshape(x, (N, G, C // G, H, W))
    
    # Compute stats channel-wise
    mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    var = np.var(x, axis=(2, 3, 4), keepdims=True)
    std = np.sqrt(var + eps)

    # Center and normalize channels
    centered_input = x - mean
    inv_std = 1 / std
    normalized_input = centered_input * inv_std
    normalized_input = np.reshape(a=normalized_input, newshape=(N, C, H, W))
    out = gamma * normalized_input + beta

    # Register cache
    cache = (
        normalized_input, gamma, centered_input, inv_std, std, G
    )

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    normalized_input, gamma, centered_input, inv_std, std, G = cache

    dbeta = np.zeros(shape=(1, C, 1, 1))
    dgamma = np.zeros(shape=(1, C, 1, 1))
    dx = np.zeros(shape=(N, C, H, W))

    dgamma_full = dout * normalized_input
    for c in range(C):
        dbeta[:,c,:,:] = np.sum(dout[:, c, :, :])
        dgamma[:,c,:,:] = np.sum(dgamma_full[:, c, :, :])

    dxhat = dout * gamma
    dxhat = dxhat.reshape(N, G, C // G, H, W)

    divar = np.sum(dxhat * centered_input, axis=(2, 3, 4)).reshape(std.shape)
    
    dxmu1 = dxhat * inv_std
    
    dsqrtvar = -1 / (std**2) * divar
    
    dvar = 0.5 * 1 / std * dsqrtvar

    D = H * W * (C // G)
    dsq = 1. / D * np.ones(shape=(N, G, C // G, H, W)) * dvar
    
    dxmu2 = 2 * centered_input * dsq
    
    dx1 = dxmu1 + dxmu2 

    dmu = -1 * np.sum(dx1, axis = (2, 3, 4)).reshape(std.shape)
    
    dx2 = 1. / D * np.ones(shape=(N, G, C // G, H, W)) * dmu
    
    dx = dx1 + dx2
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
