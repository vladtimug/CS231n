import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    ###########################################################################
    # TODO: Implement the vanilla stochastic gradient descent update formula. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    param_scale = np.linalg.norm(w.ravel())

    update = -config['learning_rate'] * dw
    update_scale = np.linalg.norm(update.ravel())
    
    w += update

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return w, config,  update_scale/param_scale


class SGDMomentum:
  def __init__(self, v) -> None:
    self.v = v
  
  # INTEGRATEME
  # v parameter needs to be initialized with zero before the first training iteration
  # v parameter needs to have the same shape as the parameter to update (i.e w or dw)
  def sgd_momentum(self, w, dw, config=None):
      """
      Performs stochastic gradient descent with momentum.
      Compute parameter update using the exponentially weighted average of the gradients.
      This allows updates to be more agressively oriented towards the loss minimum
      enabling usage of a bigger learning rate to accelerate the learning process.

      - config format:
        - learning_rate: Scalar learning rate.
      """
      if config is None:
          config = {}

      config.setdefault("learning_rate", 1e-2)
      config.setdefault("mu", 0.9)

      param_scale = np.linalg.norm(w.ravel())
      
      self.v = config["mu"] * self.v + (1 - config["mu"]) * dw
      update = -config["learning_rate"] * self.v
      
      update_scale = np.linalg.norm(update.ravel())

      w += update

      return w, config, update_scale/param_scale

# INTEGRATEME
# Implementation requires API updates and modifications
# v parameter needs to be initialized with zero at the before the first training iteration
# v parameter needs to have the same shape as the parameter to update (i.e w or dw)
def sgd_nesterov_momentum(w, dw, v, config=None):
    """
    Performs stochastic gradient descent with Nesterov momentum
    
    config format:
      - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}

    config.setdefault("learning_rate", 1e-2)
    config.setdefault("mu", 0.9)

    w_ahead = w + config["mu"] * v

    param_scale = np.linalg.norm(w_ahead.ravel())
    
    # This gradient needs to be computed outside of this function, which requires API changes.
    # The two lines below are for demonstrative purposes
    compute_dw_ahead_gradient = lambda x: x
    dw_ahead = compute_dw_ahead_gradient(w_ahead)
    
    update = config["mu"] * v -config["learning_rate"] * dw_ahead

    update_scale = np.linalg.norm(update.ravel())

    w += update

    return w, config, update_scale / param_scale

# INTEGRATEME
def sgd_adagrad(w, dw, config=None):
    """
    Performs stochastic gradient descent with Adagrad parameter update
    
    config format:
      - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("eps", 1e-6)

    param_scale = np.linalg.norm(w.ravel())

    cache += dw**2
    update = -config['learning_rate'] * dw / (np.sqrt(cache) + config["eps"])
    
    update_scale = np.linalg.norm(update.ravel())
    
    w += update

    return w, config,  update_scale / param_scale

# INTEGRATEME
def sgd_rmsprop(w, dw, config=None):
    """
    Performs stochastic gradient descent with RMSProp parameter update
    
    config format:
      - learning_rate: Scalar learning rate.
    """
    if config is None:
      config = {}
    
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("eps", 1e-6)
    config.setdefault("decay_rate", 0.9)

    param_scale = np.linalg.norm(w.ravel())

    cache = config["decay_rate"] * cache + (1 - config["decay_rate"]) * dw**2
    update = -config['learning_rate'] * dw / (np.sqrt(cache) + config["eps"])
    
    update_scale = np.linalg.norm(update.ravel())
    
    w += update

    return w, config,  update_scale / param_scale

# INTEGRATEME
def sgd_adam(w, dw, t, config=None):
    """
    Performs stochastic gradient descent with Adam parameter update
    
    config format:
      - learning_rate: Scalar learning rate.
    """
    if config is None:
      config = {}
    
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("eps", 1e-8)
    config.setdefault("beta_1", 0.9)
    config.setdefault("beta_2", 0.999)

    param_scale = np.linalg.norm(w.ravel())

    m = config["beta_1"] * m + (1 - config["beta_1"]) * dw
    mt = m / (1 - config["beta_1"] ** t)
    v = config["beta_2"] * v + (1 - config["beta_2"]) * (dw**2)
    vt = v / (1 - config["beta_2"] ** t)
    
    update = -config["learning_rate"] * mt / np.sqrt(vt) + config["eps"]

    update_scale = np.linalg.norm(update.ravel())
    
    w += update

    return w, config,  update_scale / param_scale
