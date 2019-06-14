class Layer:
  def forward(self, x):
    '''
    forward propagation

    Parameters
    ---
    x: matrix
      input from last layer

    Returns
    ---
    out: matrix
      output of current layer
    '''
    raise NotImplementedError

  def backprop(self, d):
    '''
    backward propagation

    Parameters
    ---
    d: matrix
      derivation from next layer

    Returns
    ---
    out_d: matrix
      derivation of current layer
    '''
    raise NotImplementedError

  def gradient(self, alpha):
    '''
    gradient descent

    Parameters
    ---
    alpha: float
      learning rate

    Returns:
      None
    '''
    raise NotImplementedError
