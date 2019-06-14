class Layer:
  def forward(self, x):
    '''
    forward propagation

    Parameters
    ---
    x:  input from last layer
    '''
    raise NotImplementedError

  def backprop(self, d):
    '''
    backward propagation

    Parameters
    ---
    d:  derivation from next layer
    '''
    raise NotImplementedError

  def gradient(self, alpha):
    '''
    gradient descent

    Parameters
    ---
    alpha:  learning rate
    '''
    raise NotImplementedError
