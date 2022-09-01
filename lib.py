import numpy as np

def sigmoid(Z):
  return 1 / (1 + np.exp(-Z))

def relu(Z):
  return np.maximum(0, Z)

def sigmoid_backward(dA, Z):
  sig = sigmoid(Z)
  return dA * sig * (1 - sig)

def relu_backward(dA, Z):
  dZ = np.array(dA, copy = True)
  dZ[Z <= 0] = 0
  return dZ

def forward(A_prev, W_curr, b_curr, activation="relu"):
  Z_curr = W_curr @ A_prev + b_curr
  act_func = None
  if activation == "relu":
    act_func = relu
  elif activation == "sigmoid":
    act_func = sigmoid
  else:
    raise Exception("Not supported!")
  return act_func(Z_curr), Z_curr

def loss_func(y_true, y_pred):
  m = y_true.shape[0]
  cost = -1 / m * (y_true.T @ np.log(y_pred)) + ((1 - y_true).T @ np.log(1 - y_pred))
  return np.squeeze(cost)

def backward(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):

  """ case of output backward to hidden
  dA_curr 3x1
  W_curr 3x2
  b_curr 3x1
  Z_curr 3x1
  A_prev 2x1
  """

  m = A_prev.shape[0] # 2
  
  if activation == "relu":
    backward_activation_func = relu_backward
  elif activation == "sigmoid":
    backward_activation_func = sigmoid_backward
  else:
    raise Exception('Non-supported activation function')
  
  dZ_curr = backward_activation_func(dA_curr, Z_curr) # 3 x 1
  dW_curr = (dZ_curr @ A_prev.T) / m  # 3 x 2
  db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m  # 3 x 1
  dA_prev = W_curr.T @ dZ_curr # 2 x 1

  # 2 x 1, 3 x 2, 3 x 1
  return dA_prev, dW_curr, db_curr