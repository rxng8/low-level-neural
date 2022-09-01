# %%

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

seed = 1
np.random.seed(seed)
n_in = 3
n_hidden = 2
n_out = 3

inputs = np.array([[1.0], [2.0], [3.0]])  # 3 x 1
outputs = np.array([[1.0], [0], [0]]) # 3 x 1

params = {}
params['W0'] = np.random.randn(n_hidden, n_in) * 0.1  # 2 x 3
params['b0'] = np.random.randn(n_hidden, 1) * 0.1 # 2 x 1

params['W1'] = np.random.randn(n_out, n_hidden) * 0.1  # 3 x 2
params['b1'] = np.random.randn(n_out, 1) * 0.1  # 3 x 1

params

# %%

# Forward
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

# Forward to hidden layer and ouut layer
A_hidden, Z_hidden = forward(inputs, params['W0'], params['b0'])  # 2 x 1

# A_hidden
A_out, Z_out = forward(A_hidden, params['W1'], params['b1'], activation="sigmoid") # 3 x 1
A_out


def loss_func(y_true, y_pred):
  m = y_true.shape[0]
  cost = -1 / m * (y_true.T @ np.log(y_pred)) + ((1 - y_true).T @ np.log(1 - y_pred))
  return np.squeeze(cost)

loss = loss_func(outputs, A_out)
loss


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


dA_out = - (np.divide(outputs, A_out) - np.divide(1 - outputs, 1 - A_out))
dA_out



dA_hidden, dW1, db1 = backward(
  dA_out, 
  params['W1'], params['b1'], Z_out, A_hidden
)

dA_in, dW0, db0 = backward(
  dA_hidden, 
  params['W0'], params['b0'], Z_hidden, inputs
)


learning_rate = .1
# Update

params['W1'] -= learning_rate * dW1
params['b1'] -= learning_rate * db1

params['W0'] -= learning_rate * dW0
params['b0'] -= learning_rate * db0


# %%

# Evaluate 

# # Forward to hidden layer and ouut layer
# A_hidden, Z_hidden = forward(inputs, params['W0'], params['b0'])  # 2 x 1

# # A_hidden
# A_out2, Z_out = forward(A_hidden, params['W1'], params['b1'], activation="sigmoid") # 3 x 1
# A_out2

# %%

inputs = np.array([[1.0], [2.0], [3.0]])  # 3 x 1
outputs = np.array([[1.0], [0], [0]]) # 3 x 1

losses = []

# Forward to hidden layer and ouut layer
A_hidden, Z_hidden = forward(inputs, params['W0'], params['b0'])  # 2 x 1
A_out, Z_out = forward(A_hidden, params['W1'], params['b1'], activation="sigmoid") # 3 x 1

# Loss computation
loss = loss_func(outputs, A_out)

# d_loss
dA_out = - (np.divide(outputs, A_out) - np.divide(1 - outputs, 1 - A_out))

# Backprop
dA_hidden, dW1, db1 = backward(
  dA_out, 
  params['W1'], params['b1'], Z_out, A_hidden
)

dA_in, dW0, db0 = backward(
  dA_hidden, 
  params['W0'], params['b0'], Z_hidden, inputs
)

# Update weights
params['W1'] -= learning_rate * dW1
params['b1'] -= learning_rate * db1
params['W0'] -= learning_rate * dW0
params['b0'] -= learning_rate * db0

A_out

# %%

# Forward to hidden layer and ouut layer
A_hidden, Z_hidden = forward(inputs, params['W0'], params['b0'])  # 2 x 1
A_out2, Z_out = forward(A_hidden, params['W1'], params['b1'], activation="sigmoid") # 3 x 1
A_out2
# %%


# check finite difference
# Q(z, w + delta*e_i) ≈ ( Q(z, w) + delta * g_i )
difference_y1 = learning_rate * (A_out2 - A_out)

# Update weights
params['W1'] = learning_rate * dW1
params['b1'] = learning_rate * db1
params['W0'] = learning_rate * dW0
params['b0'] = learning_rate * db0

# Forward to hidden layer and ouut layer
A_hidden, Z_hidden = forward(inputs, params['W0'], params['b0'])  # 2 x 1
difference_y2, Z_out = forward(A_hidden, params['W1'], params['b1'], activation="sigmoid") # 3 x 1


# %%

print(difference_y1)
print(difference_y2)




