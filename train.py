# %%

import numpy as np
from lib import *

def train_step(params, inputs, outputs, learning_rate = .1):
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

  return loss

def main():
  # Np seed
  # seed = 1
  # np.random.seed(seed)

  # Network initialization
  n_in = 3
  n_hidden = 2
  n_out = 3
  params = {}
  params['W0'] = np.random.randn(n_hidden, n_in) * 0.1  # 2 x 3
  params['b0'] = np.random.randn(n_hidden, 1) * 0.1 # 2 x 1
  params['W1'] = np.random.randn(n_out, n_hidden) * 0.1  # 3 x 2
  params['b1'] = np.random.randn(n_out, 1) * 0.1  # 3 x 1

  inputs = np.array([[1.0], [2.0], [3.0]])  # 3 x 1
  outputs = np.array([[1.0], [0], [0]]) # 3 x 1

  losses = []

  # Train network
  steps = 20
  for step in range(steps):
    loss = train_step(params, inputs, outputs, learning_rate=0.01)
    losses.append(float(loss))

  # Evaluate
  # Forward to hidden layer and ouut layer
  A_hidden, Z_hidden = forward(inputs, params['W0'], params['b0'])  # 2 x 1
  A_out, Z_out = forward(A_hidden, params['W1'], params['b1'], activation="sigmoid") # 3 x 1

  print(f"Loss: {losses}")
  print(f"Label: {np.squeeze(outputs).tolist()}, Prediction: {np.squeeze(A_out).tolist()}")


if __name__ == "__main__":
  main()