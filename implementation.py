import numpy as np

# Define the activation function (e.g., sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the activation function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the Lagrangian function
def lagrangian(W, X, B, C, F):
    N = len(W) - 1  # Adjusted for None at index 0
    L = C(X[N])  # Objective function
    for k in range(1, N+1):
        L += np.dot(B[k].T, X[k] - F(np.dot(W[k], X[k-1])))
    return L

# Forward propagation
def forward_propagation(W, X, F):
    N = len(W) - 1  # Adjusted for None at index 0
    for k in range(1, N+1):
        X[k] = F(np.dot(W[k], X[k-1]))  # Update the state of layer k
    return X

# Backward propagation
def backward_propagation(W, X, B, D, F_derivative):
    N = len(W) - 1  # Adjusted for None at index 0
    B[N] = 2 * (D - X[N])  # Boundary condition for the last layer
    for k in range(N-1, 0, -1):
        A = np.dot(W[k+1], X[k])
        B[k] = np.dot(W[k+1].T, F_derivative(A) * B[k+1])
    return B

# Weight update
def weight_update(W, X, B, learning_rate, F_derivative):
    N = len(W) - 1  # Adjusted for None at index 0
    for k in range(1, N+1):
        A = np.dot(W[k], X[k-1])
        Y = F_derivative(A) * B[k]
        W[k] += learning_rate * np.outer(Y, X[k-1])
    return W

# Training loop
def train_network(W, X, B, D, learning_rate, F, F_derivative, epochs):
    N = len(W) - 1  # Adjusted for None at index 0
    for epoch in range(epochs):
        # Forward propagation
        X = forward_propagation(W, X, F)
        
        # Backward propagation
        B = backward_propagation(W, X, B, D, F_derivative)
        
        # Weight update
        W = weight_update(W, X, B, learning_rate, F_derivative)
        
        # Compute the loss
        loss = np.mean((D - X[N])**2)
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss}")
    
    return W, X

# Initialize network parameters
layer_sizes = [2, 3, 3, 2]  # Sizes of each layer (including input and output layers)
N = len(layer_sizes) - 1  # Number of layers (excluding the input layer)

# Initialize weight matrices
W = [None] + [np.random.randn(layer_sizes[k], layer_sizes[k-1]) for k in range(1, len(layer_sizes))]

# Initialize state vectors (X) for each layer
X = [None] + [np.zeros(layer_sizes[k]) for k in range(len(layer_sizes))]

# Set input values
X[0] = np.array([0.1, 0.2])

# Initialize Lagrange multipliers (B) for each layer
B = [None] + [np.zeros(layer_sizes[k]) for k in range(1, len(layer_sizes))]

# Desired output
D = np.array([0.5, 0.5])

# Learning rate and number of epochs
learning_rate = 0.1
epochs = 1000

# Train the network
W, X = train_network(W, X, B, D, learning_rate, sigmoid, sigmoid_derivative, epochs)

# Print final weights and output
print("Final Weights:")
for k in range(1, N+1):
    print(f"W[{k}]:\n{W[k]}")
print("Final Output:")
print(X[N])
print("Target Output:")
print(D)
