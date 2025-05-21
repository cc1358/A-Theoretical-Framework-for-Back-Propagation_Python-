🧮 Lagrangian-Based Backpropagation in NumPy

This project implements a basic feedforward neural network with a custom Lagrangian-based backward propagation algorithm using pure NumPy. It’s designed for learners, researchers, and enthusiasts interested in the mathematical foundations of deep learning, particularly in exploring alternatives to standard gradient backpropagation.

⸻

📚 Motivation

Most neural networks today use gradient descent and automatic differentiation to update weights. But what happens when you manually define a Lagrangian and derive backpropagation from physical or optimization principles?

This project answers that question with:
	•	A 2-layer neural network (input → hidden → output)
	•	Manual computation of gradients via Lagrangian-based updates
	•	Minimal use of external libraries for transparency

⸻

🧠 Features
	•	Custom forward pass using matrix multiplication
	•	Lagrangian-based backward pass with analytically derived updates
	•	Training loop with weight updates via manual gradient application
	•	Configurable architecture (input size, hidden size, learning rate)
	•	Written entirely in NumPy for educational clarity
