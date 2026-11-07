Neural Networks from Scratch

A modular, framework-free implementation of a multi-layer neural network based on the book 'neural network from scratch' built entirely from scratch using Python and NumPy. This project is part of a custom educational implementation exploring deep learning fundamentals, core mathematical operations, activation functions, loss calculations, backpropagation, and momentum-based weight optimization.

---

FEATURES

* Dense Layer (layer_Dense): Handles forward passes, weight and bias initialization, gradient calculations (dw, db, di), and momentum-based weight updates.
* Activation Functions:
  * ReLU (activation_Relu): Introduces non-linearity with zero-clipping during backward passes.
  * Softmax (activation_softmax): Normalizes output scores into probability distributions with a numerically stable implementation.
* Loss Functions:
  * Categorical Cross-Entropy (loss_CategoricalCrossEntropy): Computes negative log-likelihood loss with prediction clipping and analytical backward gradients.
* Optimization & Learning Rate Decay: Incorporates momentum and inverse-time learning rate decay to stabilize training over epochs.
* Dataset Integration: Utilizes synthetic spiral data (nnfs.datasets.spiral_data) for multi-class classification testing.
