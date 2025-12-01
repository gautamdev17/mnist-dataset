# Neural Network From Scratch (NumPy Only)
### Learning Log — 30 November 2025

This readme1 documents the process of implementing a neural network entirely from scratch using NumPy and training it on the MNIST dataset. The log below contains a chronological record of everything attempted on 30 November 2025, including errors encountered, debugging steps, and lessons learned.

---

This was based on the tutorial posted by Samson Zhang,https://www.youtube.com/watch?v=w8yWXqWQYmU,Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)

## Learning Log (30 Nov 2025)

**8:41 AM — Initial Setup**  
Loaded MNIST using `mnist.load_data()`. Began implementing a 2-layer neural network (784 → hidden → 10). Did not fully understand the expected input shape `(features, samples)` at this stage.

**9:21 AM — Understanding Reshaping**  
Encountered confusion with:  
`X_train = X_train_raw.reshape(X_train_raw.shape[0], -1).T`  
Learned that reshaping flattens each 28×28 image into 784 features, and the transpose ensures each column corresponds to one training example. Final shape: `(784, 60000)`. This clarified why neural network implementations use column-major sample layout.

**10:27 AM — Normalization Error**  
Error encountered:  
`UFuncTypeError: Cannot cast ufunc 'divide' output from float64 to uint8`  
Cause: MNIST loaded as `uint8`. Division by 255 attempted an invalid in-place cast.  
Fix: Convert to float first:  
`X_train = X_train.astype("float32") / 255.0`.

**1:16 PM — Shape Mismatch in Forward Pass**  
Error:  
`ValueError: shapes (10,784) and (10,60000) not aligned`  
Cause: Incorrect weight matrix shape for the second layer. `W2` was constructed as `(10,784)` but should be `(10, hidden_dim)`. This clarified the necessity of consistent layer dimensioning.

**1:48 PM — Unexpected Early Termination**  
Training printed only iteration 0 and stopped without errors.  
Diagnosis: The `return` statement in `gradient_descent` was placed inside the for-loop.  
Fix: Moved the return statement outside the loop.

**2:42 PM — Accuracy Stuck at ~0.11**  
Training consistently produced approximately 0.11 accuracy across all iterations.  
Interpretation: The model was predicting a single class for all inputs. This indicated no useful gradient flow into the second layer. Root cause: Hidden layer too small (10 neurons), causing the model to collapse into a near-linear classifier unable to separate MNIST digits.

**10:34 PM — Bias Initialization Insight**  
Originally used random biases:  
`b1 = np.random.randn(10,1)`  
This caused many ReLU neurons to be inactive due to negative bias values.  
Fix: Switched to zero biases:  
`b1 = np.zeros((10,1))` and `b2 = np.zeros((10,1))`.

**11:29 PM — Inspecting Hidden Layer Activation**  
Added debugging statements to inspect hidden layer output:  
`A1 zero ratio`, `A1 mean`, `A1 shape`.  
Results showed:  
- Nonzero activations  
- A1 shape `(10, 60000)`  
Conclusion: The hidden layer size of 10 was far too small to represent 784-dimensional input in a useful way.

**12:03 AM(Dec 1) — Reproducing Tutorial Conditions**  
Adjusted training conditions to match the YouTube tutorial being referenced:  
- Hidden layer size: 10  
- Zero biases  
- Training on a small subset (first 5000 samples instead of full 60000)  
With these conditions, the model achieved approximately **96% accuracy**, confirming the implementation was correct. The earlier failures were due to attempting to train a very small network on the full MNIST dataset, which it lacks the capacity to learn.

---

## Key Technical Lessons

- MNIST requires a sufficiently large hidden layer (≥32, ideally 64 or more) for full-dataset training.  
- ReLU networks with negative random biases can suffer from dead neurons; zero biases are often safer when building from scratch.  
- Proper input shaping (flattening and transposing to `(features, samples)`) is essential for matrix-based neural network implementations.  
- Debugging intermediate activations (`Z1`, `A1`) is the most effective way to identify learning failures.  
- Small networks can overfit small subsets and appear to perform well, even if they cannot learn the full dataset.

---

## Final Result

Using the tutorial architecture (hidden size = 10, zero biases) on the first 5000 MNIST samples for 300–500 iterations with a learning rate of 0.1 resulted in approximately **96% training accuracy**. This confirms that the full NumPy implementation of forward propagation, backpropagation, and parameter updates works correctly.
Test result:- Test accuracy: 0.8655

---
## Noteworthy tweaks and what i learned

1. **Not scaling pixel values (0–255) → Accuracy ≈ 0.1126**  
   Scaling inputs with `/ 255.0` dramatically improved learning.  
   Unscaled inputs produced extremely large activations, unstable gradients, and the network collapsed to predicting a single class.  
   After scaling inputs to the range 0–1, accuracy increased to **0.9602** on the 5k subset.

2. **Changing training set size from 60k to 5k samples**
   Smaller networks can only learn proportional info. 
   Training on all 60,000 MNIST images with a very small hidden layer (10 neurons) resulted in accuracy ≈ 0.09–0.11.  
   Reducing the training set to 5,000 samples allowed the small network to fit the simpler distribution and reach **~84% accuracy**, eventually **~96%** after more iterations.  
   Lesson: small models can overfit small datasets but cannot learn the full MNIST distribution.

4. **Initializing weights to zero → No learning**  
   Setting `W1 = 0` (zero initialization) made every neuron in a layer compute the same output and receive identical gradients.  
   This caused *symmetry breaking failure*, effectively reducing the network to a structure like `784 → 1 → 10` instead of `784 → 10 → 10`.  
   With zero initialization, the network cannot learn at all.  
   Proper random initialization (e.g., small Gaussian values) is required for learning.
   
## Conclusion

This project involved building every part of a neural network manually: forward pass, activations, softmax, cross-entropy loss, and backpropagation. The debugging process covered shape alignment, initialization issues, ReLU dynamics, and data preprocessing errors. The final working model demonstrates a full understanding of neural network internals and the importance of architectural choices.
A larger ANN is recommended for mnist since it needs to capture features, preferabely 784->64->10, more hidden layers are recommended too, but train at your own risk!, overfitting may cause problems.
