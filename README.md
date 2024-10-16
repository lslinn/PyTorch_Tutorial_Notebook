# PyTorch Tutorial Notebook

This series of tutorials is designed to teach the basics of PyTorch functionality. After completing these tutorials, readers will acquire the skills to start constructing advanced machine learning algorithms using PyTorch.
The main body of this tutorial follows Patrick Loeber’s PyTorch tutorial on YouTube, which is recommended as a valuable starting material.

For PyTorch installation instructions, please refer to the official website: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

---

## Tutorials Overview

1. **tutorial1_TensorBasics.ipynb**  
   In this tutorial, readers will learn how to create and manage basic PyTorch functions related to tensor data.

   **Keywords**: `torch.empty()`, `torch.rand()`, `torch.ones()`, `torch.add/sub/div/mul/matmul()`, slicing, viewing, `torch.from_numpy()`, `tensors.clone()`, in-place/out-of-place operators

2. **tutorial2_PyTorchAutograd.ipynb**  
   This tutorial introduces PyTorch’s Autograd system, demonstrating how to build computational graphs and compute gradients using backpropagation.

   **Keywords**: `tensors.requires_grad`, `tensors.requires_grad_()`, `tensors.grad`, `tensors.grad.zero_`, `loss.backward()`, `tensor.is_leaf`, `tensors.detach()`, `with torch.no_grad()`

3. **tutorial3_Backpropagation_GradientDescent_Optimizers.ipynb**  
   Readers will learn how to use the `loss.backward()` function for gradient descent and explore PyTorch's built-in optimizers. The tutorial covers a linear regression model, both without and with PyTorch, to demonstrate PyTorch’s capabilities.

   **Keywords**: linear regression, model optimization with/without PyTorch autograd, PyTorch optimizers

4. **tutorial4_NeuralNetworkFromScratch.ipynb**  
   This tutorial begins by building neural networks from scratch, without using the `torch.nn` module. Readers will gain a deep understanding of neural network construction using only basic operations like addition and matrix multiplication. Afterward, the tutorial introduces the `torch.nn` module to showcase PyTorch’s capabilities.

   **Keywords**: building neural networks from scratch, `torch.nn`, printing model parameters, `nn.Sequential`, input/output tensor shapes (important)

5. **tutorial5_ActivationFunctions.ipynb**  
   This tutorial covers applying non-linear activation functions to dense linear layers in neural networks. This allows for predicting patterns beyond linear regions, and we will also discuss the common issue of overfitting.

   **Keywords**: activation functions, non-linear models

6. **tutorial6_Practice_ImageClassification.ipynb**  
   In this final tutorial, readers will build a basic image classification model, integrating the knowledge gained from previous lessons. This practice session will summarize the fundamental concepts needed to explore advanced machine learning algorithms using PyTorch.

   **Keywords**: image classification, summary
