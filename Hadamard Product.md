---
aliases:
  - Elementwise Product
---

The Hadamard Product (⊙) just refers to the element-wise multiplication of two matrices of vectors, rather than standard matrix multiplications.


Given matrix A
$$ \begin{array}{cc} 2 &3 \\ 4 &1 \end{array} $$
and matrix B
$$ \begin{array}{cc} 1 &0 \\ 5 &2 \end{array} $$
Then the Hadamard Product A ⊙ B would just be:
$$
\begin{array}{cc}
2 &0 \\
20 &2
\end{array}
$$

Generally, you can't take the Hadamard product between differently-sized tensors, though some programming frameworks like NumPy or PyTorch might allow it via "broadcasting" (which actually expands the smaller tensor to match the larger one).