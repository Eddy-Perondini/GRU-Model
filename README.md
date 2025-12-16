Modeling a GRU for sequential data prediction using Tensorflow.

BRIEF EXPLANATION:

# Gated Recurrent Units (GRUs)

## Overview
Gated Recurrent Units (GRUs) are an improved version of Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs), designed to mitigate the **vanishing gradient problem** while maintaining lower computational complexity.

---

## Main Problem of RNNs
- **Vanishing Gradient**

---

## Advantages of GRUs
- Fewer parameters to compute  
- Lower computational complexity compared to RNNs and LSTMs  
- Reduced memory usage  

---

## GRU Architecture

Let $\( f(\cdot) \)$ be an activation function, usually:
- $\( f = \tanh(\cdot) \)$
- $\( f = \sigma(\cdot) \) (sigmoid)$

### Update Gate
Decides which information to keep or discard and determines how much information is retained in the processing cell:

\[
$$Z_t = f(W_{xz} X_t + W_{hz} h_{t-1} + b)$$
\]

---

### Reset Gate
Controls how much past information is retained and adjusts the input based on previous memories:

\[
$$r_t = f(W_{xr} X_t + W_{hr} h_{t-1} + b)$$
\]

---

### Candidate Hidden State
Selects candidate memories for updating the cell state:


$$\hat{h}_t = f(W_{xh} X_t + W_{hh} (r_t \odot h_{t-1}) + b)$$


---

### Hidden State
Helps mitigate the vanishing gradient problem present in standard RNNs:

\[
$$h_t = (1 - Z_t) \odot h_{t-1} + Z_t \odot \hat{h}_t$$
\]

---

## References
- [Dive into Deep Learning — GRU Implementation](https://github.com/d2l-ai/d2l-tensorflow-colab/blob/master/chapter_recurrent-modern/gru.ipynb)
- [YouTube: GRU Explained](https://www.youtube.com/watch?v=rdz0UqQz5Sw)


