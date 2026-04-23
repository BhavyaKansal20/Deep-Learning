# 📉 Vanishing & Exploding Gradients in Deep Learning

> **Deep Learning Series** &nbsp;|&nbsp; 🧠 Training Deep Neural Networks &nbsp;|&nbsp; 📚 Critical Problem & Solutions

---

## 🔍 Overview

Training deep neural networks requires managing two opposite but equally dangerous problems during **backpropagation**:

| Problem | What Happens | Effect |
|:---:|---|---|
| 📉 **Vanishing Gradient** | Gradients become **too small** | Early layers stop learning |
| 📈 **Exploding Gradient** | Gradients grow **too large** | Weights go unstable / diverge |

Both problems directly affect the model's **convergence** and **overall performance**.

```
OUTPUT LAYER ─────► Gradient = 0.5
                            │  × 0.4
HIDDEN LAYER 3 ───► Gradient = 0.2
                            │  × 0.3
HIDDEN LAYER 2 ───► Gradient = 0.06   ← getting tiny (Vanishing)
                            │  × 0.2
HIDDEN LAYER 1 ───► Gradient ≈ 0.012  ← almost dead ⚠️
```

---

## 📉 Part 1 — Vanishing Gradient Problem

### What Is It?

Vanishing gradients occur when gradients become **extremely small** during backpropagation, causing **early layers to learn very slowly or stop learning entirely**.

### 🔢 The Math — Chain Rule in Backpropagation

During backpropagation, the gradient of loss $L$ w.r.t. weight $w_i$ in layer $i$ is:

$$\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdot \frac{\partial a_{n-1}}{\partial a_{n-2}} \cdots \frac{\partial a_1}{\partial w_i}$$

| Symbol | Meaning |
|---|---|
| $L$ | Loss function |
| $w_i$ | Weight parameter in layer $i$ |
| $a_n$ | Activation output of layer $n$ |
| $\frac{\partial L}{\partial w_i}$ | Gradient of loss with respect to weight |

### ⚠️ Why Sigmoid / Tanh Cause This

When **Sigmoid** or **Tanh** activation functions are used, their derivatives are **always less than 1**. Multiplied repeatedly across layers → the gradient **vanishes exponentially** as it moves backward.

![Sigmoid Activation and Its Small Derivative](https://media.geeksforgeeks.org/wp-content/uploads/20241029120537926197/Sigmoid-Activation-Function.png)

*Figure 1 — Sigmoid function: output saturates near 0 and 1, where derivatives become nearly 0 — causing vanishing gradients*

> **Key Insight:** If each layer's derivative is `0.5`, after just 10 layers the gradient becomes `0.5¹⁰ = 0.00097` — nearly **zero**.

---

## 📈 Part 2 — Exploding Gradient Problem

### What Is It?

Exploding gradients occur when gradients grow **too large** during backpropagation, leading to **unstable weight updates** and **divergence in loss**.

When derivatives or weights are **greater than 1**, repeated multiplication across layers leads to **exponential growth**:

$$\prod_{i=1}^{n} \frac{\partial a_i}{\partial a_{i-1}} \longrightarrow \infty$$

---

### 🔢 The Gradient Descent Update Rule

$$\boxed{w_{t+1} = w_t - \eta \cdot \frac{\partial L}{\partial w_t}}$$

| Symbol | Meaning |
|---|---|
| $w_t$ | Current weight value at time step $t$ |
| $\eta$ | Learning rate |
| $\frac{\partial L}{\partial w_t}$ | Gradient of loss with respect to weight |
| $w_{t+1}$ | Updated weight after applying gradient descent |

> If $\frac{\partial L}{\partial w_t}$ is **too large** → weight updates become **massive** → model loss **oscillates or diverges**.

---

## 🎯 Side-by-Side: Vanishing vs Exploding

| Property | 📉 Vanishing | 📈 Exploding |
|---|---|---|
| **Gradient magnitude** | → 0 (too small) | → ∞ (too large) |
| **Cause** | Derivatives < 1, multiplied across layers | Derivatives or weights > 1, multiplied across layers |
| **Effect on weights** | Near-zero updates — weights barely change | Massive updates — weights become unstable |
| **Learning behavior** | Early layers stop learning | Loss oscillates or diverges |
| **Common in** | Deep nets with Sigmoid / Tanh | Deep nets with poor initialization / high LR |

---

## ❓ Why Do Gradients Vanish or Explode?

```
┌─────────────────────────────────────────────────────────┐
│             ROOT CAUSES OF GRADIENT ISSUES              │
├────────────────────────┬────────────────────────────────┤
│  ⚡ Activation Fn      │  Sigmoid/Tanh → derivatives    │
│                        │  < 1 → shrinks gradients       │
├────────────────────────┼────────────────────────────────┤
│  ⚖️ Weight Init        │  Too small → vanishing         │
│                        │  Too large → exploding         │
├────────────────────────┼────────────────────────────────┤
│  🏗️ Deep Networks      │  Many layers → repeated        │
│                        │  gradient multiplication       │
├────────────────────────┼────────────────────────────────┤
│  📏 Learning Rate      │  Too high → exploding          │
│                        │  Unscaled inputs → exploding   │
└────────────────────────┴────────────────────────────────┘
```

---

## 🧪 Visual Proof — Sigmoid vs ReLU Loss Curves

The graph below shows how **Sigmoid activation (vanishing gradient)** vs **ReLU activation (healthy gradient)** behave when training a 20-layer deep neural network:

![Vanishing Gradient Effect - Sigmoid vs ReLU Loss Comparison](https://media.geeksforgeeks.org/wp-content/uploads/20260115112413238128/Vanishing-Gradient-output.png)

*Figure 2 — Training loss over 100 epochs: Sigmoid (flat/stuck) vs ReLU (rapidly decreasing) — direct visualization of vanishing gradient problem*

### 📊 Interpreting the Graph

| Activation | Loss Behavior | Conclusion |
|:---:|---|---|
| **Sigmoid** | Loss stays almost constant — flat line | ❌ Vanishing gradient — gradients are too small to update weights |
| **ReLU** | Loss decreases rapidly and steadily | ✅ Healthy gradient flow — network is learning effectively |

---

## 🔧 Impact on Different Network Types

### Feedforward / Deep Neural Networks
- Vanishing gradients → **first layers learn almost nothing**
- Exploding gradients → **training diverges**, NaN weights

### Recurrent Neural Networks (RNNs)
- Vanishing gradients → **cannot remember long-term dependencies** in sequences
- This is why **LSTM and GRU** were invented — they use gate mechanisms to preserve gradient flow over long sequences

---

## 🛠️ Solutions — How to Fix These Problems

---

### ✅ Fix 1 — Proper Weight Initialization

Choosing the right initial weights keeps gradients **balanced** during backpropagation.

#### Xavier Initialization (Glorot)

Best for **Sigmoid / Tanh** activation functions.

$$\boxed{W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in}+n_{out}}},\ +\sqrt{\frac{6}{n_{in}+n_{out}}}\right]}$$

or with Normal distribution:

$$\boxed{\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}}$$

| Symbol | Meaning |
|---|---|
| $n_{in}$ | Number of inputs to the layer |
| $n_{out}$ | Number of outputs from the layer |

> **Goal:** Keep the variance of activations consistent across all layers → prevents gradients from shrinking or growing.

#### Kaiming Initialization (He)

Best for **ReLU and its variants**.

$$\boxed{\sigma = \sqrt{\frac{2}{n_{in}}}}$$

> **Why factor of 2?** ReLU kills ~half the neurons (negatives → 0). The factor of 2 compensates for this to preserve signal strength.

| Initialization | Best For | Key Idea |
|---|---|---|
| **Xavier** | Sigmoid, Tanh | Variance = 2 / (n_in + n_out) |
| **Kaiming (He)** | ReLU, Leaky ReLU | Variance = 2 / n_in |

---

### ✅ Fix 2 — Use Non-Saturating Activation Functions

Replace Sigmoid / Tanh with activations whose derivatives **don't shrink to near-zero**:

![ReLU — Non-Saturating Activation](https://media.geeksforgeeks.org/wp-content/uploads/20241029120652402777/relu-activation-function.png)

*Figure 3 — ReLU: gradient = 1 for all positive inputs — no vanishing, no saturation*

| Activation | Gradient Behavior | Use When |
|---|---|---|
| **ReLU** | 1 for x > 0, 0 for x < 0 | Default for most deep nets |
| **Leaky ReLU** | 1 for x > 0, 0.01 for x < 0 | When dying ReLU is a concern |
| **ELU / SELU** | Smooth negative output | Self-normalizing deep nets |

---

### ✅ Fix 3 — Batch Normalization

**Normalizes** the inputs to each layer to have **zero mean and unit variance**, stabilizing gradients and accelerating convergence.

$$\boxed{\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}}$$

Then apply learnable **scale** and **shift** parameters:

$$\boxed{y_i = \gamma \hat{x}_i + \beta}$$

| Symbol | Meaning |
|---|---|
| $\mu_B$ | Mean of the current mini-batch |
| $\sigma_B^2$ | Variance of the current mini-batch |
| $\epsilon$ | Small constant for numerical stability |
| $\gamma$ | Learnable scale parameter |
| $\beta$ | Learnable shift parameter |

```
BEFORE Batch Norm:             AFTER Batch Norm:
┌─────────────────────────┐    ┌─────────────────────────┐
│ Layer 1 output:          │    │ Layer 1 output:          │
│ [1200, 0.003, 550, ...]  │    │ [0.8, -0.9, 0.5, ...]   │
│ (wildly different scale) │    │ (mean≈0, variance≈1) ✅  │
└─────────────────────────┘    └─────────────────────────┘
```

> **Why it helps:** Keeps activations in a healthy range → gradients don't vanish or explode through the network.

**Benefits of Batch Normalization:**

| Benefit | Description |
|---|---|
| ⚡ Faster Convergence | Reduces internal covariate shift → training stabilizes faster |
| 📈 Higher LR Safe | Can use higher learning rates without divergence |
| 🧹 Regularization Effect | Slight noise from batch stats → reduces need for Dropout |
| 🔧 Less Sensitive to Init | Network is more robust to poor weight initialization |

---

### ✅ Fix 4 — Gradient Clipping

**Limits gradients to a maximum threshold** to prevent them from exploding and destabilizing training.

$$\boxed{g \leftarrow \min\left(1,\ \frac{\text{threshold}}{\|g\|}\right) \cdot g}$$

Two main strategies:

| Strategy | How It Works |
|---|---|
| **Value Clipping** | Clip each gradient component individually: $g_i = \text{clip}(g_i, -c, +c)$ |
| **Norm Clipping** | Scale the entire gradient vector if its norm exceeds threshold |

```
WITHOUT Gradient Clipping:     WITH Gradient Clipping (threshold = 1.0):
gradient = 847.3  ──────────►  gradient = 1.0   ✅ (clipped)
weight update = HUGE  ──────►  weight update = controlled
loss = NaN 💥              loss = stable 📉
```

> 💡 **Gradient Clipping is used especially in RNNs and LSTMs** where exploding gradients are common during sequence training.

---

## 📋 Complete Solutions Summary

| Problem | Fix | How It Helps |
|:---:|---|---|
| 📉 **Vanishing** | Xavier / Kaiming Init | Keeps variance balanced across layers |
| 📉 **Vanishing** | ReLU / Leaky ReLU / ELU | Gradient = 1 for positive inputs — no saturation |
| 📉 **Vanishing** | Batch Normalization | Normalizes inputs — keeps gradients stable |
| 📈 **Exploding** | Gradient Clipping | Hard cap on gradient magnitude |
| 📈 **Exploding** | Weight Regularization | L2 penalty prevents weights from growing too large |
| **Both** | Batch Normalization | Stabilizes entire gradient flow end-to-end |
| **Both** | Skip Connections (ResNet) | Allows gradients to bypass deep layers directly |

---

## 🏆 Key Takeaways

```
┌─────────────────────────────────────────────────────────┐
│               GRADIENT HEALTH CHECKLIST                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ✅ Use ReLU (or Leaky/ELU) in hidden layers            │
│  ✅ Use Xavier init for Sigmoid/Tanh networks            │
│  ✅ Use Kaiming init for ReLU networks                  │
│  ✅ Add Batch Normalization after heavy layers           │
│  ✅ Use Gradient Clipping for RNNs                      │
│  ✅ Monitor loss — flat = vanishing, NaN = exploding    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 📐 All Key Formulas — Quick Reference Card

$$\boxed{\text{Chain Rule (Backprop):} \quad \frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdots \frac{\partial a_1}{\partial w_i}}$$

$$\boxed{\text{Weight Update:} \quad w_{t+1} = w_t - \eta \cdot \frac{\partial L}{\partial w_t}}$$

$$\boxed{\text{Xavier Init:} \quad W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in}+n_{out}}},\ +\sqrt{\frac{6}{n_{in}+n_{out}}}\right]}$$

$$\boxed{\text{Kaiming Init:} \quad \sigma = \sqrt{\frac{2}{n_{in}}}}$$

$$\boxed{\text{Batch Norm:} \quad \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma\hat{x}_i + \beta}$$

$$\boxed{\text{Gradient Clipping:} \quad g \leftarrow \min\!\left(1, \frac{c}{\|g\|}\right) \cdot g}$$

---

*📝 These Notes are Written and compiled by Mr. Bhavya Kansal for Understanding Deep Learning from Scratch* <br> 
*🚀 Part of IITR and NIELIT Internship Journey  — Built in Patiala, Made in India 🇮🇳*