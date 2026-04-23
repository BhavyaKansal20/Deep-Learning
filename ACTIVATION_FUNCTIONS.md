# ⚡ Types of Activation Functions

> **Deep Learning Series** &nbsp;|&nbsp; 🧠 Neural Network Fundamentals &nbsp;|&nbsp; 📚 Complete Reference Guide

---

## 🔍 What is an Activation Function?

An **Activation Function** decides whether a neuron should be activated or not — it introduces **non-linearity** into the network, enabling it to learn complex patterns that a purely linear model never could.

```
INPUT ──► WEIGHTED SUM ──► ACTIVATION FUNCTION ──► OUTPUT
               (Σ wᵢxᵢ + b)           f(a)
```

---

## 📋 Quick Comparison Table

| # | Function | Formula | Range | Best Used In |
|:---:|---|---|:---:|---|
| 1 | **Linear** | $f(x) = x$ | $(-\infty, +\infty)$ | Output Layer (Regression) |
| 2 | **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | Output Layer (Binary Classif.) |
| 3 | **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, +1)$ | Hidden Layers |
| 4 | **ReLU** | $\max(0, x)$ | $[0, +\infty)$ | Hidden Layers (CNN/DNN) |
| 5 | **Leaky ReLU** | $\max(0.01x, x)$ | $(-\infty, +\infty)$ | Deep Networks |
| 6 | **PReLU** | $\max(\alpha x, x)$ | $(-\infty, +\infty)$ | CNNs, Deep Networks |
| 7 | **Swish** | $x \cdot \sigma(x)$ | $(-\infty, +\infty)$ | Deep Networks (Google) |

---

## 1️⃣ Linear Activation Function

> *The simplest type — output is exactly equal to the input.*

$$\boxed{f(x) = x}$$

The output is the **same as the input**. The neuron doesn't transform the data — it just passes it forward as-is.

![Linear Activation Function Graph](https://media.geeksforgeeks.org/wp-content/uploads/20241029115212560858/Linear-Activation-Function.png)

*Figure 1 — Linear Activation: output = input, a straight diagonal line with no transformation*

---

### 📌 Where Do We Use It?

- **Regression tasks** — predicting continuous values like salary, house price, temperature, etc.
- **Output layer** of a neural network when we don't want output restricted to 0–1 (sigmoid) or −1 to +1 (tanh)

> **Example:** Predicting house prices — you want outputs like ₹50,00,000 or ₹80,00,000. A sigmoid would squeeze everything between 0 and 1, which makes no sense here. ✅

---

### ⚠️ Limitations

| Problem | Explanation |
|---|---|
| **Linear Model Collapse** | If you use linear activation in all layers, the whole network becomes just a linear model — no matter how many layers you add |
| **Cannot Capture Complexity** | Cannot learn complex, non-linear patterns in data |

> 💡 **That's why** hidden layers use non-linear activations like **ReLU, tanh, sigmoid** — but the output layer for regression *can* be linear.

---
---

## 2️⃣ Sigmoid Activation Function

> *An S-shaped curve that squashes any real number into a range between 0 and 1.*

$$\boxed{f(x) = \frac{1}{1 + e^{-x}}}$$

No matter how large or small the input — **output always stays between 0 and 1**.

![Sigmoid Activation Function Graph](https://media.geeksforgeeks.org/wp-content/uploads/20241029120537926197/Sigmoid-Activation-Function.png)

*Figure 2 — Sigmoid: smooth S-curve always outputting between 0 and 1*

---

### 📌 Where Do We Use It?

- **Binary classification** problems (e.g., predicting yes/no, disease/no disease, spam/not spam)
- **Output layer** when you want a **probability** as the output

> **Example:** If the sigmoid outputs `0.85`, you can interpret it as **85% chance of having heart disease**. ✅

---

### ✅ Advantages vs ⚠️ Limitations

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| Output is always a clean probability (0 to 1) | **Vanishing Gradient** — for large/small inputs, gradient → 0, slowing learning |
| Smooth and continuously differentiable | **Not preferred in hidden layers** — ReLU is preferred nowadays |

---
---

## 3️⃣ Tanh Activation Function

> *Like sigmoid, but centered at 0 — squeezes values into −1 to +1.*

$$\boxed{f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}}$$

![Tanh Activation Function Graph](https://media.geeksforgeeks.org/wp-content/uploads/20241029120618881107/Tanh-Activation-Function.png)

*Figure 3 — Tanh: S-curve centered at 0, ranging from −1 to +1*

---

### 📌 Where Do We Use It?

- **Hidden layers** of neural networks
- Useful when data has **both positive and negative values** — it centres the output around 0 (unlike sigmoid which is centred at 0.5)

---

### ✅ Advantages vs ⚠️ Limitations

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| **Zero-centred** output — better for gradient optimization | Still suffers from **vanishing gradient** for very large/small inputs |
| **Stronger gradients** than sigmoid in range (−1, 1) → faster learning | ReLU is more common in hidden layers in modern deep learning |

---
---

## 4️⃣ ReLU Activation Function

> *The most widely used activation function in modern deep learning.*

**ReLU** stands for **Rectified Linear Unit**. It's super simple:

$$\boxed{f(x) = \max(0, x)}$$

**That means:**

```
If input x < 0  →  output = 0       (blocked)
If input x > 0  →  output = x       (passed as-is)
```

![ReLU Activation Function Graph](https://media.geeksforgeeks.org/wp-content/uploads/20241029120652402777/relu-activation-function.png)

*Figure 4 — ReLU: flat zero for negatives, linear for positives — the famous "hockey stick" curve*

---

### 📌 Where Do We Use It?

- **Hidden layers** of almost all modern deep neural networks
- Works really well in **CNNs**, image recognition, NLP, and many more tasks

---

### ✅ Advantages vs ⚠️ Limitations

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| Very **fast and simple** to compute | **Dying ReLU problem** — neurons can get stuck at 0 forever |
| Helps **avoid vanishing gradient** (better than sigmoid/tanh) | **Not smooth at 0** — not differentiable there (but fine in practice) |
| Makes training **deep networks much faster** | — |

---
---

## 5️⃣ Leaky ReLU Activation Function

> *ReLU with a small fix — neurons never completely die.*

$$\boxed{f(x) = \begin{cases} x & \text{if } x > 0 \\ 0.01x & \text{if } x \leq 0 \end{cases}}$$

Instead of giving 0 for negative inputs, it gives a **tiny negative value** (`0.01 × input`). The neuron is **never completely dead**.

![Leaky ReLU vs ReLU Graph](https://media.geeksforgeeks.org/wp-content/uploads/20251008111001414919/Leaky_relu.png)

*Figure 5 — Leaky ReLU (orange) vs ReLU (blue): small negative slope instead of flat zero for x < 0*

---

### 📌 Advantages of Leaky ReLU

**🔧 Fixes "Dead Neuron" Problem**
In normal ReLU, if inputs go negative, the neuron stops learning permanently (dead neuron). Leaky ReLU allows a **small negative slope** — neurons still update weights.

**⚡ Computationally Simple**
No heavy math like exponentials in Sigmoid/Tanh — just a tiny slope for negatives.

**📈 Better Gradient Flow**
Even negative inputs have a small gradient (0.01), so the network can **continue learning** — reducing vanishing gradient.

**🏗️ Works Well in Deep Networks**
Especially useful where ReLU may suffer from many dead neurons.

---

### ⚠️ Limitations

| Problem | Detail |
|---|---|
| Biased results | Small negative slope may bias outputs |
| Hyperparameter | The `0.01` slope value needs manual tuning |

---
---

## 6️⃣ PReLU Activation Function

> *Leaky ReLU, but smarter — the slope α is learned by the model, not fixed by us.*

**PReLU** (Parametric Rectified Linear Unit) — an improved version of Leaky ReLU:

$$\boxed{f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}}$$

*where **α** is a **trainable parameter** — learned automatically during backpropagation*

![PReLU Activation Function](https://media.geeksforgeeks.org/wp-content/uploads/20250528125143444422/Activation-functions-in-Neural-Networks.webp)

*Figure 6 — PReLU: adaptive slope α is learned by the model — more flexible than Leaky ReLU*

---

### 🧠 Intuition (Easy Way)

```
ReLU:        Negative values → KILLED         (output = 0)
Leaky ReLU:  Negative values → tiny leak      (0.01x, fixed by us)
PReLU:       "I'll learn the best leak slope myself." 🤖  (α is trainable)
```

---

### ✅ Advantages vs ⚠️ Limitations

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| Fixes dead neurons (like Leaky ReLU) | Extra parameters — α adds trainable values |
| **Adaptive** — slope is learned, not fixed | Risk of **overfitting** if dataset is small |
| Better accuracy — often improves CNNs | Slightly more complex than plain ReLU |

---
---

## 7️⃣ Swish Activation Function

> *Smooth, non-linear, and introduced by Google — often outperforms ReLU in deep networks.*

$$\boxed{f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}}$$

Where $\sigma(x)$ is the sigmoid function — so: **Swish = x × Sigmoid(x)**

![Swish Activation Function Graph](https://media.geeksforgeeks.org/wp-content/uploads/20231004125429/Swish.jpg)

*Figure 7 — Swish for various β values: smooth non-monotonic curve — linear at β→0, approaches ReLU at β→∞*

---

### 🧠 Intuition (Easy Way)

```
For large positive inputs   →  output ≈ input     (like ReLU) ✅
For large negative inputs   →  output is small     (not strictly 0, like Leaky ReLU) ✅
Around zero                 →  curve is SMOOTH     (not sharp like ReLU) ✅
```

> This smoothness makes training deep networks **easier and more stable**.

---

### ✅ Advantages vs ⚠️ Limitations

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| **Smooth curve** → better gradient flow, no sharp jumps | **More computation** (requires sigmoid internally) |
| **Non-monotonic** → can adapt to complex patterns | Not always better than ReLU (problem-dependent) |
| Often **improves accuracy over ReLU** in deep nets | Slight risk of slower training vs simple ReLU |

---
---

## 📐 All Formulas — Quick Reference Card

$$\boxed{\text{Linear:} \quad f(x) = x}$$

$$\boxed{\text{Sigmoid:} \quad f(x) = \frac{1}{1+e^{-x}}}$$

$$\boxed{\text{Tanh:} \quad f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}}$$

$$\boxed{\text{ReLU:} \quad f(x) = \max(0, x)}$$

$$\boxed{\text{Leaky ReLU:} \quad f(x) = \begin{cases} x & x > 0 \\ 0.01x & x \leq 0 \end{cases}}$$

$$\boxed{\text{PReLU:} \quad f(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases} \quad (\alpha \text{ is trainable})}$$

$$\boxed{\text{Swish:} \quad f(x) = x \cdot \sigma(x) = \frac{x}{1+e^{-x}}}$$

---

## 🎯 How to Choose the Right Activation Function

```
Is this the OUTPUT LAYER?
│
├── YES → Regression?       ──► Linear
│         Binary Classif.?  ──► Sigmoid
│         Multi-class?      ──► Softmax
│
└── NO (Hidden Layer)
    │
    ├── Start with           ──► ReLU        (fast, simple, works great)
    │
    ├── Dead Neurons?        ──► Leaky ReLU  (0.01 slope for x < 0)
    │
    ├── Need adaptability?   ──► PReLU       (α is learned automatically)
    │
    └── Want state-of-art?   ──► Swish       (Google's choice for deep nets)
```

---

## ⚠️ Vanishing Gradient — Quick Verdict

| Activation | Gradient Behavior | Verdict |
|---|---|:---:|
| **Sigmoid** | Gradient → ~0 for large/small inputs | ❌ Bad in deep nets |
| **Tanh** | Better than sigmoid, but still vanishes | ⚠️ OK in shallow nets |
| **ReLU** | Gradient = 1 for x > 0, 0 for x < 0 | ✅ Mostly Good |
| **Leaky ReLU** | Small gradient even for x < 0 | ✅ Better |
| **PReLU** | Learned gradient for x < 0 | ✅ Adaptive |
| **Swish** | Smooth gradient everywhere | ✅ Best for deep |

---

*📝 These Notes are Written and compiled by Mr. Bhavya Kansal for Understanding Deep Learning from Scratch*
*🚀 Part of IITR and NIELIT Internship Journey  — Built in Patiala, Made in India 🇮🇳*