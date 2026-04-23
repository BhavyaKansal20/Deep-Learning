# 🧠 Backpropagation in Neural Networks

> **Last Updated:** 9 Feb, 2026 &nbsp;|&nbsp; 📚 Deep Learning Series &nbsp;|&nbsp; ⚡ Core Algorithm

---

## 🔍 What is Backpropagation?

**Backpropagation** *(short for Backward Propagation of Errors)* is the **core training algorithm** of neural networks. It minimizes the difference between predicted and actual outputs by:

1. Propagating errors **backward** through the network
2. Using the **chain rule of calculus** to compute gradients
3. **Iteratively updating** weights and biases

> Combined with optimization techniques like **Gradient Descent**, backpropagation enables the model to reduce loss across epochs and effectively learn complex patterns from data.

---

![Backpropagation Overview](https://media.geeksforgeeks.org/wp-content/uploads/20250701163824448467/Backpropagation-in-Neural-Network-1.webp)

*Figure 1 — Overview of the Backpropagation process in a neural network*

---

## ⭐ Why Backpropagation Matters

| Feature | Description |
|---|---|
| ⚡ **Efficient Weight Update** | Computes gradients of the loss function w.r.t. each weight using the chain rule — enabling efficient updates |
| 📈 **Scalability** | Scales well to networks with multiple layers and complex architectures, making **deep learning feasible** |
| 🤖 **Automated Learning** | The model adjusts itself automatically to optimize performance — no manual tuning needed |

---

## ⚙️ Working of the Backpropagation Algorithm

The algorithm operates in **two main passes:**

```
INPUT DATA ──► FORWARD PASS ──► PREDICTION ──► ERROR CALC ──► BACKWARD PASS ──► WEIGHT UPDATE
     ▲                                                                                  │
     └──────────────────────── REPEAT UNTIL CONVERGENCE ◄──────────────────────────────┘
```

---

## 📤 Step 1 — Forward Pass

In the **forward pass**, input data flows through the network layer by layer:

- Input data is fed into the **input layer**
- Inputs combined with **weights** are passed to hidden layers
- In a network with two hidden layers `h1` and `h2` → output of `h1` becomes input of `h2`
- A **bias** is added to the weighted inputs before applying any activation function
- Each hidden layer computes the **weighted sum** `a`, then applies an **activation function** (e.g., ReLU) to get output `o`
- Final layer uses **softmax** to convert weighted outputs into probabilities for classification

---

![Forward Pass Diagram](https://media.geeksforgeeks.org/wp-content/uploads/20250701163954688803/Backpropagation-in-Neural-Network-2.webp)

*Figure 2 — Forward pass: data flows from input → hidden layers → output*

---

## 📥 Step 2 — Backward Pass

In the **backward pass**, the error is propagated back through the network to update weights and biases.

### 🔢 Error Calculation — Mean Squared Error (MSE)

$$\text{MSE} = (\text{Predicted Output} - \text{Actual Output})^2$$

Once the error is calculated:
- **Gradients** are computed using the **chain rule**
- Gradients indicate *how much* each weight and bias should be adjusted
- The pass continues **layer by layer** — ensuring the network improves progressively
- The **derivative of the activation function** plays a crucial role in this computation

---

## 🧮 Full Worked Example — Step by Step

> **Setup:**
> - Activation Function: **Sigmoid**
> - Target Output: **0.5**
> - Learning Rate (η): **1**

---

![Network Diagram with Weights](https://media.geeksforgeeks.org/wp-content/uploads/20250701164029130520/Backpropagation-in-Neural-Network-3.webp)

*Figure 3 — Neural network with initial weights for the backpropagation example*

---

## ➡️ Forward Propagation

### 1️⃣ Weighted Sum at Each Node

$$a_j = \sum (w_{i,j} \times x_i)$$

| Symbol | Meaning |
|---|---|
| $a_j$ | Weighted sum of all inputs and weights at node $j$ |
| $w_{i,j}$ | Weight between the $i^{th}$ input and $j^{th}$ neuron |
| $x_i$ | Value of the $i^{th}$ input |

**Output after activation:**

$$o_j = \text{activation\_function}(a_j)$$

---

### 2️⃣ Sigmoid Activation Function

Returns a value between **0 and 1**, introducing **non-linearity** into the model:

$$y_j = \frac{1}{1 + e^{-a_j}}$$

---

![Sigmoid Function Diagram](https://media.geeksforgeeks.org/wp-content/uploads/20250701164114106895/Backpropagation-in-Neural-Network-4.webp)

*Figure 4 — Sigmoid function applied to compute outputs y₃, y₄ and y₅*

---

### 3️⃣ Computing Node Outputs

#### 🔵 At Hidden Node `h1` → computing `y₃`

$$a_1 = (w_{1,1} \cdot x_1) + (w_{2,1} \cdot x_2) = (0.2 \times 0.35) + (0.2 \times 0.7) = 0.21$$

$$y_3 = F(a_1) = \frac{1}{1 + e^{-0.21}} = \boxed{0.56}$$

---

#### 🔵 At Hidden Node `h2` → computing `y₄`

$$a_2 = (w_{1,2} \cdot x_1) + (w_{2,2} \cdot x_2) = (0.3 \times 0.35) + (0.3 \times 0.7) = 0.315$$

$$y_4 = F(0.315) = \frac{1}{1 + e^{-0.315}} = \boxed{0.57}$$

---

#### 🔵 At Output Node `O3` → computing `y₅`

$$a_3 = (w_{1,3} \cdot y_3) + (w_{2,3} \cdot y_4) = (0.3 \times 0.57) + (0.9 \times 0.59) = 0.702$$

$$y_5 = F(0.702) = \frac{1}{1 + e^{-0.702}} = \boxed{0.67}$$

---

![Computed Values y3, y4, y5](https://media.geeksforgeeks.org/wp-content/uploads/20250701164956768059/Backpropagation-in-Neural-Network-5.webp)

*Figure 5 — Network with computed values: y₃ = 0.56, y₄ = 0.57, y₅ = 0.67*

---

### 4️⃣ Error Calculation

> **Predicted:** `y₅ = 0.67` &nbsp;|&nbsp; **Target:** `0.5`

$$\text{Error}_j = y_{\text{target}} - y_5 = 0.5 - 0.67 = \mathbf{-0.17}$$

> ⚠️ We now use this error value to **backpropagate** through the network.

---

## ⬅️ Backward Propagation

### 1️⃣ Weight Update Formula

$$\Delta w_{ij} = \eta \times \delta_j \times O_j$$

| Symbol | Meaning |
|---|---|
| $\delta_j$ | Error term for each unit |
| $\eta$ | Learning rate |
| $O_j$ | Output of the neuron |

---

### 2️⃣ Output Unit Error Term — `δ₅` at O3

$$\delta_5 = y_5 (1 - y_5)(y_{\text{target}} - y_5)$$

$$= 0.67 \times (1 - 0.67) \times (-0.17) = \boxed{-0.0376}$$

---

### 3️⃣ Hidden Unit Error Terms

#### At `h1` — computing `δ₃`

$$\delta_3 = y_3(1 - y_3)(w_{1,3} \times \delta_5)$$

$$= 0.56 \times (1 - 0.56) \times (0.3 \times -0.0376) = \boxed{-0.0027}$$

#### At `h2` — computing `δ₄`

$$\delta_4 = y_4(1 - y_4)(w_{2,3} \times \delta_5)$$

$$= 0.59 \times (1 - 0.59) \times (0.9 \times -0.0376) = \boxed{-0.00819}$$

---

### 4️⃣ Weight Updates

#### 🔄 Hidden → Output Layer Weights

$$\Delta w_{2,3} = 1 \times (-0.0376) \times 0.59 = -0.022184$$

$$w_{2,3}^{\text{(new)}} = -0.022184 + 0.9 = \boxed{0.877816}$$

---

#### 🔄 Input → Hidden Layer Weights

$$\Delta w_{1,1} = 1 \times (-0.0027) \times 0.35 = -0.000945$$

$$w_{1,1}^{\text{(new)}} = -0.000945 + 0.2 = \boxed{0.199055}$$

---

### 📊 All Updated Weights Summary

| Weight | Old Value | Δ (Change) | New Value |
|:---:|:---:|:---:|:---:|
| $w_{1,1}$ | 0.200 | −0.000945 | **0.199055** |
| $w_{2,1}$ | 0.200 | — | **0.269445** |
| $w_{1,2}$ | 0.300 | — | **0.273225** |
| $w_{2,2}$ | 0.300 | — | **0.18534** |
| $w_{1,3}$ | 0.300 | — | **0.086615** |
| $w_{2,3}$ | 0.900 | −0.022184 | **0.877816** |

---

![Updated Weights Network](https://media.geeksforgeeks.org/wp-content/uploads/20250918132203487723/backpropagation_in_neural_network_11.webp)

*Figure 6 — Network with updated weights after one backward pass*

---

## 🔁 Next Forward Pass — After Weight Update

With the newly updated weights, running the forward pass again gives:

| Node | Previous Output | New Output |
|:---:|:---:|:---:|
| $y_3$ | 0.56 | **0.57** |
| $y_4$ | 0.59 | **0.56** |
| $y_5$ | 0.67 | **0.61** |

**New Error:**

$$\text{Error} = y_{\text{target}} - y_5 = 0.5 - 0.61 = \mathbf{-0.11}$$

> 📉 Error reduced from **−0.17 → −0.11** ✅
> Since `y₅ = 0.61` is still not the target `0.5`, the process **continues iterating**.

---

## 🔄 The Full Training Loop

```
┌─────────────────────────────────────────────────────────┐
│               BACKPROPAGATION TRAINING LOOP             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  EPOCH 1                                                │
│  ├── Forward Pass  →  y₅ = 0.67  →  Error = −0.17      │
│  └── Backward Pass →  Update all weights                │
│                                                         │
│  EPOCH 2                                                │
│  ├── Forward Pass  →  y₅ = 0.61  →  Error = −0.11      │
│  └── Backward Pass →  Update all weights                │
│                                                         │
│  EPOCH N                                                │
│  ├── Forward Pass  →  y₅ ≈ 0.50  →  Error ≈ 0.00  ✅   │
│  └── CONVERGED — Training Complete!                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 📐 Key Formulas — Quick Reference Card

$$\boxed{\text{Weighted Sum:} \quad a_j = \sum_i w_{i,j} \cdot x_i}$$

$$\boxed{\text{Sigmoid:} \quad y_j = \frac{1}{1+e^{-a_j}}}$$

$$\boxed{\text{MSE:} \quad \text{Loss} = (y_{\text{target}} - y_{\text{pred}})^2}$$

$$\boxed{\text{Output Error:} \quad \delta_5 = y_5(1-y_5)(y_{\text{target}}-y_5)}$$

$$\boxed{\text{Hidden Error:} \quad \delta_j = y_j(1-y_j)\sum_k w_{jk} \cdot \delta_k}$$

$$\boxed{\text{Weight Update:} \quad w_{ij}^{\text{new}} = w_{ij} + \eta \cdot \delta_j \cdot O_i}$$

---

## 🧩 Concept Summary

```
BACKPROPAGATION = CHAIN RULE + GRADIENT DESCENT

         ┌──────────────┐      ┌───────────────┐
INPUT ──►│ FORWARD PASS │─────►│ COMPUTE LOSS  │
         └──────────────┘      └───────┬───────┘
                                       │ Error
                                       ▼
         ┌──────────────┐      ┌───────────────┐
WEIGHTS  │ UPDATE W & B │◄─────│ BACKWARD PASS │
UPDATED  └──────────────┘      └───────────────┘
                                 (Chain Rule)
```

> 🎯 **Goal:** Drive the **Loss → 0** by iteratively tuning every weight and bias in the network using the gradient signal flowing backward from the output.

---

*📝 These Notes are Written and compiled by Mr. Bhavya Kansal for Understanding Deep Learning from Scratch*
*🚀 Part of IITR and NIELIT Internship Journey  — Built in Patiala, Made in India 🇮🇳*