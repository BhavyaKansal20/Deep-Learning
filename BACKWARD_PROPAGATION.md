# 🔁 **Backpropagation in Neural Networks**


Backpropagation (Backward Propagation of Errors) is a core algorithm used to train neural networks by minimizing the difference between **predicted output** and **actual output**.

It works by:
- Propagating errors **backward**
- Using the **chain rule of calculus**
- Updating **weights & biases iteratively**

Combined with optimization techniques like **gradient descent**, it enables models to **learn complex patterns efficiently**.

---

## 🧠 Backpropagation Overview

![Backpropagation](https://media.geeksforgeeks.org/wp-content/uploads/20250701163824448467/Backpropagation-in-Neural-Network-1.webp)

---

## ⚡ Why Backpropagation Matters

- ✅ Efficient Weight Updates  
- 🚀 Scalable to deep networks  
- 🤖 Fully automated learning  

> 🔥 Without backpropagation → neural networks don’t learn.

---

# ⚙️ Working of Backpropagation Algorithm

## 🔹 1. Forward Pass

- Input flows layer by layer  
- Weighted sum + activation applied  

---

![Forward inside Backprop](https://media.geeksforgeeks.org/wp-content/uploads/20250701163954688803/Backpropagation-in-Neural-Network-2.webp)

---

## 🔹 2. Backward Pass

- Error computed  
- Gradients calculated  
- Weights updated  

---

## 📉 Error Function (MSE)

\[
\boxed{
MSE = (Predicted - Actual)^2
}
\]

---

# 🧪 Example Setup

- Activation = **Sigmoid**
- Target Output = **0.5**
- Learning Rate = **1**

---

![Example Network](https://media.geeksforgeeks.org/wp-content/uploads/20250701164029130520/Backpropagation-in-Neural-Network-3.webp)

---

# 🔹 Forward Propagation

## 1️⃣ Initial Calculation (Weighted Sum)

\[
\boxed{
a_j = \sum (w_{i,j} \cdot x_i)
}
\]

**Where:**
- \( a_j \) → weighted sum  
- \( w_{i,j} \) → weights  
- \( x_i \) → inputs  

---

## 2️⃣ Output After Activation

\[
\boxed{
o_j = activation(a_j)
}
\]

---

## 3️⃣ Sigmoid Function

\[
\boxed{
y_j = \frac{1}{1 + e^{-a_j}}
}
\]

---

![Intermediate Outputs](https://media.geeksforgeeks.org/wp-content/uploads/20250701164114106895/Backpropagation-in-Neural-Network-4.webp)

---

## 4️⃣ Computation

### Hidden Layer h1

\[
\boxed{
a_1 = (0.2 \cdot 0.35) + (0.2 \cdot 0.7) = 0.21
}
\]

\[
\boxed{
y_3 = \frac{1}{1 + e^{-0.21}} = 0.56
}
\]

---

### Hidden Layer h2

\[
\boxed{
a_2 = (0.3 \cdot 0.35) + (0.3 \cdot 0.7) = 0.315
}
\]

\[
\boxed{
y_4 = \frac{1}{1 + e^{-0.315}}
}
\]

---

### Output Layer

\[
\boxed{
a_3 = (0.3 \cdot 0.57) + (0.9 \cdot 0.59) = 0.702
}
\]

\[
\boxed{
y_5 = \frac{1}{1 + e^{-0.702}} = 0.67
}
\]

---

![Final Outputs](https://media.geeksforgeeks.org/wp-content/uploads/20250701164956768059/Backpropagation-in-Neural-Network-5.webp)

---

## ❌ Error Calculation

\[
\boxed{
Error = y_{target} - y_5 = 0.5 - 0.67 = -0.17
}
\]

---

# 🔁 Backward Propagation

## 1️⃣ Gradient Update Rule

\[
\boxed{
\Delta w_{ij} = \eta \cdot \delta_j \cdot O_j
}
\]

---

## 2️⃣ Output Layer Error

\[
\boxed{
\delta_5 = y_5(1 - y_5)(y_{target} - y_5)
}
\]

\[
\boxed{
= 0.67(1 - 0.67)(-0.17) = -0.0376
}
\]

---

## 3️⃣ Hidden Layer Errors

### h1

\[
\boxed{
\delta_3 = y_3(1 - y_3)(w_{1,3} \cdot \delta_5)
}
\]

\[
\boxed{
= 0.56(1 - 0.56)(0.3 \cdot -0.0376) = -0.0027
}
\]

---

### h2

\[
\boxed{
\delta_4 = y_4(1 - y_4)(w_{2,3} \cdot \delta_5)
}
\]

\[
\boxed{
= 0.59(1 - 0.59)(0.9 \cdot -0.0376) = -0.00819
}
\]

---

## 4️⃣ Weight Updates

### Hidden → Output

\[
\boxed{
\Delta w_{2,3} = 1 \cdot (-0.0376) \cdot 0.59 = -0.022184
}
\]

\[
\boxed{
w_{2,3}^{new} = 0.9 - 0.022184 = 0.877816
}
\]

---

### Input → Hidden

\[
\boxed{
\Delta w_{1,1} = 1 \cdot (-0.0027) \cdot 0.35 = -0.000945
}
\]

\[
\boxed{
w_{1,1}^{new} = 0.2 - 0.000945 = 0.199055
}
\]

---

### Updated Weights

- \( w_{1,2}^{new} = 0.273225 \)  
- \( w_{1,3}^{new} = 0.086615 \)  
- \( w_{2,1}^{new} = 0.269445 \)  
- \( w_{2,2}^{new} = 0.18534 \)  

---

![Updated Weights](https://media.geeksforgeeks.org/wp-content/uploads/20250918132203487723/backpropagation_in_neural_network_11.webp)

---

# 🔄 Iteration Continues

\[
\boxed{
y_3 = 0.57,\quad y_4 = 0.56,\quad y_5 = 0.61
}
\]

---

## 📉 New Error

\[
\boxed{
Error = 0.5 - 0.61 = -0.11
}
\]

---

# 🧠 Final Insight

Backpropagation = **iterative error correction loop**

1. Forward pass  
2. Compute error  
3. Backward pass  
4. Update weights  

Repeat until convergence.

---

# ⚠️ Reality Check

If you can’t:
- Derive gradients  
- Understand chain rule  
- Explain why sigmoid derivative matters  

Then you're not building AI — you're just executing code.

---

# 🚀 Core Takeaway

Forward Propagation = **Prediction**  
Backpropagation = **Learning**

Together = **Real Intelligence**

---