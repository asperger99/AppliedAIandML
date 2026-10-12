> Supervised Learning: Support vector machines

### 1. Supervised Learning

#### Basic idea

Supervised learning learns a relationship between:

- Features `X` → input variables
- Labels `Y` → expected output
- Model `f` → relationship between `X` and `Y`

General form:

```text
Y = f(X)
```

Examples of models:

- Linear Regression
- Logistic Regression
- Linear Classifier
- Decision Tree
- Neural Network

#### Features

Features are measurable attributes of the objects being classified.

Types:

- Numerical → age, salary, temperature
- Categorical → gender, city, product type
- Ordinal → low, medium, high

#### Model

A model is an imagined mathematical relationship between input and output.

Example:

```text
y = 2x + 3
```

More complex example:

```text
P(Y=1) = 1 / (1 + exp(-3x₁ + x₂))
```

There is usually no predefined “true” mathematical relationship between features and labels.

The ML model tries to find a useful approximation.

#### Learning Algorithm

The learning algorithm estimates the model’s parameters by minimizing a loss function.

```text
Training Data
     ↓
Learning Algorithm
     ↓
Model Parameters
     ↓
Trained Model
```

### 2. Concept of ML Model

An ML model is a mathematical description that attempts to explain the relationship between features and labels.

```text
Label y = f(features x)
```

The best model depends on:

- Dataset
- Choice of features
- Distribution/layout of data
- Model complexity

There is usually no single “true model”. We try to find the model that performs best for the problem.

### 3. Linear Classifiers

A linear classifier separates two classes using a:

- Line → 2D
- Plane → 3D
- Hyperplane → higher dimensions
- ![[Pasted image 20261005082749.png]]

Example:

```text
Class +1 | Class -1
         |
---------|---------  ← Decision boundary
         |
```

#### Suitable features

Basic linear classifiers work directly with numerical features.

Categorical features generally need to be converted into numerical representations first.

### 4. Linear Classifier Equation

#### 2D equation

A linear classifier in 2D can be represented as a straight line:

```text
w₁x₁ + w₂x₂ + w₃ = 0
```

Where:

- `x₁, x₂` → features
- `w₁, w₂` → feature weights
- `w₃` → bias

The bias can be viewed as the coefficient of a default feature (whose value is always `1`).

#### General form

```text
W · X = 0
```

For a D-dimensional feature space:

```text
W and X → D + 1 dimensions
```

Prediction:

```text
y = sign(W · X)
```

The `sign()` function gives:

```text
+1 → one class
-1 → other class
```

### 5. Linear Separability

A dataset is linearly separable if at least one linear classifier can perfectly separate the two classes.
![[Pasted image 20261005083005.png]]

```text
Linearly separable
→ A straight line/hyperplane can separate the classes
```

Not every dataset is linearly separable.

Example:

```text
X X
O O
```

can potentially be separated with a line.

But patterns such as XOR cannot be perfectly separated using a single linear boundary.
![[Pasted image 20261005083029.png]]

### 6. Perceptron Algorithm

The Perceptron algorithm can find a linear classifier when the dataset is linearly separable.

Basic idea:

1. Start with an initial classifier.
2. Check each training example.
3. If an example is misclassified, update the weights.
4. Continue until all examples are correctly classified or maximum iterations are reached.

#### Update rule

For training example `(Xᵢ, yᵢ)`:

```text
if yᵢ × sign(W · Xᵢ) < 0:(wrong classification)

    W = W + yᵢXᵢ
```

Where:

```text
yᵢ ∈ {+1, -1}
```

#### Important property

If the dataset is linearly separable, the Perceptron will find a perfect linear classifier in a finite number of iterations.

### 7. Problem with Linear Classifiers

#### Linear classifiers are not unique

A dataset can have many different lines/hyperplanes that correctly separate the classes.

However, not all separating boundaries are equally good.

A boundary very close to the data points is sensitive to:

- Noise
- Small changes in data
- Outliers

Therefore, we prefer a classifier that stays as far away from the training examples as possible.

This leads to the concept of margin.

### 8. Margin

Margin is the minimum perpendicular distance between the decision boundary and the closest training example.

```text
Large margin → safer separation
Small margin → more sensitive to noise
```

Example:
![[Pasted image 20261005223658.png]]

```text
Solution 1:
Blue margin = 1
Red margin  = 1

Solution 2:
Blue margin = 6
Red margin  = 4
```

Solution 2 is preferable because it has a larger margin.

### 9. Max-Margin Linear Classifier

**Goal**

- Find a linear classifier that maximizes the distance from both classes.

```text
Maximum margin
      ↓
Better separation
      ↓
More robust to small noise
```

- The optimal decision boundary lies midway between two parallel boundaries surrounding the classes i.e. on equal distance from nearest data point from both classes.
- ![[Pasted image 20261006081001.png]]

- We can imagine two parallel `boundaries` of both classes where at least one data point touches this boundary.
  Optimal max-margin classifier should lie exactly between them.
- ![[Pasted image 20261006081030.png]]

**Total margin**
- Geometrical distance between two parallel `boundaries` of both classes where at least one data point touches this boundary.
- Minimum distance formula for two lines 
	- Y = MX + C1
	- Y = MX + C2
	- distance = 
	  $$d = \frac{\vert{}C_2 - C_1\vert{}}{\sqrt{M^2}}$$
```text
Total Margin = 2k / ||W||
```
Where:
- ||W|| is L2 norm
- ||W|| = $\sqrt{w_0^2 + w_1^2 + w_2^2 + \dots + w_D^2}$

>The objective is to maximize the margin.

### 10. Optimization Formulation

#### Objective

Find the weight vector $\mathbf{W}$ for which the total margin is maximized, ensuring all points lie on the correct side away from the decision boundary.

---

#### Mathematical Formulation

1. **Total Margin Formula:**
   $$\text{Total Margin} = \frac{2k}{\|\mathbf{W}\|}$$

2. **Formulating the Objective:**
   Maximizing the margin $\frac{2k}{\|\mathbf{W}\|}$ is equivalent to minimizing the denominator $\|\mathbf{W}\|$
   $$\max_{\mathbf{W}} \frac{2k}{\|\mathbf{W}\|} \iff \min_{\mathbf{W}} \frac{1}{2} \|\mathbf{W}\|^2$$

   *(Note: We square $\|\mathbf{W}\|$ and scale by $\frac{1}{2}$ to obtain a smooth, differentiable quadratic objective function.)*
   
   *Even though we have changed signature of function, point at which minima will be achieved will remain same. And this new form is mathematically easier to calculate.*

---

#### Primal Optimization Problem

$$\min_{\mathbf{W}} \frac{1}{2} \|\mathbf{W}\|^2$$

$$\text{subject to } \quad y_i(\mathbf{W} \cdot \mathbf{X}_i) \ge k, \quad \forall i \in \{1, \dots, N\}$$

Where:
- $y_i = +1, \quad \mathbf{W} \cdot \mathbf{X}_i \ge k \quad \text{(for red points)}$
- $y_i = -1, \quad \mathbf{W} \cdot \mathbf{X}_i \le -k \quad \text{(for blue points)}$

---

#### Key Takeaways

- **Minimize Model Complexity:** Minimizing $\frac{1}{2} \|\mathbf{W}\|^2$ keeps weight magnitudes small and prevents overfitting.
- **Correct Classification:** The constraint $y_i(\mathbf{W} \cdot \mathbf{X}_i) \ge k$ enforces that all samples lie on or outside their respective boundary lines ($\mathbf{W} \cdot \mathbf{X} = k$ and $\mathbf{W} \cdot \mathbf{X} = -k$).
- **Problem Type:** This forms a **Convex Quadratic Programming (QP)** problem with linear inequality constraints.

### 11. Outliers and Non-Separable Data

- Real-world datasets may contain:
	- Outliers
	- Noise
	- Incorrect labels
	- Overlapping classes
- ![[Pasted image 20261007075705.png]]

Therefore, perfect classification may not be possible.

Some points may violate the margin constraints.

### 12. Soft-Margin Formulation

#### Idea

- Instead of forcing every point to satisfy the margin constraint, allow some violations using a slack variable.

- The slack variable is represented as:

```text
δᵢ
```

- Optimization:

```text
Minimize:

½ ||W||² + C Σδᵢ
```

Subject to:

```text
yᵢ(W · Xᵢ) + δᵢ ≥ k
```

#### Meaning of slack

```text
δ = 0
→ point satisfies the required margin

0 < δ < k
→ point violates the margin but may still be correctly classified

δ > k
→ point may be misclassified
```
- ![[Pasted image 20261009082240.png]]
#### Role of C

`C` controls the trade-off between:

```text
Large margin
        vs
Training errors
```

High `C`:

```text
Less tolerance for errors
→ tries harder to classify training points correctly
→ potentially smaller margin
```

Low `C`:

```text
More tolerance for errors
→ allows more violations
→ generally encourages a larger margin
```

### 13. Support Vector Machine

SVM finds a maximum-margin linear classifier while allowing controlled violations through the soft-margin formulation.

The solution can be represented as:

```text
W = Σ αᵢyᵢXᵢ
```

Where `αᵢ` is an additional scalar associated with each training example.

Sum of product of α and y is always zero.
```
Σαᵢyᵢ = 0
Eg:
Y  α
1  0
1  7
-1 3.5
-1 3.5


```
#### Support Vectors

Most training examples have:

```text
αᵢ = 0
```

Only important points have non-zero `αᵢ`.

These points are called:

```text
Support Vectors
```

They are typically:

- Closest to the optimal linear classifier
- Or misclassified by the optimal linear classifier

The important idea:

```text
Decision boundary mainly depends on support vectors.
```

### 14. SVM Prediction (Classification by SVM)

For a new test point:

```text
Y = sign(WᵀXtest)
```

Since:

```text
W = Σ αᵢyᵢXᵢ
```

we get:

```text
Y = sign(Σ αᵢyᵢ(XᵢᵀXtest))
```

So prediction essentially requires calculating the dot product between the test point and the support vectors.


- **Objective:** Given a new feature vector $X_{\text{test}}$, the goal is to predict its corresponding class label $Y$.
    
- **Weight Vector Formulation:** The weight vector $W$ is represented as a linear combination of training data:
    $$W = \sum_{i} \alpha_i Y_i X_i$$
    
    where $\alpha_i \ge 0$ are the Lagrange multipliers. Only support vectors have non-zero $\alpha_i$ ($\alpha_i > 0$).
    
- **Prediction Formula:** Substituting $W$ into the decision boundary equation $Y = \text{sign}(W^T X_{\text{test}})$ yields:
    $$Y = \text{sign}\left(\sum_{i} \alpha_i Y_i X_i^T X_{\text{test}}\right)$$
- **Key Takeaway:** Classifying a new test sample fundamentally relies on taking the **dot product** of $X_{\text{test}}$ with the support vectors ($\alpha_i > 0$). This dot-product formulation is crucial because it allows SVMs to easily extend to non-linear classification using the **kernel trick**.

**NOTE**
- T denotes Transpose of a vector in above formula
- To multiply two column vectors together to calculate their **dot product** (scalar product), you must transpose the first vector into a **row vector**
$$W = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_d \end{bmatrix}, \quad X_{\text{test}} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix}$$
$$W^T = \begin{bmatrix} w_1 & w_2 & \dots & w_d \end{bmatrix}$$
$$W^T X_{\text{test}} = w_1 x_1 + w_2 x_2 + \dots + w_d x_d = W \cdot X_{\text{test}}$$

### 15. Non-Linear SVM

Some datasets cannot be separated using a straight line/hyperplane.

However, they may become linearly separable after transforming the features.

```text
Original feature space
        ↓
Non-linear transformation φ(X)
        ↓
New feature space
        ↓
Linear separation
```

After transformation 
- Y = sign(Σ αᵢyᵢ(XᵢᵀXtest)) 
  SVM prediction becomes
- Y = sign(Σ αᵢyᵢφ(Xᵢ)ᵀφ(Xtest))
### 16. Kernel Trick

- We don't need to find transformation function `φ`
- Instead of explicitly calculating the transformed features `φ(X)`, use a kernel function.

```text
K(Xᵢ, Xtest)
```

- The kernel effectively calculates similarity in the transformed feature space.
  i.e. kernal function gives the dot product of transformed features
- SVM prediction becomes:

```text
Y = sign(Σ αᵢyᵢK(Xᵢ, Xtest))
```

**Kernelized SVM**

```text
Non-linear problem
       ↓
Kernel function
       ↓
Implicit feature transformation
       ↓
Linear separation in transformed space
```

**Kernel Function examples**

- Multiple kernels can be combined to create the final decision function.
- Different kernels may work better for different datasets.
- ![[Pasted image 20261010122345.png]]

### 17. Bias of Models

- Suppose we train the same model on multiple different training datasets and get different learned models fi:

```text
D1 → f₁
D2 → f₂
D3 → f₃
...
DM → fM
```

- For the same input `X`, each model can produce a different prediction.

**Ensemble mean prediction**

```text
g(X) = (1/M) × [f₁(X) + f₂(X) + ... + fM(X)]
```

Bias:
```text
Bias(X) = Y - g(X)
```

Where:
- `Y` → true value (here Y is real number as we are considering regression)
- `g(X)` → average model prediction


**Low vs High Bias**

Low bias:
```text
Mean prediction ≈ True value
```

High bias:
```text
Mean prediction far from True value
```

- High bias usually indicates that the model is not suitable or is too simple for the data.
- ![[Pasted image 20261010130533.png]]

### 18. Variance of Models

- Variance measures how much model predictions change when the training dataset changes.
- ![[Pasted image 20261010130002.png]]
- Variance can be represented conceptually as the average squared deviation of individual model predictions from the ensemble mean.

```text
Variance(X)
= Average[(fᵢ(X) - g(X))²]

Ensemble Mean Prediction:
g(X) = (1/M)*(f1(X) + f2(X) + ……. + fM(X))

Variance(X) = (1/M)*(Y – g(X))2
```

**Low vs High Variance**
- ![[Pasted image 20261010130228.png]]
- If models trained on different datasets give similar predictions:
```text
Low variance
```

- If their predictions are widely different:
```text
High variance
```

- High variance usually indicates overfitting.

### 19. Bias vs Variance

There are four possible situations:

| **Bias** | **Variance** | **Result**   |
| -------- | ------------ | ------------ |
| High     | Low          | Underfitting |
| High     | High         | Poor model   |
| Low      | Low          | Ideal        |
| Low      | High         | Overfitting  |

**High Bias**

```text
Predictions consistently differ from true value, irrespective of training set
→ Model is too simple / unsuitable
→ Underfitting
```

**High Variance**

```text
Predictions change significantly with training data
→ Model learns training data too closely
→ Overfitting
```

**Ideal Model**

```text
Low Bias + Low Variance
```

### **20. Bias-Variance Tradeoff**

#### **Error decomposition**

For the simplified formulation in the slides:

```text
Error(X) = Bias(X)² + Variance(X)
```

Therefore, high prediction error can come from:

```text
High Bias
    ↓
Underfitting
```

or:

```text
High Variance
    ↓
Overfitting
```

#### **Simple intuition**

```text
Too simple model
      ↓
High Bias
      ↓
Underfitting

Good complexity
      ↓
Low Bias + Low Variance
      ↓
Good generalization

Too complex model
      ↓
High Variance
      ↓
Overfitting
```

### **21. Bias-Variance Numerical Example**

#### **Example 1**

Suppose:

```text
X = [3, 5]
Y = 12
```

Four different training datasets produce predictions:

```text
D1 → 11.3
D2 → 11.3
D3 → 14.7
D4 → 11.3
```

Mean prediction:

```text
g(X) = 12.15
```

Bias:

```text
12 - 12.15 = -0.15
```

Variance from the example:

```text
2.17
```

The example represents:

```text
Low Bias + Low Variance
```

#### **Example 2**

Same model predictions, but true value changes:

```text
Y = 18
```

Mean prediction remains:

```text
12.15
```

Bias:

```text
18 - 12.15 = 5.85
```

The slide summarizes this as:

```text
High Bias + Low Variance
```

This suggests that the model may be systematically unsuitable for this data.

### 22. Class Imbalance

- Class imbalance occurs when one class has significantly more examples than another.
- Example:
```text
Blue = 12
Red  = 5
```

The classes are not equally represented.

#### Why it is a problem

- Consider two classifiers with total error = 3.
- ![[Pasted image 20261010131630.png]]
- Classifier 1:
```text
Blue: 0/12 errors
Red:  3/5 errors
```

- Classifier 2:
```text
Blue: 3/12 errors
Red:  0/5 errors
```

- Both have:
```text
Total Error = 3
```

>But the second classifier is fairer because it performs well on both classes.
  A model can achieve good overall accuracy while performing poorly on the minority class.

#### Adjusting Class Weights

**Problem**

- Standard algorithms may treat every training example equally.
- In an imbalanced dataset:

```text
Majority class → many examples
Minority class → few examples
```

- The model may therefore favor the majority class.

**Solution**

1. Give different importance to different classes.
   This encourages the model to pay more attention to minority-class errors.

```text
Minority class
→ Higher weight

Majority class
→ Lower weight
```

2. Adjust the frequencies of class
#### Oversampling

**Definition**

Oversampling increases the number of minority-class training examples.

```text
Minority class
     ↓
Create additional similar examples
     ↓
More balanced dataset
```

New examples should be inexact copies that are close to the original samples in feature space.
![[Pasted image 20261010132636.png]]

> Now we will always prefer second solution with total error 6

**Goal**

```text
Increase minority-class representation
→ Reduce bias toward majority class
```

#### Undersampling

**Definition**

Undersampling reduces the number of majority-class examples.

```text
Majority class
     ↓
Remove selected examples
     ↓
More balanced dataset
```

Preferably retaining examples closer to the decision boundary because they are more informative for classification.

![[Pasted image 20261010132839.png]]
> Now we will always prefer second solution with total error 0

**Goal**

```text
Reduce majority-class dominance
→ Give minority class relatively more influence
```

#### Oversampling vs Undersampling

|**Technique**|**What happens**|**Main benefit**|**Main risk**|
|---|---|---|---|
|Oversampling|Add minority examples|Retains majority data|Can increase dataset size / overfitting risk|
|Undersampling|Remove majority examples|Smaller dataset|May discard useful information|
|Class weighting|Increase minority-class importance|No need to duplicate/remove data|Requires suitable weights|

### 23. Key Takeaways

**Linear Classifier**

```text
Uses a line/hyperplane to separate classes.
```

**Perceptron**

```text
Updates weights whenever a point is misclassified.
```

**Margin**

```text
Distance between decision boundary and closest data points.
```

**Max-Margin Classifier**

```text
Finds the boundary with maximum separation from classes.
```

**SVM**

```text
Maximum-margin classifier
+ support vectors
+ soft-margin handling of violations
```

**Kernel SVM**

```text
Uses kernel functions to handle non-linear boundaries.
```

**Bias**

```text
How far the average prediction is from the true value.
High bias → underfitting.
```

**Variance**

```text
How much predictions change across different training datasets.
High variance → overfitting.
```

**Class Imbalance**

```text
One class has significantly fewer examples.
```

**Imbalance Solutions**

```text
Class weighting
Oversampling
Undersampling
```
