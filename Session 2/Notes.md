
>Supervised learning: linear and logistic regression

### Biological Neuron

- It receives information through its different Presynaptic cell(x1, x2, .., xn, similar to parameters of and ML model
- After that a chemical reaction happpens which produces output, similar to ML model function F(x)
- Further output is passed to other connected Neuron but with different intensity. Some connected neuron get more where as some get less information based on how much information is needed for each Neuron and It is decided by Synapses.
- ![[Pasted image 20260927101109.png]]

- Biological Neural Network is a connection of Biological Neurons.


**Artificial Neuron to Neural Network**
- ![[Pasted image 20260927101616.png]]
**Number of Neurons in a Biological Neural Network**
- ![[Pasted image 20260928084653.png]]

**Working of human brain**
- ![[Pasted image 20260927102137.png]]

- Th is threshold
- McCulloch said Neurons work with binary i.e. input, weights and output all are binary only threshold is real number.
- ![[Pasted image 20260927102338.png]]

**Hebbian's theory for Neuron**
- ![[Pasted image 20260929080700.png]]
- Later Hebbian said something is binary based assumption is not correct as Real world does not always work with binary instead he said it should be real number.
- He said input/output, weights, threshold all are real number.

**Threshold or Bias**
- x1w1 + x2w2 + .. xnwn >= th
- x1w1 + x2w2 + .. xnwn - th >= 0 
  here we shifted threshold to left and now it seems like threshold is 0
  we say threshold a bias parameter
- Why we need bias?
### Linear Models

#### Linear Classifier
- It classifies input into binary classes
- It works by finding a linear boundary—a line in 2D space, a plane in 3D, or a **hyperplane** in higher dimensions—that separates data points into different categories.
- ![[Pasted image 20260929081437.png]]


#### Linear Regression with Bias

- ![[Pasted image 20260929081812.png]]
- Above shown equation is **linear regression** as it gives output for continuous values of x
- Used to predict a continuous numerical value (e.g., house prices, sales revenue, temperature) based on one or more input features.
- y = b+w1x1
	- y: prediction
	- b: bias
	- w1: weight
	- x1: feature value
- b and w1 gets calculated from training
- weights is directly proportional to no of parameters i.e. o(n)
- It gives a hyper plane
- **Why we need Bias ?**
	- If we have no bias hyper plane will always go through origin i.e. when input is 0 then output will be 0 i.e. equation becomes y=wx
	- Real-world data rarely passes through the point $(0,0)$. For example, if you are predicting house prices based on size:
		- Even a 0-square-foot lot has a baseline land value (e.g., $\$50,000$).
		- Without bias, the model would force the predicted price of a 0-sq-ft house to be $\$0$, skewing the line and leading to inaccurate predictions across all house sizes.
### Polynomial Regression or Model
- Lineal function is not enough for all problems
- ![[Pasted image 20260927104932.png]]
- Polynomial gives better expressivity
- Polynomial expression can be described as 
	- not linear in x
	- linear in higher order terms of x
	- linear in terms of its parameters (weights).
- Example: $$y = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \dots + w_n x^n$$
	- Where:
		- $y$ is the predicted continuous target.
		- $x$ is the input feature.
		- $n$ is the degree of the polynomial (e.g., quadratic $n=2$, cubic $n=3$).
		- $w_0, w_1, \dots, w_n$ are the weights (coefficients to be learned).
		- $w_0$ is bias
- With too many parameters we can fit a function that has no significance in physical world i.e. overfitting
	- Here model might get fitted using parameters that should have no contribution to output

### Polynomial Model Regularization

- We use regularization to control overfitting in Polynomial Function or Model which occurs due to meaningless parameters present data
#### L1 , L2 Norms

- The **L1** and **L2** norms are mathematical functions used to measure the length or magnitude of a vector.
- The L1 norm is the sum of the absolute values of the vector's components:
$$L_1 = \sum_{i=1}^n \vert{}x_i\vert{} = \vert{}x_1\vert{} + \vert{}x_2\vert{} + \dots + \vert{}x_n\vert{}$$
- The L2 norm is the square root of the sum of the squared components:
$$L_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}$$

| **Feature**             | **L1 Norm (Manhattan / Taxicab Norm)**                                         | **L2 Norm (Euclidean Norm)**                                                |
| ----------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| **Geometric Intuition** | Distance along grid axes (like walking block-by-block in a city grid)          | Straight-line distance between two points ("as the crow flies")             |
| **Sparsity**            | Encourages **sparse solutions** (drives non-essential weights strictly to `0`) | Encourages **small, distributed weights** (spreads error across components) |
| **Computational Ease**  | Non-differentiable at `0`                                                      | Smoothly differentiable everywhere                                          |
| **Outlier Sensitivity** | Robust to outliers                                                             | Sensitive to outliers (large values get squared)                            |

#### Normalization using L1 and L2 Norms

- In machine learning and statistics, these norms are used as **regularization** penalties added to a loss function to prevent overfitting of model
##### 1. L1 Regularization (Lasso)
$$J(\mathbf{\theta}) = \text{MSE}(\mathbf{\theta}) + \alpha \sum_{i=1}^{n} \vert{}\theta_i\vert{}$$
(a) $\theta$ is the bias and coefficients of the polynomial model (i.e. weights)
(b) $\alpha$ is the regularization weight

- **Effect:** Performs automatic feature selection by zeroing out coefficients of uninformative features.
    
- **When to use:** High-dimensional data where you suspect only a small subset of features is actually relevant.

##### 2. L2 Regularization (Ridge)
$$J(\mathbf{\theta}) = \text{MSE}(\mathbf{\theta}) + \alpha \frac{1}{2} \sum_{i=1}^{n} \theta_i^2$$
(a) $\theta$ is the bias and coefficients of the polynomial model (i.e. weights)
(b) $\alpha$ is the regularization weight

- **Effect:** Shrinks feature weights toward zero, but rarely sets them exactly to zero.
    
- **When to use:** Dense data with highly correlated features (multicollinearity) where you want to keep all variables while stabilizing the model.
##### 3. Elastic Net

- Combines both L1 and L2 penalties to balance feature sparsity with the grouping effect of correlated features.

##### Understanding L1 and L2 penalty in 1-D space

![[Pasted image 20260929225241.png]]


#### How exactly Ridge or Lasso impacts weights mathematically

When you perform linear regression, the goal is to minimize the **Mean Squared Error (MSE)**, which measures how far the model's predictions are from the actual data points
$$\text{Loss}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

When you add Ridge or Lasso regularization, you modify this loss function by adding a **penalty term** directly proportional to the size of the weights $w$:

$$\text{Total Loss} = \text{MSE} + \text{Penalty}$$

Here is mathematically and intuitively how each penalty forces the weights to behave differently.

##### 1. Ridge Regression ($L_2$ Regularization): Shrinking Weights

In Ridge regression, the penalty is proportional to the **squared magnitude** of the weights:

$$\text{Total Loss} = \text{MSE} + \alpha \sum_{j} w_j^2$$

**How it impacts weight values:**

- **Gradient Descent Perspective:** To minimize the loss, gradient descent takes the derivative with respect to each weight $w_j$:
    $$\frac{\partial \text{Total Loss}}{\partial w_j} = \frac{\partial \text{MSE}}{\partial w_j} + 2\alpha w_j$$
    
    When updating the weight at each step with learning rate $\eta$:
    $$w_j^{\text{new}} = w_j - \eta \left( \frac{\partial \text{MSE}}{\partial w_j} + 2\alpha w_j \right) = \mathbf{(1 - 2\eta\alpha)w_j} - \eta \frac{\partial \text{MSE}}{\partial w_j}$$
    
- **Weight Decay:** Notice the $(1 - 2\eta\alpha)$ factor multiply $w_j$. Before making an update based on the data gradient ($\text{MSE}$), Ridge **scales down (decays)** the existing weight by a fraction.
    
- **Result:** Weights are continuously squeezed closer and closer to $0$, but because the force pulling them down shrinks as $w_j$ gets smaller (since $2\alpha w_j \to 0$), **weights rarely become exactly $0$**.
##### 2. Lasso Regression ($L_1$ Regularization): Zeroing Weights (Sparsity)

In Lasso regression, the penalty is proportional to the **absolute value** of the weights:
$$\text{Total Loss} = \text{MSE} + \alpha \sum_{j} \vert{}w_j\vert{}$$

**How it impacts weight values:**

- **Gradient Descent Perspective:** The derivative of $\vert{}w_j\vert{}$ with respect to $w_j$ is simply its sign, $+1$ (if $w_j > 0$) or $-1$ (if $w_j < 0$):
    $$\frac{\partial \text{Total Loss}}{\partial w_j} = \frac{\partial \text{MSE}}{\partial w_j} + \alpha \cdot \text{sign}(w_j)$$
    
    When updating the weight:
    $$w_j^{\text{new}} = w_j - \eta \frac{\partial \text{MSE}}{\partial w_j} - \mathbf{\eta\alpha \cdot \text{sign}(w_j)}$$
- **Constant Push:** Unlike Ridge, where the penalty diminishes as $w_j$ gets smaller, Lasso subtracts or adds a **constant value** ($\eta\alpha$) at every single step, regardless of how small $w_j$ is.
    
- **Result:** It continuously drives small weights straight through zero. Once a weight hits zero, the non-differentiable "corner" at $w_j = 0$ locks it there unless the MSE gradient becomes strong enough to pull it out. This acts as **automatic feature selection** by driving non-essential feature weights to **exactly $0$**.
##### Summary Comparison

| **Feature**                   | **Ridge (L2​)**                                   | **Lasso (L1​)**                                       |
| ----------------------------- | ------------------------------------------------- | ----------------------------------------------------- |
| **Penalty Term**              | $\alpha \cdot w_j^2$                              | $\alpha \cdot \vert{}w_j\vert{}$                      |
| **Gradient Contribution**     | $2\alpha \cdot w_j$ (proportional to weight size) | $\alpha \cdot \text{sign}(w_j)$ (constant magnitude)  |
| **Impact on Small Weights**   | Very weak push (slows down as weight nears $0$)   | Strong constant push (drives weights directly to $0$) |
| **Final Weight Distribution** | Many small, non-zero weights                      | Sparse weights (many are **exactly 0**)               |
| **Best Used For**             | Handling correlated/multicollinear features       | Feature selection when many features are irrelevant   |

#### Ridge vs Lasso Regression

- ![[Pasted image 20260930080022.png]]

- This image illustrates how **regularization ($\alpha$)** controls model complexity and prevents overfitting in high-degree polynomial regression using **Ridge Regression** vs. **Lasso Regression**.

##### Key Concepts Shown

- **Data Points (Blue Dots):** Represent noisy training observations.
    
- **$\alpha = 0$ (Solid Blue Line):** **No Regularization.** The model fits a high-degree polynomial that oscillates wildly to hit as many training points as possible, resulting in severe **overfitting**.
    
- **Small $\alpha$ (Green Dashed Line):** **Light Regularization.** Reduces the severe oscillations while retaining enough flexibility to capture the underlying non-linear trend.
    
- **$\alpha = 1$ (Red Dotted Line):** **Heavy Regularization.** Heavily restricts parameter sizes, smoothing out curves significantly.
##### Ridge vs. Lasso Behavior

| **Feature**                        | **Ridge Regression (L2​ Regularization) PNG**                                                                                         | **Lasso Regression (L1​ Regularization) PNG**                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Penalty Term**                   | Shrinks weights proportionally by adding the squared magnitude: $\alpha \sum w_i^2$.                                                  | Shrinks weights uniformly by adding the absolute magnitude: $\alpha \sum \lvert w_i \rvert$.                                             |
| **Effect on Weights**              | **Shrinks coefficients close to zero**, but rarely makes them exactly zero.                                                           | **Forces coefficients to exactly zero**, performing feature selection.                                                                   |
| **Visual Outcome at $\alpha = 1$** | Produces a slightly sloped line ($\approx \text{linear constraint}$) because high-degree features are penalized but still contribute. | Produces a **flat horizontal line** (y = intercept) because all polynomial feature weights ($w_1, w_2, \dots$) were set to exactly zero. |
##### Takeaway

- **Ridge ($L_2$)** is best when all features contribute to the target variable and you want to shrink large weights continuously.
    
- **Lasso ($L_1$)** is best when only a few features are relevant, as it naturally zeros out unnecessary parameters to create simpler, sparse models.

### Logistic Regression

- **Logistic Regression** is a supervised machine learning algorithm used primarily for **binary classification** (predicting two outcome classes, such as Yes/No, Spam/Not Spam, weather forecasting 0/1).
- Despite its name containing "regression," it is used for classification rather than predicting continuous numbers.
- It works around probability not exact class.
- Output ranges from 0 to 1

#### Key Concepts
- Logistic Regression starts by computing a linear combination of input features ($x_1, x_2, \dots, x_n$) using weights ($w$) and a bias ($w_0$):
  $$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + w_0 = \mathbf{w}^T \mathbf{x} + w_0$$
- Instead of outputting $z$ directly (which can be any real number from $-\infty$ to $+\infty$), it passes $z$ through a **logistic function** (also called the **sigmoid function**) to map the output to a probability score between $0$ and $1$.
- **Sigmoid function (Logistic Function):**
    $$\sigma(x) = \frac{e^{x}}{1 + e^{x}}$$
  OR:
  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
	- As $x \to \infty$, $\sigma(x) \to 1$
	- As $x \to -\infty$, $\sigma(x) \to 0$
	- When $x = 0$, $\sigma(x) = 0.5$
	- ![[Pasted image 20260930085300.png]]
	- Shape of Sigmoid function:
	  ![[Pasted image 20261001075047.png]]
- Decision Boundary
  To classify an observation into a category, a threshold is applied to the output probability $\hat{y} = \sigma(z)$:
  $$\hat{y}_{class} = \begin{cases} 1 & \text{if } \sigma(x) \ge 0.5 \\ 0 & \text{if } \sigma(x) < 0.5 \end{cases}$$


#### Preliminaries
- Logistic regression is inherently a 2-class classifier
- classes traditionally called 0 or 1

$$p(y=1|x)$$

- It represents probability that x belongs to class 1 
  where x is an input feature vector

##### Odds
- **Odds** measure how likely an event is to happen **compared to how likely it is not to happen**.

$$  
\text{Odds} = \frac{P(y=1|x)}{1-P(y=1|x)}  
$$

- **Properties of Odds**
	- Always Positive
	- Odds can never be negative. Lies between (0 and infinity)


#### Logistic Regression Model

- Instead of predicting a simple straight line, logistic regression works with **odds** (the probability of "Yes" divided by the probability of "No").

- **Rule:** The model assumes that taking the logarithm of these odds results in a straight-line equation:
$$\log\left(\frac{P(y = 1\vert{}\mathbf{x})}{1 - P(y = 1\vert{}\mathbf{x})}\right) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n = Z(say)$$
- **$x_1, x_2, \dots$** are your input features (e.g., age, income, test scores).
    
- **$\theta_0, \theta_1, \dots$** are the weights or parameters that control how important each feature is.

**Converting to a Probability (S-Curve)**

- Because a straight line can go from negative infinity to positive infinity, it cannot directly represent a probability (which must stay between **0** and **1** or **0%** and **100%**).
- To fix this, We Solve above equation for $P(y = 1\vert{}\mathbf{x})$ which yields the **sigmoid function** representation:
  $$P(y = 1\vert{}\mathbf{x}) = \frac{e^Z}{1 + e^Z}$$
- The probability ($P$) that the output is 1 ($y=1$), given our input inputs ($\mathbf{x}$) and weights ($\boldsymbol{\theta}$)
$$P(y = 1\vert{}\mathbf{x}; \boldsymbol{\theta}) = \frac{e^Z}{1 + e^Z}$$

#### What Kind of Boundary Logistic Regression Can  Model?

- Logistic regression creates a **straight-line decision boundary** (linear boundary) to separate data into two classes (Class 0 and Class 1).

**The Decision Threshold (The Split Point)**

- To decide whether a data point belongs to **Class 1** or **Class 0**, we set a cutoff probability, usually **0.5 (50%)**:
	- **Predict Class 1** if $P(y = 1\vert{}\mathbf{x}; \boldsymbol{\theta}) > 0.5$
	- **Predict Class 0** if $P(y = 1\vert{}\mathbf{x}; \boldsymbol{\theta}) < 0.5$
- ![[Pasted image 20261001085340.png]]

**How $Z$ Controls the Decision**

- The value of $Z$ ($\theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$) directly dictates which side of the boundary a point falls on:

	- **When $Z > 0$:** The probability is greater than 0.5, so the point is classified as **Class 1** (red dots on the right).
	
	- **When $Z < 0$:** The probability is less than 0.5, so the point is classified as **Class 0** (blue dots on the left).
	    
	- **When $Z = 0$:** The probability is exactly **0.5**, which forms the actual blue boundary line separating the two groups.

#### What about non-linear boundary

- What happens when a single straight line can’t cleanly separate your data?
- If $Z$ uses only basic terms ($x_1, x_2$), the boundary is a **straight line**.
- To draw a **curved boundary**, we change the formula for $Z$ to include squared or higher-power terms (e.g., quadratic).
- ![[Pasted image 20261001085643.png]]
### Optimize Weights
- Applicable to all ML models


