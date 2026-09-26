## Introduction to Machine Learning

>Data analysis, pre-processing and visualization, ML workflow, cross-validation, error types and performance metrics

### Machine Learning

Machine Learning (ML) is a way of building systems that **learn patterns from data** and use those patterns to make predictions or decisions.

Instead of explicitly programming every rule, we provide:

- Data
- Examples
- A learning algorithm/model

The model learns the relationship between input and output.

### Machine Learning - Overall Picture
![[Pasted image 20260926095802.png]]

```text
                          MACHINE LEARNING
	                               │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
     Supervised              Unsupervised            Reinforcement
       Learning                 Learning                Learning
          │                        │                        │
     ┌────┴────┐               ┌───┴────┐                 Agent
     │         │               │        │                   ↓
Classification Regression   Clustering Association          │
     │         │                                            │
     └────┬────┘                                            │
          │                                                 │
       Labels                                          Environment
          │                                                 │
       Model                                              Action
          │                                                 │
        Loss                                              Reward
          │
    Optimization
          │
    Gradient Descent
```

### Applications of Machine Learning

Some applications shown in the slides:

- Robotic manufacturing
- Agriculture
- Loan approval
- Smart diagnosis
- Depression classification
- Recommendation systems
- Time-series forecasting
- Computer vision

### Types of Data

Machine learning can work with different types of data.

#### Time-Series Data

Time-series data is data collected over time.

Example:

```text
Time →  Jan  Feb  Mar  Apr
Value → 100  120  115  140
```

A common application is **time-series forecasting**, where we use historical data to predict future values.

#### Image Data

Images can be represented as numerical data.

Examples:

- Grayscale images → 2-dimensional representation
	- For 8 bit representation pixel values ranges from 0 to 255 
	- 255 = 2^8 - 1
	- 0 = black
	- 255 = white
	- ![[Pasted image 20260922082222.png]]
- Color images → 3-dimensional representation
	- R G B

#### Video Data

- Video can be viewed as a sequence of images over time.
- Therefore, video introduces an additional **time dimension**.
- For grayscale video, Dimension is 3-D

### Why Machine Learning Did Not Take Off in the 1990s

Many important ML/AI techniques already existed before the 1990s.
However, two major limitations existed:
#### Limited Big Data

There was not enough large-scale data available for ML models to learn effectively.

#### Limited Computational Power

- Computers did not have enough computational power to process very large datasets efficiently.

### Why Machine Learning Is Trending Now

Two important changes made modern ML practical:
#### Big Data Availability

Huge amounts of data are now generated every day.
- Google : One trillion images
- Facebook : 350 million images uploaded per day
- Youtube : 100 hours of video uploaded per minute
- Walmart : 2.5 petabytes of data every minute

#### Increased Computational Power

Modern systems use **parallel processing**, multicore CPUs, especially GPUs which has thousands of core, to process large amounts of data efficiently.

So, modern ML benefits from:

```text
More Data
   +
More Computing Power
   ↓
Practical Machine Learning
```

### Case Study: Reading Postcodes

A major example is using AI to automatically read handwritten postcodes.
![[Pasted image 20260922083756.png]]

The basic idea is:

```text
Images + Labels
       ↓
Neural Network
       ↓
Training
       ↓
Trained Model
       ↓
New Image
       ↓
Prediction
```
![[Pasted image 20260922084710.png]]

#### Image Representation

- Computers do not directly understand an image as a human does.
- An image is represented using numerical values.
- For example, a grayscale image can be represented using pixel intensity values.
- The postcode example uses images that are:
	- 28 × 28 pixels
	- Grayscale
	- 60,000 training examples
	- 10,000 test examples
	- Output is an integer from 0–9
- ![[Pasted image 20260922084001.png]]

#### Invariance

A good image-recognition system should ideally recognize the same object even when its appearance changes.
![[Pasted image 20260922084211.png]]

**Translation Invariance**
The object can move to a different position in the image.

**Rotation Invariance**
The object can be slightly rotated.

**Scale Invariance**
The object can appear larger or smaller.

**Squeeze Invariance**
The object can become compressed or stretched.

**Stroke-Width Invariance**
The thickness of a handwritten stroke can change.

**Noise Invariance**
The image may contain noise but the model should still recognize the object.

### Supervised Learning

- Supervised learning is machine learning where the model learns from **labelled data**.
- Each training example contains:

```text
Input → Correct Output
```

Example:

```text
Image of digit 7 → Label: 7
Image of digit 3 → Label: 3
Image of digit 9 → Label: 9
```

The model learns the relationship between the input and the expected output.

### How to Solve a Supervised Learning Problem

#### 1. Collect Labelled Dataset
- Collect examples where both input and correct output are known.
- We can get datasets from **Kaggle**, Domain experts or self create dataset

```text
Input + Label
Input + Label
Input + Label
...
```

#### 2. Design Machine Learning Model

Choose a model that can learn the relationship between input and output.
For the postcode example, an **Artificial Neural Network (ANN)** is used.

#### 3. Define Loss Function
A loss function measures how different the model’s prediction is from the correct answer.

```text
Prediction vs Actual Answer
          ↓
       Loss Value
```

Lower loss generally means the prediction is closer to the desired output.

#### 4. Optimize Weights

- The model contains parameters called **weights**.
- Training adjusts these weights so that the loss becomes smaller.
- Overall:
```text
Dataset
   ↓
Model
   ↓
Prediction
   ↓
Loss
   ↓
Update Weights
   ↓
Better Model
```

### Artificial Neural Network

An Artificial Neural Network (ANN) is a machine learning model inspired by the way biological neural networks process information.

An ANN consists of interconnected computational units arranged into layers.

**Input Layer**
- Receives the input data.
- For an image, the input can be pixel values.

**Hidden Layers**
- Transform the input into increasingly useful representations.

**Output Layer**
- Produces the final prediction.
- For digit classification:

```text
Input Image
    ↓
Hidden Layers
    ↓
Output
    ↓
0 - 9
```

The postcode example uses multiple layers to transform the input image into a digit prediction.

![[Pasted image 20260923081448.png]]

### Classification

Classification means predicting a **category/class**.

Example:

```text
Input: Image
Output: 0, 1, 2, ..., 9
```

The output belongs to one of a fixed set of categories.
![[Pasted image 20260923081615.png]]
### Regression

Regression means predicting a **continuous numerical value**.

Example:

```text
Input → House features
Output → House price
```

So:

```text
Classification → Category
Regression     → Continuous value
```

The slides explicitly contrast classification and regression.
![[Pasted image 20260923081654.png]]
### Loss Functions

A loss function tells us **how wrong the model is**.

General idea:

```text
Actual Output
      ↓
Compare
      ↑
Predicted Output
      ↓
     Loss
```

During training, the objective is generally to minimize the loss.

#### Cross-Entropy Loss

Cross-entropy loss is commonly associated with **classification problems**.

It measures the difference between the true class distribution and the probability distribution predicted by the model.

Simple intuition:

```text
Correct and confident prediction
        ↓
     Low Loss

Wrong and confident prediction
        ↓
     High Loss
```

The slides introduce cross-entropy loss in the classification section.
$$  
LCE = \sum_{i=1}^{n}t_ilog(p_i) ~~~~~for~n~classes  
$$
where _ti_ is the true label and _pi_ is the Softmax probability for i class
Example:
- ![[Pasted image 20260923082653.png]]
- ![[Pasted image 20260923082752.png]]
#### Mean Squared Error (MSE)

MSE is a loss function used for measuring the squared difference between actual and predicted values.

Formula:
$$  
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2  
$$
Where:
- $y_i$ = actual value
- $\hat{y}_i$ = predicted value
- $n$ = number of observations

![[Pasted image 20260923083711.png]]

Example:

```text
Actual:    10
Predicted: 8

Error = 10 - 8 = 2
Squared Error = 4
```

- MSE can be used in the regression context.

### Optimizing an ANN

- After defining a loss function, we need to find **weights that minimize the loss**.
- Two concepts are: Closed-Form Optimization & Gradient Descent
#### Closed-Form Optimization

For some mathematical models, an optimal solution can be calculated directly using a closed-form expression.
- Loss function is a function of parameters y= f(x)
- we need to find parameters for which loss is minimum
  i.e. we need to find local minimum
  ![[Pasted image 20260924085137.png]]
  

#### Gradient Descent
- For many ML models, especially complex neural networks, we use iterative optimization.
- Gradient descent is an optimization technique used to reduce the loss.
- It iteratively adjusts parameters (weights and biases) in the direction of the steepest decrease of the function to find the optimal solution.
- The basic update idea is:
$$  
New\ Weight = Old\ Weight - Learning\ Rate \times Gradient  
$$
- The gradient indicates the direction in which the loss changes.
- The learning rate controls **how large each weight update is**.
  A suitable learning rate is important for effective optimization.
- Basic process:
	- **Initialize Parameters:** Start with random values for the parameters (weights $w$).
	
	- **Calculate the Gradient:** Compute the partial derivatives of the loss function $L(w)$ with respect to each parameter. The gradient vector $\nabla L(w)$ points in the direction of the steepest _increase_.
    
	- **Update Parameters:** Step in the _opposite_ direction of the gradient to reduce the loss:$$w_{new} = w_{old} - \alpha \cdot \nabla L(w)$$
	    - $\alpha$ is the **learning rate** (step size).
        
	- **Iterate:** Repeat until the gradient approaches zero or the change in loss is negligible (convergence).

##### Example:

**Problem Setup**

- **Model:** $\hat{y} = w \cdot x$ (predicting $y$ using a weight $w$, setting bias to $0$ for simplicity)
- **Training Data:** A single point $(x, y) = (2, 5)$
- **Loss Function (Mean Squared Error):**
    $$L(w) = (\hat{y} - y)^2 = (2w - 5)^2$$
- **Learning Rate ($\alpha$):** $0.1$
- **Initial Weight ($w_0$):** $0.0$

**Goal:** Find $w$ such that the loss $L(w)$ is minimized (the ideal weight is $2.5$ because $2.5 \times 2 = 5$).

**Step 1: Calculate the Gradient Formula**

The gradient is the derivative of the loss function with respect to $w$:

$$\frac{dL}{dw} = 2(2w - 5) \cdot 2 = 4(2w - 5) = 8w - 20$$

**Step 2: Iteration 1**

1. **Current prediction:**
    $$\hat{y}_0 = 0.0 \times 2 = 0$$
    
2. **Current loss:**
    $$L(w_0) = (0 - 5)^2 = 25$$
    
3. **Calculate gradient at $w_0 = 0$:**
    $$\nabla L(0) = 8(0) - 20 = -20$$
    
4. **Update the weight:**
    $$w_1 = w_0 - \alpha \cdot \nabla L(w_0)$$
    $$w_1 = 0 - 0.1 \times (-20) = 2.0$$
**Step 3: Iteration 2**

1. **New prediction:**
    $$\hat{y}_1 = 2.0 \times 2 = 4$$
2. **New loss:**
    $$L(w_1) = (4 - 5)^2 = 1$$
    _(Loss dropped significantly from 25 to 1!)_
    
3. **Calculate gradient at $w_1 = 2.0$:**
    $$\nabla L(2.0) = 8(2.0) - 20 = -4$$
    
4. **Update the weight:**
    $$w_2 = w_1 - \alpha \cdot \nabla L(w_1)$$
$$w_2 = 2.0 - 0.1 \times (-4) = 2.4$$
**Convergence Summary**

|**Iteration (k)**|**Weight (wk​)**|**Prediction (y^​)**|**Loss (L)**|**Gradient (∇L)**|
|---|---|---|---|---|
|**0**|$0.00$|$0.00$|$25.000$|$-20.00$|
|**1**|$2.00$|$4.00$|$1.000$|$-4.00$|
|**2**|$2.40$|$4.80$|$0.040$|$-0.80$|
|**3**|$2.48$|$4.96$|$0.0016$|$-0.16$|

As iterations continue, $w$ smoothly approaches $2.5$, the gradient approaches $0$, and the loss reaches minimum.



### Parametric vs Non-Parametric Methods

#### Parametric Methods

- A parametric model assumes a particular form and uses a fixed number of parameters.
- To minimize loss these parameters are changed
- Most of the models uses parametric method these days.

Example:
```text
y = ax + b
```
Here, the model has parameters:
- `a`
- `b`

#### Non-Parametric Methods

- Non-parametric methods do not assume a fixed mathematical form with a fixed number of parameters.
- The complexity can depend more directly on the data.
- Less likely these days.

Simple intuition:

```text
Parametric
→ Fixed model structure/parameters

Non-parametric
→ More flexible relationship with data
```

### Data Challenges

Real-world data is rarely perfect.
Several important data challenges:
1. Missing data
2. Errors and artifacts
3. Outliers
4. Curse of dimensionality

#### Missing Data

- Missing data occurs when some expected values are unavailable.
- Example:
```text
Age   Salary
25    50000
30    ?
28    60000
```

- Missing data can introduce errors and cause problems during analysis or training.
- Imputation means filling missing values using an appropriate method.
- Imputation methods : mean, median for numeric data, maximum repeating thing for other type

#### Errors vs Artifacts

Real-world data can contain:
- Raw data
- Processing effects
- Errors
- Artifacts

An **error** represents incorrect information. It can't be recovered.
eg: Blur in an image caused by a patient moving during the X-ray scan
eg: Measurement numbers from a defected machine

An **artifact** is an unwanted pattern introduced by the measurement, collection, or processing process.
#### Outliers

An outlier is a data object that is significantly different from normal observations.

Example:

```text
Normal credit card purchases:
₹500
₹700
₹1,200
₹900

Unusual purchase:
₹5,00,000
```

The unusual transaction may be an outlier.

We should try to remove outliers so that model accuracy does not get impacted.
![[Pasted image 20260925083527.png]]

We can use methods like IQR (Interquartile Range) for this.
![[Pasted image 20260925083249.png]]

**Outlier vs Error**

An outlier is **not necessarily an error**.

An outlier can contain useful information.

For example, an unusual credit card purchase could indicate:

- Fraud
- A genuine expensive purchase
- A different type of customer behavior

#### Curse of Dimensionality

The curse of dimensionality occurs when the number of features/dimensions becomes very large.

As dimensionality increases:
- More data may be required.
- Models can become more complex.

Simple intuition:

```text
More Features
     ↓
More possible combinations
     ↓
More data required
     ↓
More complex model
```

We should try our data has only required features.
### Model Challenges

Apart from data challenges, ML models themselves have challenges.
1. Model complexity
2. Inductive bias
3. Evaluation

#### Model Complexity

- A model that is too simple may fail to capture the underlying relationship.
- A model that is too complex may fit the training data excessively.
- We can explain model complexity using polynomial models of increasing order.
- ![[Pasted image 20260926090722.png]]

**Overfitting**

Overfitting occurs when a model learns the training data too closely and does not generalize well to unseen data.

Conceptually:

```text
Training Data
     ↓
Model learns
     ↓
Very good training performance

But...

New Data
     ↓
Poor performance
```

The goal is not simply to memorize training data.
The goal is to **generalize to new data**.

**Training vs Validation**

To detect overfitting, data can be separated into training and validation sets.

**Training Set**
Used to train the model.

```text
Training Data
     ↓
Learn Model Parameters
```

**Validation Set**
Used to evaluate the model during model development.

```text
Validation Data
     ↓
Check Generalization
```

The slides specifically show training and validation data being used to study overfitting.

**K-Fold Cross Validation**

K-fold cross-validation divides the dataset into `K` parts, called folds.

Process:

```text
Fold 1
Fold 2
Fold 3
...
Fold K
```

For each iteration:

- Train using `K-1` folds.
- Use the remaining fold for validation.
- Repeat until every fold has been used as the validation fold.

Example with 5 folds:

```text
Round 1 → Train: 2,3,4,5 | Validate: 1
Round 2 → Train: 1,3,4,5 | Validate: 2
Round 3 → Train: 1,2,4,5 | Validate: 3
Round 4 → Train: 1,2,3,5 | Validate: 4
Round 5 → Train: 1,2,3,4 | Validate: 5
```

![[Pasted image 20260926090853.png]]
#### Inductive Bias

- **Inductive bias** is the set of assumptions a machine learning algorithm uses to predict outputs for unseen data based on its training inputs.
- Because training data cannot contain every possible example, the model needs assumptions to generalize from observed examples to unseen examples.

#### Model Evaluation

- After training a model, we need to determine how well it performs.
- Evaluation is especially important because a model can perform well on training data but poorly on unseen data.
#### Classification Errors

For binary classification, predictions can be divided into positive and negative outcomes.

**True Positive**

Model predicts positive and the actual result is positive.

```text
Predicted: Positive
Actual:    Positive
```

**True Negative**

Model predicts negative and the actual result is negative.

```text
Predicted: Negative
Actual:    Negative
```

**False Positive**

Model predicts positive but the actual result is negative.

```text
Predicted: Positive
Actual:    Negative
```

**False Negative**

Model predicts negative but the actual result is positive.

```text
Predicted: Negative
Actual:    Positive
```

These error types are important because different applications( weather prediction, cancer diagnosis, and anti-missile systems ) have different consequences for false positives and false negatives.

### Performance Metrics

Classification models can be evaluated using performance metrics derived from their predictions.

Important concepts include:

- True Positive
- True Negative
- False Positive
- False Negative

These form the basis for commonly used classification metrics.
![[Pasted image 20260923084751.png]]

### Unsupervised Learning

- Unsupervised learning works with data **without predefined labels**.
- Instead of:
```text
Input → Known Label
```
- we have:
```text
Input Data
   ↓
Algorithm
   ↓
Discover Structure
```
- ![[Pasted image 20260926100200.png]]
#### Clustering

- Clustering groups similar data points together.
- One of the major unsupervised learning approaches.
- Example:
```text
Data
 ↓
┌─────────┐
│ Group A │
│ similar │
└─────────┘

┌─────────┐
│ Group B │
│ similar │
└─────────┘
```

- ![[Pasted image 20260926100053.png]]
- There are no predefined labels telling the algorithm which group each point belongs to.
- The algorithm attempts to discover groups based on similarity.

#### Association

- Association learning looks for relationships between items or events.
- Association is another unsupervised learning approach
- Example:
```text
Customer buys Keyboard
        +
Customer buys Mouse
        ↓
Possible association between Keyboard and Mouse
```

- A common use case is discovering relationships in transaction data.

### Reinforcement Learning

Reinforcement Learning is a learning approach where an **agent interacts with an environment** and learns from the results of its actions.

![[Pasted image 20260926095841.png]]

Basic structure:

```text
        Action
Agent ───────────→ Environment
  ↑                    │
  │                    │
  └── Reward/Feedback ─┘
```

The agent learns which actions are useful based on feedback from the environment.

#### Reinforcement Learning Objective

Move forward while minimizing how often it fails and the energy it expends.

So the agent needs to balance:

- Achieving its objective
- Avoiding failures
- Reducing energy expenditure

#### Learning Phase

During the learning phase, the agent interacts with the environment and learns from its experiences.

Conceptually:

```text
Observe Environment
        ↓
Choose Action
        ↓
Receive Feedback
        ↓
Update Learning
        ↓
Choose Better Action
```


### Key Takeaways

1. **Machine Learning learns patterns from data.**
2. Modern ML became practical because of **large datasets and increased computational power**.
3. **Supervised learning** uses labelled data.
4. **Classification** predicts categories.
5. **Regression** predicts continuous values.
6. A **loss function** measures prediction error.
7. **Gradient descent** can be used to optimize model parameters.
8. Real-world data can contain **missing values, errors, artifacts, and outliers**.
9. High dimensionality can lead to the **curse of dimensionality**.
10. A model that is too complex can **overfit** the training data.
11. **Validation data and K-fold cross-validation** help evaluate generalization.
12. **Inductive bias** represents assumptions made by a learning algorithm.
13. **Unsupervised learning** discovers structure without predefined labels.
14. **Clustering** groups similar data.
15. **Association** discovers relationships between items.
16. **Reinforcement learning** learns through interaction with an environment and feedback.
17. The overall ML workflow can be remembered as:

```text
Data
 ↓
Model
 ↓
Prediction
 ↓
Loss / Feedback
 ↓
Optimization / Learning
 ↓
Evaluation
 ↓
Generalization
```