> Boosting : Adaptive and Gradient boosting

This is about supervised learning and continuation of decision tree.

### 1. What is Boosting?

Boosting is an ensemble learning technique that combines multiple weak learners to create a stronger model.

- A weak learner is a model that performs slightly better than random guessing.
- Boosting trains models sequentially, one after another.
- Each new model focuses more on the mistakes made by previous models.
- The final prediction combines the predictions of all the models.

**Intuition**

Imagine a student preparing for an exam:

1. The student takes a test.
2. Finds the questions they answered incorrectly.
3. Focuses more on those questions.
4. Takes another test.
5. Repeats this process.

Boosting works similarly. Each new model tries to improve the mistakes made by earlier models.

---

### 2. Business Problem — Customer Churn Prediction

Customer churn means a customer stops using a company’s service.

The business wants to predict:

Which customers are likely to leave next month so that the retention team can contact them before they leave?

**Dataset**

We are using IBM Telco Customer Churn dataset.

- 7,043 customers
- 21 columns
- Customer information such as:
    - Contract type
    - Tenure
    - Internet services
    - Monthly charges
    - Total charges
    - Payment method
- Approximately 26.5% customers churned.

**Why prediction is useful**

Instead of offering discounts to everyone, the company can identify high-risk customers and target them with retention offers.

---

### 3. Data Exploration
Before training a model, understand the dataset.

For example, analyze churn rate for different contract types.

|**Contract Type**|**Churn Rate**|
|---|---|
|Month-to-month|42.7%|
|One-year|11.3%|
|Two-year|2.8%|

This indicates that customers with month-to-month contracts are more likely to leave.

We can use pandas to analyze the relationship:

```python
df.groupby("Contract")["Churn"].value_counts(normalize=True)
```

---

### 4. Data Preprocessing

Before training a machine learning model, data must be prepared.

#### 4.1 Convert numeric-looking text into numbers

`TotalCharges` appears numeric but is stored as text.

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

`errors="coerce"` converts invalid values into `NaN`.

This reveals 11 missing values in the `TotalCharges` column.

#### 4.2 Separate features and target

```python
X = df.drop(columns=["customerID", "Churn"])

y = (df["Churn"] == "Yes").astype(int)
```

- `X` → input features
- `y` → target variable
- `customerID` → removed because it is an identifier, not a useful predictive feature
- `Churn` → converted into binary values

|**Churn**|**Encoded value**|
|---|---|
|No|0|
|Yes|1|

---

### 5. Handling Missing Values

Missing values can be handled differently depending on the type of column.

**Numerical columns**

Use the median.

```python
SimpleImputer(strategy="median")
```

**Categorical columns**

Use the most frequently occurring value.

```python
SimpleImputer(strategy="most_frequent")
```

**Implementation**

```python
num_cols = X.select_dtypes("number").columns
cat_cols = X.columns.difference(num_cols)

fill_num = SimpleImputer(strategy="median")
fill_cat = SimpleImputer(strategy="most_frequent")

X[num_cols] = fill_num.fit_transform(X[num_cols])
X[cat_cols] = fill_cat.fit_transform(X[cat_cols])
```

**Why median?**

The median is less affected by extreme values than the mean.

For example:

`10, 20, 30, 40, 1000`

- Mean = 220
- Median = 30

The extreme value `1000` significantly changes the mean but does not affect the median as much.

---

### 6. Encoding Categorical Data

Machine learning algorithms generally need numerical input.

Categorical data contains text such as:

- Month-to-month
- One year
- Two year

We can convert categories into numerical values using `OrdinalEncoder`.

```python
X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
```

Example:

|**Contract**|**Encoded Value**|
|---|---|
|Month-to-month|0|
|One year|1|
|Two year|2|

Trees can split based on these numerical codes.

For example:

```text
Contract <= 0.5
```

This separates month-to-month customers from other contract types.

In tree-based models, the assigned numerical order does not necessarily represent a meaningful ranking of categories; it simply provides codes that the model can split on.

---

### 7. Train-Test Split

We divide the dataset into training and testing data.

```python
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

- Training data → used to train the model.
- Testing data → used to evaluate the model on unseen examples.

**Dataset split**

|**Dataset**|**Samples**|
|---|---|
|Training|5,634|
|Testing|1,409|
|Total|7,043|

**What is stratify?**

`stratify=y` ensures that the proportion of churn and non-churn customers remains approximately the same in both training and testing datasets.

This is important when the target classes are imbalanced.

---

### 8. Baseline Model — Decision Tree

Before using boosting, train a single decision tree.

```python
tree = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

tree.fit(X_tr, y_tr)
```

This model provides a baseline performance that we can compare against boosted models.

**Why establish a baseline?**

A more complex model is useful only if it improves performance over a simpler model.

---

### 9. AdaBoost

AdaBoost stands for Adaptive Boosting.

It trains multiple weak learners sequentially and adjusts the importance of training samples based on previous mistakes.

Here the weak learners are decision tree stumps.

**What is a decision stump?**

A decision stump is a decision tree with only one split.
It asks a single question.
Example:

```text
Is the customer on a month-to-month contract?
```

It makes a simple prediction based on that question.

**AdaBoost implementation**

```python
ada_stumps = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

ada_stumps.fit(X_tr, y_tr)
```

**Parameters**

|**Parameter**|**Meaning**|
|---|---|
|`estimator`|Weak learner used in each boosting round|
|`max_depth=1`|Each tree is a stump with one split|
|`n_estimators=100`|Maximum number of weak learners|
|`learning_rate=0.5`|Controls contribution of each learner|
|`random_state`|Makes results reproducible|

---

#### 9.1 How AdaBoost Works

AdaBoost follows these steps:

**Step 1 — Start with equal weights**

- Initially, every training sample has equal importance.

```text
Customer 1 → Equal weight
Customer 2 → Equal weight
Customer 3 → Equal weight
Customer 4 → Equal weight
```

**Step 2 — Train the first weak learner**

- The first decision stump is trained on the data.
- For example, it may predict that everyone will stay.
- It correctly predicts most customers but misses churners.

**Step 3 — Increase importance of incorrectly classified samples**

- Customers that were incorrectly classified receive greater importance.
- The next stump focuses more on these difficult examples.

**Step 4 — Train the next stump**

- The second stump learns from the updated sample weights.

**Step 5 — Repeat**

- This process continues for multiple rounds.

**Step 6 — Combine predictions**

- All weak learners contribute to the final prediction through a weighted vote.
- More accurate learners receive greater influence.

##### Execution Flow

```text
Training Data
     ↓
Train Weak Learner 1
     ↓
Identify Mistakes
     ↓
Increase Weight of Misclassified Samples
     ↓
Train Weak Learner 2
     ↓
Identify Mistakes Again
     ↓
Repeat
     ↓
Combine Predictions
     ↓
Final Model
```

---

#### 9.2 AdaBoost vs Single Decision Tree

|**Model**|**Accuracy**|
|---|---|
|Always predict stays|73.5%|
|Single Decision Tree|77.6%|
|AdaBoost|79.8%|

AdaBoost improved accuracy by approximately 2.2 percentage points over the single decision tree.

**Churners detected**

|**Model**|**Churners Caught**|
|---|---|
|Single Decision Tree|31%|
|AdaBoost|54%|

The boosted model identified nearly twice as many churners as the single tree in this run.

---

#### 9.3 AdaBoost vs Random Forest

Both Random Forest and Boosting combine multiple decision trees, but they work differently.

|**Random Forest**|**Boosting**|
|---|---|
|Trees are trained independently|Trees are trained sequentially|
|Trees can be trained in parallel|Later trees depend on earlier trees|
|Usually uses deeper trees|Often uses smaller trees|
|Combines predictions using averaging or voting|Combines predictions using weighted contributions|
|Reduces variance|Focuses on correcting previous errors|
|Less sensitive to noisy labels|Can focus too much on noisy or mislabeled samples|

**Analogy**

Random Forest = Panel of experts
Each expert independently gives an opinion, and the results are combined.

Boosting = Relay team
Each new member builds upon the work and mistakes of the previous member.

---

### 10. Gradient Boosting

- Gradient Boosting also builds models sequentially.
- However, instead of simply increasing weights for misclassified samples as in the AdaBoost explanation, Gradient Boosting trains each new model to correct the remaining errors of the current ensemble.

**Basic idea**

```text
Initial Prediction
        ↓
Calculate Errors
        ↓
Train Tree to Correct Errors
        ↓
Update Prediction
        ↓
Calculate New Errors
        ↓
Train Next Tree
        ↓
Repeat
```

Each new tree improves the overall model gradually.

---

### 11. Industry Boosting Libraries

##### HistGradientBoosting

A gradient boosting implementation in scikit-learn.

```python
hgb = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=1000,
    max_leaf_nodes=15,
    early_stopping=True,
    n_iter_no_change=50,
    categorical_features="from_dtype",
    random_state=42
)
```

Key points:

- Uses histogram-based learning.
- Supports categorical features.
- Can handle missing values.
- Supports early stopping.
- Works well when staying within the scikit-learn ecosystem.

---

##### XGBoost

- XGBoost stands for Extreme Gradient Boosting.
- It is widely used for structured/tabular datasets.

```python
xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,
    tree_method="hist",
    early_stopping_rounds=50,
    eval_metric="auc",
    random_state=42
)
```

Key points:

- Efficient implementation of gradient boosting.
- Supports regularization.
- Supports row and column sampling.
- Supports early stopping.
- Widely used in industry and competitions.

---

##### LightGBM

- LightGBM is a gradient boosting framework designed for speed and efficiency.

```python
lgbm = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=15,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
```

Key points:

- Designed for fast training.
- Efficient on large datasets.
- Uses leaf-based tree growth.
- Supports categorical features.
- Supports early stopping through callbacks.

---

#### 11.1 Comparison of Boosting Libraries

|**Setting**|**HistGradientBoosting**|**XGBoost**|**LightGBM**|
|---|---|---|---|
|Number of trees|`max_iter`|`n_estimators`|`n_estimators`|
|Learning rate|`learning_rate`|`learning_rate`|`learning_rate`|
|Tree size|`max_leaf_nodes`|`max_depth`|`num_leaves`|
|Row sampling|—|`subsample`|`subsample` + `subsample_freq`|
|Column sampling|`max_features`|`colsample_bytree`|`colsample_bytree`|
|Minimum cases per leaf|`min_samples_leaf`|`min_child_weight`|`min_child_samples`|
|Early stopping|`early_stopping=True`|`early_stopping_rounds`|`early_stopping()` callback|

---

#### 11.2 Model Comparison — Churn Dataset**

Example result:

|**Model**|**AUC**|**Trees Used**|
|---|---|---|
|XGBoost|0.845|143|
|HistGradientBoosting|0.844|126|
|LightGBM|0.843|117|
|AdaBoost|0.840|100|
- Here AUC means Area under curve. 

**Conclusion**

All three gradient boosting implementations produced almost identical AUC scores in this experiment.

The best model is not always the one with the highest score. Other considerations include:

- Training speed
- Dataset size
- Deployment environment
- Hardware requirements
- Ease of maintenance

---

#### 11.3 Important Boosting Hyperparameters

**17.1 Learning Rate**

Controls how much each new tree contributes to the model.

```python
learning_rate=0.05
```

- Smaller learning rate → slower, more gradual learning.
- Larger learning rate → faster updates but may overshoot optimal solutions.

A smaller learning rate generally requires more trees.

**17.2 Number of Trees**

```python
n_estimators=1000
```

Maximum number of trees to train.

Rather than manually guessing the number, set a sufficiently high limit and use early stopping.

**17.3 Tree Size**

Controls the complexity of each tree.

Examples:

- `max_depth`
- `max_leaf_nodes`
- `num_leaves`

Starting point:

- Depth: 3–6
- Leaves: 15–31

Smaller trees help prevent overfitting.

**17.4 Sampling**

Sampling allows each tree to use only a subset of rows and/or columns.

```python
subsample=0.8
colsample_bytree=0.8
```

This means approximately:

- 80% of training rows
- 80% of available features

Sampling introduces variation and can help reduce overfitting.

---

#### 11.4 Early Stopping

Early stopping prevents the model from training unnecessarily after performance stops improving.

**How it works**

1. Train the model on training data.
2. Evaluate performance on validation data.
3. Track performance after each boosting round.
4. Stop when performance fails to improve for a certain number of rounds.

Example:

```python
early_stopping_rounds=50
```

This means training can stop if the validation score does not improve for 50 consecutive rounds.

**Why is it useful?**

- Prevents unnecessary training.
- Helps reduce overfitting.
- Automatically determines an appropriate number of trees.

**Data split**

```text
Training Data
     ↓
Training Subset + Validation Subset
     ↓
Train Model
     ↓
Monitor Validation Performance
     ↓
Early Stopping
     ↓
Evaluate on Test Data
```

The test set should remain untouched during model tuning.

---

### 12. Choosing the Correct Loss Function

The business problem determines which loss function to use.

|**Business Question**|**Example**|**Setting**|**Purpose**|
|---|---|---|---|
|How much or how long?|Delivery time, revenue|`squared_error`|Predicts average|
|How much with extreme outliers?|Insurance claim cost|`absolute_error`|Reduces influence of extreme values|
|How much to plan for?|Stock planning|`quantile`, `quantile=0.9`|Predicts a high percentile|
|Yes or no?|Fraud, churn, default|Classifier with log-loss|Predicts probabilities|
|Which item first?|Search, recommendation ranking|XGBRanker, LGBMRanker|Optimizes ranking|

The objective function should match the actual business goal.

---

### 13. Probability vs Decision

A model may predict:

```text
Customer A → 95% churn probability
Customer B → 70% churn probability
Customer C → 15% churn probability
```

- But a probability is not automatically a decision.
- The business must decide:
	- At what probability should we contact the customer?
	- This is called choosing a decision threshold or cutoff.
- The default cutoff for binary classification is often `0.5`, but it may not be optimal.

---

#### 20.1 Cost-Based Decision Threshold

In churn prediction, two types of mistakes are possible.

**False Negative — Missed Churner**
- The customer actually leaves, but the model does not flag them.
- Cost example:
```text
Cost = 500
```

**False Positive — Wasted Offer**
- The customer would have stayed anyway, but the company gives them a discount.
- Cost example:
```text
Cost = 100
```

Since missing a churner costs more than a wasted offer, we may want to contact customers at a lower probability threshold.

---

#### 20.2 Calculate Total Business Cost

Define the costs:

```python
COST_MISSED_CHURNER = 500
COST_WASTED_OFFER = 100
```

Function:

```python
def total_cost(y_true, prob, cutoff):
    flag = prob >= cutoff

    missed = ((~flag) & (y_true == 1)).sum()
    wasted = (flag & (y_true == 0)).sum()

    return (
        COST_MISSED_CHURNER * missed
        + COST_WASTED_OFFER * wasted
    )
```

**Explanation**

`flag = prob >= cutoff`
- Customers with probability greater than or equal to the cutoff are contacted.

`missed`
- Actual churners who were not contacted.

`wasted`
- Customers who were contacted but would not churn.

The total cost is:

```text
Total Cost =
(Missed Churners × Cost of Missing a Churner)
+
(Wasted Offers × Cost of a Wasted Offer)
```

---

#### 20.3 Finding the Best Cutoff

Instead of automatically using `0.5`, test different thresholds and choose the one with the lowest business cost.

Example results:

| **Cutoff** | **Total Cost** |
| ---------- | -------------- |
| 0.13       | 47,200         |
| 0.50       | 77,800         |

The optimal threshold in the example’s validation run was approximately `0.13`.

**Why a lower threshold?**

- Missing a churner is much more expensive than sending an unnecessary discount.
  Therefore, the model flags more customers.

---

### 14. Data Leakage

Data leakage occurs when the model uses information that would not be available at the time the prediction is made.

#### **Example**

Suppose we want to predict whether a customer will churn.

Our dataset contains:

```text
Contract
Tenure
Monthly Charges
Cancellation Reason
```

The cancellation reason is only recorded after the customer has already left.

If we include it in the training data, the model can learn from information that is unavailable when predicting future churn.

This leads to unrealistically high performance.

#### **Rule**

Never use information that becomes available only after the outcome you are predicting.

Examples of potential leakage:

- Cancellation reason when predicting churn.
- Final transaction status when predicting fraud.
- Post-diagnosis information when predicting disease.
- Data created after the target event.

---

### 29. Hyperparameter-Tuning**

- A **hyperparameter is a setting that we decide before training a machine learning model** to control how the model learns.
- Parameter vs Hyperparameter

|**Parameter**|**Hyperparameter**|
|---|---|
|Learned automatically during training|Set by us before training|
|Example: weights, coefficients|Example: number of trees, learning rate|
|Model figures out its value|We choose its value|
- For example, in a decision tree:
	- `max_depth = 3` → We tell the tree it can have a maximum of 3 levels.
	- The tree’s actual splitting rules → The model learns these from the data.
- Hyperparameters in boosting

|**Hyperparameter**|**What it controls**|**Simple meaning**|
|---|---|---|
|`n_estimators`|Number of models|How many learners to build|
|`learning_rate`|Contribution of each learner|How much each learner affects the final prediction|
|`max_depth`|Complexity of each tree|How deeply each tree can learn|
|`subsample`|Amount of data used per learner|Whether each learner uses all or part of the data|
|`min_samples_split`|Minimum samples needed to split|How easily a tree creates another branch|
|`min_samples_leaf`|Minimum samples in a leaf|Prevents leaves from becoming too small|

#### Tuning hyperparameters

- **Step 1**
	- Start with:
		- Learning rate = 0.05
		- High tree limit
		- Early stopping enabled
- **Step 2**
	- Adjust:
		- Tree size
		- Minimum cases per leaf
- **Step 3**
	- Add row and column sampling.
	- Start around:
```python
subsample=0.8
colsample_bytree=0.8
```

- **Step 4**
	- Try regularization settings.

- **Step 5**
	- For the final model, reduce the learning rate and retrain.

---

### 30. Common Boosting Mistakes

**Leakage**
Using features that are recorded after the prediction outcome.

**Time Leakage**
For time-dependent datasets, training on future data and testing on past data.
Correct approach:
```text
Past Data → Training
Future Data → Testing
```

**Imbalanced Classes**
When positive cases are rare, the model may struggle to identify them.
Possible approaches:

- `scale_pos_weight` in XGBoost
- Class weights
- AUC
- Precision-recall metrics

**Test Set Peeking**
Repeatedly tuning the model using test results makes the test set no longer independent.
Use:

```text
Training → Fit Model
Validation → Tune Model
Test → Final Evaluation
```

---

### 31. Applications of Boosting

|**Industry**|**Problem**|**Example**|
|---|---|---|
|Banking|Credit risk|Will an applicant default?|
|Payments|Fraud detection|Is a transaction fraudulent?|
|Telecom / SaaS|Churn prediction|Will the customer cancel?|
|Retail|Demand forecasting|How many units will sell next week?|
|E-commerce|Search ranking|Which products should appear first?|
|Manufacturing|Predictive maintenance|Will a machine fail soon?|

Boosting is especially useful for structured or tabular datasets.

---

### 32. Advantages of Boosting

- Often more accurate than a single model.
- Focuses on difficult examples.
- Works well on tabular business data.
- Supports classification and regression.
- Can identify useful features.
- Provides probability predictions that can be converted into business decisions.
- Modern libraries support missing values and categorical data.

---

### 33. Limitations of Boosting

- Training is sequential, so it can be slower than models trained independently.
- Can overfit if trees are too complex.
- Sensitive to noisy or mislabeled data.
- Requires tuning multiple hyperparameters.
- More difficult to interpret than a single decision tree.
- Deep learning may be more suitable for image, text, and audio data.

---

### 34. Which Boosting Algorithm Should We Use?

|**Algorithm**|**When to Use**|
|---|---|
|AdaBoost|Understanding boosting concepts or quick baseline on clean data|
|HistGradientBoosting|When using scikit-learn and wanting a simple integrated solution|
|XGBoost|Mature, widely supported solution with many deployment options|
|LightGBM|When training speed and large datasets are important|

The best algorithm depends on the dataset.

It is often useful to try two or more algorithms and compare their results using validation data.

---

### 35. End-to-End Boosting Workflow

```text
Define Business Problem
        ↓
Collect Dataset
        ↓
Explore Data
        ↓
Identify Target Variable
        ↓
Remove IDs and Leakage
        ↓
Handle Missing Values
        ↓
Encode Categorical Variables
        ↓
Split into Train / Validation / Test
        ↓
Train Baseline Model
        ↓
Train Boosting Model
        ↓
Tune Hyperparameters
        ↓
Use Early Stopping
        ↓
Evaluate on Test Data
        ↓
Choose Business Threshold
        ↓
Explain Predictions using SHAP
        ↓
Deploy and Monitor
```

---

### 36. Final Takeaways

1. Boosting combines multiple weak learners to create a stronger model.
2. Models are trained sequentially.
3. Each new learner focuses on improving the mistakes of previous learners.
4. AdaBoost adjusts the importance of incorrectly classified samples.
5. Gradient Boosting fits new trees to the remaining errors.
6. XGBoost, LightGBM, and HistGradientBoosting are popular implementations.
7. Learning rate, tree size, sampling, and early stopping are important hyperparameters.
8. A probability is not a decision — business costs determine the optimal cutoff.
9. SHAP helps explain why a model makes predictions.
10. Always avoid data leakage and keep the test set untouched during tuning.

---

#### Quick Revision

|**Concept**|**One-Line Explanation**|
|---|---|
|Boosting|Combines weak learners sequentially to create a strong learner|
|Weak Learner|A model that performs slightly better than random guessing|
|Decision Stump|A decision tree with one split|
|AdaBoost|Increases focus on misclassified samples|
|Gradient Boosting|Learns to correct the remaining errors|
|Learning Rate|Controls the contribution of each new tree|
|Early Stopping|Stops training when validation performance stops improving|
|XGBoost|Optimized gradient boosting framework|
|LightGBM|Fast gradient boosting framework|
|HistGradientBoosting|Histogram-based boosting in scikit-learn|
|AUC|Measures ranking ability of a classifier|
|Threshold|Probability cutoff used to make a classification decision|
|SHAP|Explains how features influence predictions|
|Data Leakage|Using information unavailable at prediction time|
