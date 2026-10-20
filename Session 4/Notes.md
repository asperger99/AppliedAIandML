> Ensemble methods: Decision Tree, Bagging and random forests

### Features for Classification

- A dataset can contain many features, but only some are useful for predicting the target.
- A discriminative feature helps distinguish target classes.
- A non-discriminative feature provides little help because its values overlap across classes.
- Example: the number of legs may not distinguish cats from dogs if both classes have the same value.
- ![[Pasted image 20261013081114.png]]

#### Discriminative Features

- A feature is useful and Discriminative when different values correspond to meaningfully different target-label distributions.
- Example: company may help predict a camera rating if ratings for different companies differ.
- A feature can appear useful on a very small group but still be unreliable due to limited examples.

### Feature Selection

- Feature selection identifies which input features are most useful for predicting the target.
- Example camera features: company, colour, resolution, video rate, and price.
- Target: the user's rating from 1 to 5 stars.
- A user may consider some product features more important than others.

### Decision Trees

- Idea: identify the “most discriminative” feature, use it to classify!
- A decision tree predicts a target through a sequence of feature-based decisions.
- **Root**: the first decision.
- **Internal node**: a feature test.
- **Branch**: an outcome of a test.
- **Leaf**: the final prediction.
- The algorithm chooses a feature/split that best separates the target labels.

**Problem 1**: How to quantify “discriminative-ness”?
- entropy

**Problem 2**: What if no feature is very discriminative?
- try a sequence of features

### Entropy: Measure of discriminativeness

- Entropy measures uncertainty or impurity in the target labels at a node.
- A balanced class distribution has higher entropy.
- A distribution dominated by one class has lower entropy.
- A pure node has entropy 0.

#### Entropy Formula

For class probabilities pᵢ:

> H = -Σ pᵢ log₂(pᵢ)

- pi is probability of ith class or label
- If pi = 0 then H = 0
  If pi = 1 then also H= 0 because log 1 = 0
  When means when label is certain then entropy will be 0
- Here we have written 2 as log base but it should be k where k is number of possible class or labels
- Probabilities (0.5, 0.5) give entropy 1 bit.
- Probabilities (0.9, 0.1) give lower entropy, approximately 0.47 bits.
- Lower entropy means the labels are more concentrated in one class.

### Information Gain

- Information gain measures how much a split reduces entropy.
- Choose the feature/split with the highest information gain.

> IG = Original Entropy – (Split1_size\*Split1_ entropy + Split2_size\*Split2_ entropy)

- ![[Pasted image 20261013084814.png]]
- We multiply with Split_size (which is percentage of frequency of labels)
  to balance the scenario where one split may have very less examples or datapoints and it has less entropy. 
  So after multiplication, entropy with less examples will have less weightage.

**Comparing Splits**

- Company split: information gain ≈ 0.07.
- Price < 300 split: information gain ≈ 0.
- Price < 500 split: information gain ≈ 0.01.
- Company is the better split in this example because it has the highest information gain.
- A small branch may look pure simply because it contains very few examples.

### Decision Stump

- A decision stump is a decision tree with only one split.
- It is simple to interpret but may not capture enough patterns.
- For example, splitting by company gives 69% training accuracy.![[Pasted image 20261014081853.png]]
#### Making a Prediction

1. Start at the root.
2. Evaluate the feature condition.
3. Follow the branch matching the example.
4. Continue until reaching a leaf.
5. Return the leaf's prediction.

Example from the lecture:
- Company = C2 and Price = 350.
- Follow Company = C2, then Price < 500.
- Predicted class: HIGH.
### Decision Tree Training Algorithm

1. Identify the feature that results in maximum information gain
2. Split the dataset accordingly
3. Identify if any feature can result in further decent information gain on the split sets
4. If yes, split further. If no, stop.
5. Goto 3
6. At each leaf, the prediction is the mode label

**Test:**
- Follow sequence of decisions based on features of test example
- Make prediction according to leaf

**Example:**
- ![[Pasted image 20261014082251.png]]

> [!NOTE]
> In a decision tree:
> - All leaf nodes may not lie on same level
> - A split may result in more than 2 branches


#### Overfitting and Stopping

- A very deep tree may learn noise or details specific to the training set. This is overfitting.
- Limit tree growth using stopping conditions, such as maximum depth or minimum samples per split.
- Information gain is a heuristic and does not guarantee a globally optimal tree.

### Multi-class Decision Trees

- Decision trees can predict more than two classes.
- At each leaf, predict the most frequent class among the examples reaching that leaf.
- ![[Pasted image 20261018105720.png]]

### Advantages and Disadvantages of Decision Trees

**Advantages**

- Easy to interpret.
- Easy to use for predictions.
- Provides a ranking of features (according to usefulness)

**Disadvantages**

- Deep trees can overfit.
- Greedy information-gain splitting is a heuristic, not a guaranteed global optimum.
- Uncontrolled growth can create many small branches.

### Regression Trees

- Classification trees predict discrete classes.
- Regression trees predict continuous numerical values, such as an average product rating.
- Eg: Predict Average Rating of a product 
  which is a real number between 1 and 5

#### Regression Split Criteria

- Regression trees use target variance(instead of information gain) to measure how spread out the labels are.
- A good split reduces the weighted variance of child nodes.

> Reduction in Variance = Parent Variance − Weighted Average of Child Variances

- In below example, splitting by company gives variance reduction of about 0.06.
  ![[Pasted image 20261018110653.png]]
- Splitting by video rate gives variance reduction of about 0.14, making it the better of these two splits.
  ![[Pasted image 20261018110705.png]]

#### Regression Leaf Prediction

- A regression tree predicts the mean target value of the examples in a leaf.

### Ensemble Supervised Learning

- An ensemble combines predictions from multiple models.
- Models may be trained on different datasets or different subsets of features.
- The final prediction combines the outputs of the individual models.
- Combining models can reduce variance, especially when their errors are not highly correlated.
- ![[Pasted image 20261018142430.png]]

### Multi-class Classification Strategies

- A basic linear classifier is naturally suited to **binary classification**.
- Multi-class strategies combine several binary classifiers to handle multiple classes.
- ![[Pasted image 20261018142635.png]]

#### One-vs-One

- Train a classifier for every pair of classes.
- ![[Pasted image 20261018143118.png]]
- Each classifier distinguishes its two selected classes and ignores the rest.
- **For K classes, the number of classifiers is K(K − 1) / 2 i.e. $$o(k^2)$$**
- Classify a new example with all classifiers and select the class with the most votes.

Example:
- RED vs BLUE → RED
- BLUE vs GREEN → BLUE
- RED vs GREEN → RED
- Final prediction: RED, because it receives the most votes.
- ![[Pasted image 20261018143133.png]]

#### One-vs-All

- Train one classifier for each class.
- For each classifier, the selected class is positive and all other classes are negative.
- ![[Pasted image 20261018143300.png]]
- **For K classes, we need to train K classifiers** i.e. $$o(k)$$
- Combine their outputs to choose the final class.
- ![[Pasted image 20261018143528.png]]

### Combining Model Predictions

#### Simple Voting

- Classification problem: choose the mode (most frequent predicted class).
- Regression problem: calculate the mean of the predictions.

Example(Classification Problem):
- Predictions: 1, 1, 2, 1, 3, 2.
- Class 1 gets 3 votes, class 2 gets 2, and class 3 gets 1.
- Final classification prediction: class 1.

#### Weighted Voting for Classification

- Assign a weight to each classifier based on its reliability.
- Give more reliable classifier higher weights.
- Add the weights for each predicted class.
- Choose the class with the highest total weight.

Example:
- ![[Pasted image 20261018144441.png]]

#### Weighted Averaging for Regression

- Give more reliable models higher weights.
- Calculate the weighted average of predictions.

> Weighted Prediction = Σ(weightᵢ × predictionᵢ) / Σ(weightᵢ)

- Example:
  ![[Pasted image 20261018144927.png]]
- Model weights can be estimated by evaluating model performance on a common validation set.
### How to build ensemble of classifiers ?

There are multiple ways:

1. Same training set, different classifier functions
2. Split the dataset into multiple parts; train a classifier on each part
3. Select different subsets of features; train a classifier for each subset
4. Bootstrap Aggregation (BAGG-ing): Common and Famous way

Final prediction from ensemble mean or mode -> reduces variance!
#### Bootstrap Aggregation (Bagging)

- Bagging means Bootstrap Aggregation.
- Train multiple models on different bootstrap samples of the original training data. We use one of two ways to create multiple dataset:
	- [[#^249900|Bootstrap Sampling]]
	- [[#^229c1a|Feature Bagging]]
- Give weightage to each classifier according to its accuracy on validation set
- Each bootstrap sample has the same size as the original training set.
- Combine predictions using voting for classification or averaging for regression.
- Bagging can reduce variance and make predictions more stable.
- ![[Pasted image 20261020175455.png]]

##### Bootstrap Sampling

For a training set containing N examples:

1. Give a training set of N examples
   Draw N examples at random with replacement.
2. Each draw selects each original example with probability 1/N.
3. Repeat to create a bootstrap dataset of size N.
4. Train a model on each bootstrap dataset.

**Example**
- ![[Pasted image 20261020175643.png]]
**Notes**
- The probability that a particular example is never selected is (1 − 1/N)ᴺ.
- For large N, this approaches e⁻¹ ≈ 0.368.
  i.e. around 36.8% of original examples are typically absent from a bootstrap sample.

##### Feature Bagging

- Instead of sampling examples, sample subsets of features.
- Each model uses all training examples but only some available features.
- Different feature subsets encourage models to learn different patterns.
- Eg: If we have 5 features, we can select below 5 combinations for five training datasets:![[Pasted image 20261020175921.png]]

### Random Forest

- Random Forest is an ensemble of decision trees.
- Trees are trained using bootstrap samples and random subsets of features.
- Classification: combine trees using majority voting.
- Regression: combine trees by averaging their predictions.
- Randomness in samples and features encourages diversity among trees and can reduce variance.
- ![[Pasted image 20261020181020.png]]

