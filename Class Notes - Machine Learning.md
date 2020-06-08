# Machine Learning - Andrew Ng - Coursera
# Week1 - Introduction
## Intrduction
1. What is ML?
    > "Machine learning is the field of study that gives computers the ability to learn without being explicitly learned" - 1959, Arthur Samuel

    > "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." - 1998, Tom Mitchell  
    > Example of playing checkers:
    > * E = the experience of playing many games of checker
    > * T = the task of playing checkers
    > * P = the probability that the program will win the next game
2. Supervised/Unsupervised Learning
    * Supervised $\to$ inputs and outputs are known in advanced
    * Unsupervised $\to$ only inputs are known and dividing them into clusters is wanted, and there is no feedback based on the prediction results
3. Regression/Clasification Problem
    * Regression $\to$ predict results within a continuous output
    * Classification $\to$ predict results in a discrete output, mapping input variables into categories

## Model and Cost Function
1. Linear Regression
    * Univariate Linear Regression $\to$ only one independent variable
2. Training set $\to$ list of m training examples $(x_{(i)}, y_{(i)})$ where $i=1,\dots,m$
3. The goal is to learn a function h: $X \to Y$ so that $h_{(x)}$ is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis.
4. The cost function $J_{(\theta)}$ depends on the parameter $\theta$ of $h_{(x, \theta)}$, so the goal is to find which $\theta$ makes $J$ minimum, and so making the error of prediction minimum.

## Parameter Learning
1. The params are updated all at the same time.
2. Gradient Descent algorithm is based on updating parameters until convergence:  
    $\theta_j := \theta_j − α \dfrac{\partial J_{(\theta_0, \theta_1)}}{\partial \theta_j}$


---
# Week2 - Linear Regression with Multiple Variables
## Multiple Features
1. Multivariate Linear Regression
    * $x_j^{(i)}$ = value of feature j in the ith training example
    * $x^{(i)}$ = the input (features) of the ith training example
    * $m$ = the number of training examples
    * $n$ = the number of features
    * Regressor: $h_{\theta(x)} = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + ⋯ + \theta_n x_n$ where $x_0=1$ (for matrix operations convenience as it represents the bias)
1. Gradient Descent For Multiple Variables
    * Update parameters until convergence:  
    $\displaystyle \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \quad$ for $j := 0, \dots, n$
2. Feature Scaling and Mean Normalization
    * To avoid inefficient oscillation down to the optimum when the variables are very uneven, modify the ranges of our input variables so that they are all roughly the same:  
    $x_i:=​ \dfrac{x_i​−μ_i}{s_i}​​$  
    $\dots$ where $μ_i$ is the average of all the values for feature $i$, and $s_i$​ may be the range of values $(max - min)$, or the standard deviation as well.
3. Gradient Descent in Practice II - Learning Rate
    * Not too large (may not converge), not too small (may take too long to converge)
4. Features and Polynomial Regression
    * Combine multiple features into one, or even create a new feature which is a function of one or more features (e.g. by making a feature a quadratic, cubic or square root function)

## Computing Parameters Analytically
1. Normal Equation
    * In this method, minimize $J$ by explicitly taking its derivatives with respect to the $\theta j$’s, and setting them to zero, by using this formula:  
    $\theta=(X^T X)^{−1} X^T y$
    * Comparison table:
        | **Gradient Descent**         | **Normal Equation**                                     |
        |------------------------------|---------------------------------------------------------|
        | Need to choose alpha         | No need to choose alpha                                 |
        | Needs many iterations        | No need to iterate                                      |
        | $\mathcal{O}(kn^2)$          | $\mathcal{O}(n^3)$, need to calculate inverse of $X^TX$ |
        | Works well when $n$ is large | Slow if $n$ is very large                               |
        | May require feature scaling  | **No need** to do feature scaling                       |

    * In practice, when n > 10,000 it might be a good time to go from a normal solution to an iterative process
1. Normal Equation Noninvertibility
    * Use the `pinv` (pseudo-inverse matrix function) function rather than `inv`, as it gives a result even if $X^T X$ is not invertible
    * The common causes:
        * Redundant features (i.e. they are linearly dependent)
        * Too many features (e.g. $m \leq n$). If so, delete some features or use regularization


---
# Week3 - Logistic Regression
## Classification and Representation
1. Classification
    * Discrete outputs, with a finite range of possible values; usually $\in [0, 1]$
    * Linear regression doesn't work well because classification is not actually a linear function; it could even take values outside the possible range
2. Hypothesis Representation
    * Binary classification so far
    * Logistic Regression works well on classification tasks
    * Let's introduce a non-linear function to constrain the possible output values:
        * $h_\theta(x) = g(\theta^T x) \quad$ where:
            * $z = \theta^T x$
            * $g(z)$ is the sigmoid function or logistic function: $g(z) = \dfrac{1}{1+e^{−z}}$
    * Outputs represent estimated probabilities
3. Decision Boundary
    * If the threshold is 0.5:
        * $h_\theta(x) \geq 0.5 \to y=1$
        * $h_\theta(x) < 0.5 \to y=0$
    * $\dots$ then we can say:
        * $g(z) \geq 0.5 \quad$ when $z \geq 0$
    * $\dots$ so:
        * $\theta^Tx \geq 0 ⇒ y=1$
        * $\theta^Tx < 0 ⇒ y=0$
    * The decision boundary is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.
    * The input to the sigmoid function $g(z)$ doesn't need to be linear, and could be a function that describes a circle (e.g. $z=\theta_0 + \theta_1 x_1^2 + \theta_2 x_2^2$) or any shape to fit our data.
## Logistic Regression Model
1. Cost Function
    * We cannot use the same cost function (mse) because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.
    * Cost function for logistic regression:
        * $J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)})$
        * $\mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x))\quad$ if $y=1$
        * $\mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x))\quad$ if $y=0$
    * This leads to these conclusions:
        * if $h_\theta(x)=y \quad ⇒ \quad \mathrm{Cost}(h_\theta(x),y) = 0$
        * if $y=0 \land h_\theta(x) \to 1 \quad ⇒ \quad \mathrm{Cost}(h_\theta(x),y) \to \infty$ 
        * if $y=1 \land h_\theta(x) \to 0 \quad ⇒ \quad \mathrm{Cost}(h_\theta(x),y) \to \infty$
        * This way $J(\theta)$ convexity is guaranteed
2. Simplified Cost Function and Gradient Descent
    * Cost function into a single form, without conditional cases:
        * $\mathrm{Cost}(h_\theta(x),y) = - y \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))$
    * From gradient descent:
        * $\theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\theta)$
    * $\dots$ leads to:
        * $\displaystyle \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$
    * Notice that this algorithm is identical to the one we used in linear regression
    * We still have to simultaneously update all values in $\theta$
    * A vectorized implementation is:
        * $\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})$
3. Advanced Optimization
    * "Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize $\theta$ that can be used instead of gradient descent (e.g. they don't need the learning rate)
    * Octave provides them (there are implementation details)
## Multiclass Classification
1. One-vs-all (or One-vs-rest)
    * Now we have:
        * $y \in \lbrace0, 1, \dots, n\rbrace$
        * $h_\theta^{(0)}(x) = P(y = 0 | x; \theta)$
        * $h_\theta^{(1)}(x) = P(y = 1 | x; \theta)$
        * $\dots$
        * $h_\theta^{(n)}(x) = P(y = n | x; \theta)$
        * $\mathrm{prediction} = \max_i( h_\theta^{(i)}(x) )$
    * So we use binary logistic regression to each class, and then use the hypothesis that returned the highest value as our prediction
## Regularization / Solving the Problem of Overfitting
1. The Problem of Overfitting
    * **Underfitting**: high bias, the model can fit better on the train set. It is usually caused by a function that is too simple or uses too few features.
    * **Overfitting**: high variance, fits too well on the train set but does not generalize well to predict new data (can perform better on the validation set). It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.
    * Applies to both linear and logistic regression
    * Adding many new features to the model makes it more likely to overfit the training set
    * Addressing Overfitting:
        1. Reduce the number of features
            * Manually select which features to keep
            * Use a model selection algorithm
        2. Regularization
            * Reduce the magnitude of parameters $\theta_j$
            * Regularization works well when we have a lot of slightly useful features
2. Cost Function
    * We can regularize all the theta parameters in a single summation as:
        * $\min_\theta\ \dfrac{1}{2m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2$
    * The $\lambda$ is the regularization parameter and it determines how much the costs of our theta parameters are inflated. This way, we can smooth the output of our hypothesis function to reduce overfitting.
    * If $\lambda$ is too large, it may smooth out the function too much and cause underfitting.
    * If $\lambda$ is too small, the penalty may be shadowed at the cost function, allowing it to have large values for $\theta$ parameters.
3. Regularized Linear Regression (**Gradient Descent**)
    * Separate out $\theta_0$​ from the rest of the parameters (no sense to penalize it):
        * $\theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)}$
        * $\theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right]$ where $j \in \lbrace 1,2,\dots,n\rbrace$
    * The term $\frac{\lambda}{m}\theta_j$ performs our regularization.
    * It can also be expressed as:
        * $\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$
        * The first term $\alpha\frac{\lambda}{m}$ will be less than 1, reducing the value of $\theta_j$​ on every update
        * The second term is now exactly the same as it was before
3. Regularized Linear Regression (**Normal Equation**)
    * $\theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^T y$
    * where $L$ is the identity matrix except for the upper-left element with a cero instead:
        $$L = \left(\begin{array}{cc}
        0\\
        & 1\\
        && 1\\
        &&& .\\
        &&&& .\\
        &&&&& .\\
        &&&&&& 1\\
        \end{array}\right)$$ 
    * Even if $X^T X$ is non-invertible, adding $λL$ guarantees the overall to be invertible.
4. Regularized Logistic Regression
    * Same as before, we can regularize this equation by adding a term to the end:  
    $J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$


---
# Week4 - Neural Networks: Representation
## Motivations
1. Non-linear Hypotheses
    * Combining features may be a good idea for small number of features
    * Number of cuadratic features becomes $O(n^2)$
    * Most of the combinations will not be useful at all
2. Neurons and the Brain
    * "Neuro-rewiring experiments": rewire neurons from different origins, that part of the brain learns as if it were designed for it, even though it has not been originally routed there!
    * "Seeing with your tongue": electrodes from a camera in the tongue make you "see" through it upon some minutes!
## Neural Networks
1. Model Representation I
    * Model from brain
        * Dendrite $\to$ inputs wire
        * Axion $\to$ output wire
        * Nuclues $\to$ computation core
    * Naming convention
        * $x_0$​ input node is called the `bias unit`, and is always equal to $1$
        * The `input layer`  is the first layer, `layer 1`
        * The last layer is called `output layer`
        * The remaining are the `hidden layers`
    * Indexing convention
        * $a_i^{(j)} =$ `activation` of `unit` $i$ in `layer` $j$
        * $\Theta^{(j)} =$ matrix of `weights` mapping from `layer` $j$ to `layer` $j+1$
2. Model Representation II
    * Flow
        * $z^{(j)} = \Theta^{(j-1)}a^{(j-1)}$
        * $a^{(j)} = g(z^{(j)})$
    * Shape: If network has $s_j$ units in layer $j$ and $s_{j+1}$ units in layer $j+1$
        * $\dots$ then $\Theta^{(j)}$ will be of dimension $(s_{j+1}) \times ((s_j) + 1)$
## Applications
1. Examples and Intuitions I
    * Combinations with logical gates $\to$ `AND`, `OR`, `NOT`, `XNOR`
2. Examples and Intuitions II
    * LeNet5 $\to$ classify handwritten numbers for zip codes
3. Multiclass Classification
    * There is $1$ `unit` in the `output layer` per `class`
    * `One Hot Encoding` is used for the ground truth $y$


---
# Week5 - Neural Networks: Learning
## Cost Function and Backpropagation
1. Cost Function
    * $J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2$
    * $L$ = total number of layers in the network
    * $s_l$​ = number of units (not counting bias unit) in layer $l$
    * $K$ = number of output units/classes
    * The number of rows in our current theta matrix is equal to the number of nodes in the next layer (excluding the bias unit).
2. Backpropagation Algorithm
    * Our goal is to minimize our cost function: $\min_\Theta J(\Theta)$
    * $\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .*\ a^{(l)}\ .*\ (1 - a^{(l)})$
    * $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$
    * $D^{(l)}_{i,j} := \dfrac{1}{m}\left(\Delta^{(l)}_{i,j} + \lambda\Theta^{(l)}_{i,j}\right)\quad$ if $j≠0$
    * $D^{(l)}_{i,j} := \dfrac{1}{m}\Delta^{(l)}_{i,j}\quad$ if $j=0$
    * The matrix $D$ is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get $D_{ij}^{(l)} = \frac \partial {\partial \Theta_{ij}^{(l)}} J(\Theta)$
3. Backpropagation Intuition
    * $\delta_j^{(l)} = \dfrac{\partial}{\partial z_j^{(l)}} cost(t)$
## Backpropagation in Practice
1. Implementation Note: Unrolling Parameters
    * In order to use optimizing functions such as `fminunc()`, "unroll" all the elements into one long vector: `D = [D1(:); D2(:); D3(:)]`
    * If the shapes are `Theta1`: $10x11$, `Theta2`: $10x11$, `Theta3`: $1x11$, to get back to our original matrices from the "unrolled" versions:  
        ```
        Theta1 = reshape(thetaVector(1:110),10,11)
        Theta2 = reshape(thetaVector(111:220),10,11)
        Theta3 = reshape(thetaVector(221:231),1,11)
        ```
2. Gradient Checking
    * $\dfrac{\partial}{\partial\Theta}J(\Theta) \approx \dfrac{J(\Theta + \epsilon) - J(\Theta - \epsilon)}{2\epsilon}$
    * $\dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon}$
    * ${\epsilon = 10^{-4}}$ being a small value guarantees that the math works out properly. If too small, we may end up with numerical problems. 
    * Code:
        ```
        epsilon = 1e-4;
        for i = 1:n
            thetaPlus = theta;
            thetaPlus(i) += epsilon;
            thetaMinus = theta;
            thetaMinus(i) -= epsilon;
            gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon);
        end
        ```
    * Once you have verified that your backprop is correct, turn it off, as the code to compute gradApprox can be very slow.
3. Random Initialization
    * When we backpropagate, all nodes will update to the same value repeatedly $\to$ all neurons learning the same feature $\to$ redundant information.
    * Instead, randomly initialize the weights (e.g. uniform distribution $\in [-\epsilon,\epsilon]$: `Theta1 = rand(10,11) * 2*INIT_EPSILON - INIT_EPSILON;`)
4. Putting It Together
    * Define architecture of the NN:
        * Nº of input units $=$ dimension of features $x^{(i)}$
        * Nº of output units $=$ nº of classes
        * 1 hidden layer by default. If more, then it is recommended the same nº of units in every hidden layer
    * Training a Neural Network:
        1. Randomly initialize the weights
        2. Implement forward propagation to get $h_\Theta(x^{(i)})$ for any $x^{(i)}$
        3. Implement the cost function
        4. Implement backpropagation to compute partial derivatives
        5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
        6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.
    * However, keep in mind that $J(\Theta)$ is not convex and thus we could end up in a local minimum.
## Application of Neural Networks
1. Autonomous Driving
    * "ALVINN"


---
# Week6 [Part 1] - Advice for Applying Machine Learning
## Evaluating a Learning Algorithm
1. Deciding What to Try Next
    * Which way is more efficient to take and spend time in order to improve your model?
    * Diagnostics:
        * They can give guidance as to what might be more fruitful things to try to improve a learning algorithm
        * They can be time-consuming to implement and try, but they can still be a very good use of your time.
2. Evaluating a Hypothesis
    * Split up the data into a training set and a test set
3. Model Selection and Train/Validation/Test Sets
    * By using the validation set to select the best model, some specific settings have been fit to the cross validation set, and thus we need a new set to estimate the generalization error.
    * Training set: $60\%$ - Cross validation set: $20\%$ - Test set: $20\%$
        1. Fit using training set
        2. Find the model with the least error using the cross validation set
        3. Estimate the generalization error using the test set
## Bias vs. Variance
1. Diagnosing Bias vs. Variance
    * $\uparrow$ High bias $\to$ underfitting $($usually $J_{CV(\Theta)} \approx J_{train(\Theta)})$
    * $\uparrow$ High variance $\to$ overfitting $(J_{CV(\Theta)} >> J_{train(\Theta)})$
2. Regularization and Bias/Variance
    * Tune hyperparameter $\lambda$:
        1. Fit several models with different values for $\lambda$
        2. Find the model with the least validation error $J_{CV(\Theta)}$ (computed without regularization: $\lambda=0$)
        3. Estimate the generalization error using the test set
3. Learning Curves
    * If a learning algorithm is suffering from high bias, getting more training data will not help much
    * If a learning algorithm is suffering from high variance, getting more training data is likely to help
4. Deciding What to Do Next Revisited
    * Our decision process can be broken down as follows
        * Getting more training examples: Fixes high variance
        *  Trying smaller sets of features: Fixes high variance
        *  Adding features: Fixes high bias
        *  Adding polynomial features: Fixes high bias
        *  Decreasing λ: Fixes high bias
        *  Increasing λ: Fixes high variance.
    * Diagnosing Neural Networks
        * A NN with fewer parameters is prone to underfitting. It is also computationally cheaper.
        * A large NN with more parameters is prone to overfitting. It is also computationally expensive.
    * Model Complexity Effects
        * Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
        * Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.
        * In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.
# Week6 [Part 2] - Machine Learning System Design
## Building a Spam Classifier
1. Prioritizing What to Work On
    * System Design Example: Given a data set of emails, construct a vector so that each entry represents a word. The vector normally contains 10,000 to 50,000 entries gathered by finding the most frequently used words in our data set, and indicates presence of each word in the email. Once we have all our x vectors ready, we can train our algorithm to classify if an email is a spam or not.
        * So how could you spend your time to improve the accuracy of this classifier?
            * Collect lots of data (e.g. "honeypot" project)
            * Develop sophisticated features (e.g. using email header data in spam)
            * Develop algorithms to process text (e.g. recognizing misspellings in spam)
        * It is difficult to tell which of the options will be most helpful.
2. Error Analysis
    *  Recommended approach:
        *  Start with a simple algorithm, implement it quickly, and test it early on your cross validation data.
        * Plot learning curves to decide if more data, more features, etc. are likely to help.
        * Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.
        * It is very important to get error results as a single, numerical value. Otherwise it is difficult to assess the overall algorithm's performance.
    * e.g.: Assume that we have 500 emails and our algorithm misclassifies a 100 of them. Analyze the 100 emails and categorize them based on what type of emails they are. Try to come up with new cues and features that would help us classify these 100 emails correctly. Hence, if e.g. most of our misclassified emails are those which try to steal passwords, then we could find some features that are particular to those emails and add them to our model.
    * e.g.: If we use stemming, which is the process of treating the same word with different forms (fail/failing/failed) as one word (fail), and get a 3% error rate instead of 5%, then we should definitely add it to our model. However, if we try to distinguish between upper case and lower case letters and end up getting a 3.2% error rate instead of 3%, then we should avoid using this new feature.
## Handling Skewed Data
1. Error Metrics for Skewed Classes
    * Skewed Classes: minority, rare classes, those that have critically lower number of examples in the dataset.
    * Useful error metrics for these cases:
        * $\text{Precision} = \frac{\text{True positives}}{\text{\# predicted as positive}} = \frac{\text{True positives}}{\text{True positives + False positives}}$
        * $\text{Recall} = \frac{\text{True positives}}{\text{\# actual positives}} = \frac{\text{True positives}}{\text{True positives + False negatives}}$
2. Trading Off Precision and Recall
    * For a binary classifier, remember:
        * $y_{pred}=1 \quad$ if $h_\theta(x) \geq \text{threshold}$
        * $y_{pred}=0 \quad$ otherwise
    * Different values for `threshold` $\to$ different values of precision $(P)$ and recall $(R)$
        * Plot precision $(P)$ vs recall $(R)$
    * To get a unique value to measure performance in our models:
        * Average $(\frac{P+R}{2})$ is not a good metric, as it doesn't reflect the problem for skewed classes.
        * `F1-score` $(2\frac{PR}{P+R})$ falls near $0$ when any of $P$ or $R$ are very low, and tends to $1$ when $P$ and $R$ tend to $1$.
## Using Large Data Sets
1. Data For Machine Learning
    * Can expert humans do a good job (confidently) given the inputs of the dataset?
        * If yes, large amount of data may help
    * Using a complex predictor help having low bias
    * Using a large dataset help having low variance


---
# Week7 - Support Vector Machines
## Large Margin Classification
1. Optimization Objective
    * Similar to the cost function
    * Classification:
        * $h_\theta(x) = \begin{cases}
            1 &\quad\text{if}\ \theta^Tx \geq 0 \\
            0 &\quad\text{otherwise} \\
            \end{cases}$
    * If $C = \frac{1}{\lambda}$, these two optimization problems will give the same value of $\theta$:
        * $\displaystyle\mathop{min}_\theta\ \frac{1}{m}\left[\sum_{i=1}^{m}y^{(i)}\text{cost}_1(\theta^Tx^{(i)}) + (1-y^{(i)}) \text{cost}_0(\theta^Tx^{(i)})\right]+\frac{\lambda}{2m}\sum_{j=1}^n\theta^2_j$
        * $\displaystyle\mathop{min}_\theta\ C\left[\sum_{i=1}^{m}y^{(i)}\text{cost}_1(\theta^Tx^{(i)}) + (1-y^{(i)}) \text{cost}_0(\theta^Tx^{(i)})\right]+\frac{1}{2}\sum_{j=1}^n\theta^2_j$
        * Note that $\text{cost}_0()$ corresponds to the cost function when $(y=0)$, and $\text{cost}_1()$, when $(y=1)$.
        * A large $C$ parameter tells the SVM to try to classify all the examples correctly.
2. Large Margin Intuition
    * Works well with linearly separable data.
    * The large margin refers to the fact that SVM looks for the boundary limit that maximizes the distance between the classes to split.
    * For large values of $C$, SVM will adapt more to fit all examples (even for outliers), whereas in contrast, for not too large values of $C$, SVM will adapt more to the mayority of the examples (and not overfit to outliers e.g.).
3. Mathematics Behind Large Margin Classification
    * Suppose 2 vectors $\{u, v\}$:
        * Their inner product $u^Tv = \|u\| \cdot p \quad$ where:
            * $p$ is the projection of $v$ onto $u$
            * $\|u\|$ is the norm of $u$
    * Applied to SVM: $\theta^Tx^{(i)} = \|\theta\|\cdot p^{(i)}$
        * $p^{(i)}$ is the signed (positive or negative) projection of $x^{(i)}$ onto $\theta$
        * $\displaystyle\mathop{min}_\theta\ \frac{1}{2}\sum_{j=1}^n\theta_j^2 = \frac{1}{2}\|\theta\|^2 \quad \text{such that}
            \begin{cases}
                \|\theta\|\cdot p^{(i)} \geq +1 &\quad \text{if}\ y^{(i)} = 1 \\
                \|\theta\|\cdot p^{(i)} \leq -1 &\quad \text{if}\ y^{(i)} = 0 \\
                \end{cases}$
    * The Decision Boundary is orthogonal to the vector $\theta$
    * If the decision boundary (intrinsically set by $\theta$) makes the margins to be small, then $\|\theta\|$ need to be large to fulfill the condition $\|\theta\|\cdot p^{(i)} \geq 1$ for positive examples and $\|\theta\|\cdot p^{(i)} \leq -1$ for negative examples.
        * This is why SVM, while minimizing $\|\theta\|^2$ and accomplishing the aforementioned conditions, will tend to choose $\theta$ such that its decision boundary makes the margins to be large, so that $\|\theta\|$ may be as small as possible.
## Kernels
1. Kernels I
    * Used for a non-linear decision boundary
    * Instead of using the inputs $x$ directly, we can use new features based on proximity to `landmarks` placed on the $x$ domain.
        * Then, we can compute the similarity function (`kernel`) as a distance between any example $x$ and the the first landmark $l^{(1)}$:
            * $f_1(x) = \text{similarity}(x, l^{(1)}) = \text{sim}(x, l^{(1)})$
        * In the case of choosing *Gaussian kernels*:
            * $f_1(x)  = e^{\displaystyle -\frac{\|x-l^{(1)}\|}{2\sigma^2}} = \begin{cases}
                \approx 1 &\quad\text{if}\ x\ \text{is close to}\ l^{(1)} \\
                \approx 0 &\quad\text{if}\ x\ \text{is far from}\ l^{(1)} \\
                \end{cases}$
            * It is a smooth normal distribution centered at the landmark, and with a bandwidth parameter $\sigma$, which determines how fast the similarity metric decreases to 0 as the examples are further apart
2. Kernels II
    * The landmarks are chosen exactly over each of the training examples: $l^{(i)} = x^{(i)}$:
        * There are $m$ landmark features $\implies n=m$
        * $f_i(x) = \text{sim}(x, x^{(i)}) \implies f_i(x^{(i)})=1$
    * There are 2 parameters to adjust: $C(=\frac{1}{\lambda})$ and $\sigma$.
        * Overfitting: to regularize and improve variance you may $\begin{cases}
            \downarrow C \\
            \uparrow \sigma^2 \\
            \end{cases}$
        * Underfitting: to improve bias you may $\begin{cases}
            \uparrow C \\
            \downarrow \sigma^2 \\
            \end{cases}$
    * There is an implementation detail: $\sum_{j=1}^n\theta_j^2 = \|\theta\|^2 = \theta^T \theta$ which is computationally more efficient. Even more, a slightly different but more efficient version is $\theta^T M \theta$.
## SVMs in Practice
1. Using An SVM
    * Use a library or framework that solves the optimization algorithms for us.
    * We have to:
        * Tune $C(=\frac{1}{\lambda})$
        * Choose the `kernel` (similarity function)
            * `Gaussian kernel` (most common chosen) $\to$ tune $\sigma$.
            * `No kernel` (*linear kernel*) $\to$ standard linear classifier (may be useful if $n$ is large and $m$, small).
            * `Ploynomial kernel`: $k_{(x)} = (x^T l + \text{constant})^\text{degree}$. It is usually used when $x$ are all positive values.
            * More esoteric: `string kernel` (used for strings), `chi-square kernel`, `histogram kernel`, `intersection kernel`.
        * Kernels have to satisfy *Mercer's Theorem* (to not diverge)
        * Feature scaling is expected
        * To tune up your models, choose whatever performs best on the cross-validation data
    * Multi-class SVM
        * Use a framework
        * One-vs-All
    * Logistic Regression vs SVM
        * If $n$ is large compared to $m$ (e.g. $n=10^4 \wedge m=10 \sim 10^3$) $\implies$ use logistic regression or SVM with linear kernel.
        * If $n$ is small and $m$ intermediate (e.g. $n=1 \sim 10^3 \wedge m=10 \sim 10^4$) $\implies$ use SVM with Gaussian kernel.
        * If $n$ is small and $m$ is large $(\text{e.g.}\ n=1 \sim 10^3 \wedge m=10^5 \sim +) \implies$ create/add more features and use logistic regression or SVM with linear kernel.
    * Neural Networks are likely to work well for most of these settings, but may be slower to train.


---
# Week8 [Part 1] - Unsupervised Learning
## Clustering
1. Unsupervised Learning: Introduction
    * Our examples have inputs, but not expected outputs
    * The goal is to find some kind of "structure" in the data.
    * Application examples: Mostly clustering (grouping examples)
        * Market Segmentation (e.g. split T-shirts for Small, Medium and Large sizes)
        * Social Network Analysis
        * Orginze computing clusters
        * Astronomical Data Analysis
2. K-Means Algorithm
    * Has $K$ cluster centroids
    * Iterate algorithm until convergence:
        1. Cluster assignment: each example is linked to the closest cluster centroid.
            * If somehow one cluster gets no examples linked to it, then you can delete it (and keep going with $K-1$ clusters) or reinitilize all clusters, although in practice this happens not that often.
        2. Move centroids: each cluster centroid is moved to the mean of all its linked examples.
    * The goal is to minimize the overall distance $\|x^{(i)}-\mu_k\|^2 \quad\text{where}\ k\in\{1,2,\dots,K\}$.
3. Optimization Objective
    * $\displaystyle J = \dfrac{1}{m} \sum_{i=1}^{m} \|x^{(i)}-\mu_{c^{(i)}}\|^2 \quad \text{where} \begin{cases}
            c^{(i)}\ \text{is the linked cluster of}\ x^{(i)}\\
            \mu_{c^{(i)}}\ \text{is the centroid location of}\ c^{(i)} \\
            \end{cases}$
    * Also named *Cost function* or *Distortion function* of `K-means`.
    * Moving each centroid to its linked examples mean is proven to minimize the cost function.
    * The cost function across iterations should never increase.
4. Random Initialization
    * There are several methods. The most used:
        * Pick $K$ (where $K<m$) random training examples and set the centroids on them.
    * Multi-Random Initialization: to increase the odds of converging to the global optima, run it several $(50 \sim 1000)$ times and pick the one with lowest $J$.
        * If $K<10$, it is likely to get a largely better solution
        * If $K>10$, the first solution is likely to work pretty decently (maybe get a slightly better solution)
5. Choosing the Number of Clusters
    * The most common thing is actually to choose the number of clusters by hand
        * But sometimes it is ambiguous to choose it manually
    * Elbow Method
        * Train and get solutions for differents values of $K$, then plot the cost function $J$ for each one as a function of $K$; it should generally decrease as $K$ incereases (try multi-random initialization).
            * If there is a clear break on the smoothness of $J$ (an "elbow"), then it would be ok to choose $K$ on that point.
            * Sometimes it is not clear where is "the elbow".
    * Evaluate the purpose of the problem
        * E.g. choose T-shirt sizes into $K=3$ (S, M, L) or $K=5$ (XS, S, M, L, XL).
# Week8 [Part 2] - Dimensionality Reduction
## Motivation
1. Motivation I: Data Compression
    * The goal is to reduce redundancy, lower the number of dimensions by finding highly correlated features.
        * $\{x^{(1)},\dots, x^{(m)}\}$, where $x^{(i)}\in\mathbb{R}^n \to \{z^{(1)}, \dots, z^{(m)}\}$ where $z^{(i)} \in \mathbb{R}^k$ for some value of $k\leq n$.
    * Think of a 2D line (with almost no area) that can be mapped into new one dimension of data. Or maybe a 3D plane (with almost no volume) that me be mapped into a new 2 dimensions of data.
        * Project the examples over the line / plane to reduce one dimension.
2. Motivation II: Visualization
    * It may be very helpful to plot somehow yous high dimensional data, but it is not very easy to do that when $n>3$.
    * For visualization purposes, we may try to reduce the $n$-dimensional data into a $k$-dimensional plot, where $(k\leq n) \wedge (k=2 \vee k=3)$ for 2D or 3D plots.
## Principal Component Analysis
1. Principal Component Analysis Problem Formulation
    * Reduce from $n$-dimensions to $k$-dimensions
        * Find $k$ vectors $\{u^{(1)},\dots, u^{(k)}\}$ onto which to project the data, so as to minimize the projection distances error.
        * $k$ is also called *the Number of Principal Components*.
    * Finding those vectors may look like we are doing Linear Regression, **which is incorrect**:
        * Linear Regression works with a function where the distances are over the output axis (e.g. "vertical" distances for 2D data: $x \to y$)
        * The distances are measured against their shortest distance (orthogonal to the projection vector), and there is no $x \to y$ mapping.
2. Principal Component Analysis Algorithm
    1. Preprocess all your data to have zero mean (mandatory), and (optionally) apply feature scaling (or normalize it directly).
    2. Compute the covariance matrix $\text{Sigma}$
        * $\displaystyle \text{Sigma} = \dfrac{1}{m}\sum_{i=1}^n (x^{(i)})(x^{(i)})^T \quad \text{where}\ \text{Sigma} \in \mathbb{R}^{n \times n}$
    3. Compute the eigen-vectors of $\text{Sigma}$
        * `[U, S, V] = svd(Sigma);`
        * `svd()` stands for *Single Value Decomposition*.
            * `eig()` would return the same eigen-vectors for this matrix, although `svd()` is more numerically stable.
        * $U$ is the matrix of $u$-vectors
            * $U = \begin{bmatrix}\vert & & \vert \\ u^{(1)} & \cdots & u^{(n)}\\ \vert & & \vert \end{bmatrix} \quad \text{where}\ U \in \mathbb{R}^{n \times n}$
            * The first columns represent the $u$-vectors on which the projection error is lower.
    4. Take the first $k$ columns of $U \to U_{\text{reduce}} \in \mathbb{R}^{n \times k}$
    5. To map  $x^{(i)} \to z^{(i)}$:
        * $z^{(i)} = (U_{\text{reduce}})^T x^{(i)} = \begin{bmatrix} \text{---} & (u^{(1)})^T & \text{---}\\ & \vdots & \\ \text{---} & (u^{(k)})^T & \text{---}\end{bmatrix} x^{(i)} \quad \text{where} \begin{cases}
            x^{(i)} \in \mathbb{R}^{n \times 1} \\
            z^{(i)} \in \mathbb{R}^{k \times 1} \\
            \end{cases}$
        * $z_j = (u^{(j)})^T x$
## Applying PCA
1. Reconstruction from Compressed Representation
    * Mapping $z^{(i)} \to x^{(i)}$:
        * $x^{(i)} \approx x_{\text{approx}}^{(i)} = U_{\text{reduce}}\ z^{(i)}$
2. Choosing the Number of Principal Components
    * Average Squared Projection Error $\to \displaystyle \text{ASPE} = \frac{1}{m} \sum_{i=1}^m \|x_{\text{approx}}^{(i)} - x^{(i)}\|^2$
    * Total Variation in the Data $\to \displaystyle \text{TVD} = \frac{1}{m} \sum_{i=1}^m \|x^{(i)}\|^2$
    * Lost Variance $\to \text{LV} = \dfrac{\text{ASPE}}{\text{TVD}}$
    * Retained Variance $\to \text{RV} = 1 - \text{LV}$
    * How to choose $k$?
        * Typically, choose $k$ so that at least $``99\%$ of variance is retained$"\to \text{LV} \leq 0.01\ (1\%)$
        * How? Iterate (from $k=1$ on) and compute $\text{LV}$ until our condition is met
            * Don't train for each $k$ (very inefficient), use the output `S` of `svd(Sigma)` instead.
                * $\text{S}$ is a diagonal matrix where $\text{RV} = \dfrac{\sum_{i=1}^k \text{S}_{ii}}{\sum_{i=1}^n \text{S}_{ii}}$
        * Most common values for $\text{RV} = 90\% \sim 99\%$
    * The $\text{RV}$ is a very natural and popular way to express the performance of your PCA model, as it tells **how well your reduced representation is approximating your original data**.
3. Advice for Applying PCA
    * Supervised Learning Speedup and Memory Saving
        * When using labeled data, we may speed up the trainings (barely affecting the accuracy of the learning algorithm) by applying PCA to the input examples of the whole dataset, but computing the mapping $x^{(i)} \to z^{(i)}$ only with the training set.
        * It is also a good way to save memory space.
    * Do not use PCA to prevent overfitting!
        * While it is true that the number of input features decreases, and thus, it may help a little preventing to overfit data, PCA doesn't take in mind the ground truth outputs of the data and may throw away some valuable information, so this is a bad idea.
        * Other regularization methods work much better and they are less likely to remove valuable information (as the outputs are involved in the cost function).
    * Advise: don't plan to use PCA directly in combination to another training algorithm, just try it without PCA and see how it goes. Only apply PCA if there is a reason to do so:
        * Training data don't fit in memory / disk space
        * Training is too slow


---
# Week9 [Part 1] - Anomaly Detection
## Density Estimation
1. Problem Motivation
    * Given unlabeled data, build a model that outputs the probability of a new given example to be normal (vs anomalous)
        * Then, if that probability is low enough $(p_{(x)} < \epsilon) \implies$ flag as an *anomaly*.
    * Application examples:
        * Fraud Detection $\to$ users activities
        * Manufacturing $\to$ aircraft engine anomalies
        * Monitoring $\to$ computers in a data center; supplying electricity to customers and want to see if anyone might be behaving strangely.
        * A security application, where you examine video images to see if anyone in your company’s parking lot is acting in an unusual way.
2. Gaussian Distribution
    * $x \sim \mathcal{N}(\mu, \sigma^2) \quad \to x$ is distributed as a *Normal* (Gaussian) function $\mathcal{N}$, represented by the parameters: mean $\mu$, and variance $\sigma^2$ (or standard deviation $\sigma$).
        * $p{(x;\ \mu,\ \sigma^2)} = \dfrac{1}{\sqrt{2\pi}\ \sigma} e^{-\dfrac{(x−\mu)^ 2​}{2 \sigma^2}}$
        * The *Area* under the curve of the bell is $1$.
    * Parameter estimation
        * Considering each feature $j$ of any example in the dataset $\sim \mathcal{N}(\mu, \sigma^2)$:
            * $\displaystyle \mu_j ​= \frac{1}{m} \sum_{​i=1}^{m} ​x_j^{(i)}$
            * $\displaystyle \sigma_j^2 ​= \frac{1}{m} \sum_{​i=1}^{m} (​x_j^{(i)} - \mu_j)​^2$
    * Density Estimation
        * $\displaystyle p{(x \in \mathbb{R}^n)} = \prod_{​j=1}^{n} p(x_j;\ \mu_j,\ \sigma_j^2)$
3. Algorithm
    1. Choose relevant features to detect anomalies.
    2. Fit the $n$ pairs of parameters $(\mu_j,\ \sigma_j^2)$.
    3. Given a new example, compute $p_{(x)}$.
    4. If $p_{(x)} < \epsilon \implies$ flag $x$ as anomalous.
## Building an Anomaly Detection System
1. Developing and Evaluating an Anomaly Detection System
    * It would be nice to evaluate somehow our models
        * To do so, we'll collect some labeled data of both anomalous and non-anomalous examples.
        * Assume our training set are (at least almost) all non-anomalous examples.
        * Create the validation and test sets, both with the all the labeled data, mixed with some unlabeled examples as well.
            * E.g. $\begin{cases}
            \text{train set} \to 6000 \text{ good engines}\\
            \text{valid set} \to 2000 \text{ good engines} + 10 \text{ anomalous}\\
            \text{test set} \to 2000 \text{ good engines} + 10 \text{ anomalous}\\
            \end{cases}$
    * Very skewed classes $\implies$ choose very carefully your metrics
        * Confusion matrix
        * Precision/Recall
        * $F_1\text{-score}$
    * Tune your models using your validation set
        * Parameter $\epsilon$
        * Features to include
2. Anomaly Detection vs. Supervised Learning
    * Anomaly Detection:
        * Very small number of positive examples ($0 \sim 20$ is common).
        * Many different "types" of anomalies (hard to learn from very specific examples what an anomaly looks like, as future anomalies may look nothing like the shown examples).
    * Supervised Learning:
        * Large number of positive and negative examples
        * Enough positive examples to get a sense of what a future positive examples should look like.
        * Examples
            * Email spam classification
            * Weather prediction
            * Cancer classification
3. Choosing What Features to Use
    * Non-gaussian features
        * Make them gaussian performing some transformation (if possible):
            * $log(x_\text{non-gaussian} + c) \sim \text{gaussian}$
            * $(x_\text{non-gaussian})^a \sim \text{gaussian}$
        * Help yourself plotting the histogram: `hist(X, bins)`
    * Error analysis
        * Carefully examine the anomalies and use them to inspire youself on finding new features that made them anomalies. Even think of combinations of already existing features (this is automatically captured when using the covariance matrix in the next videos).
            * Data Center e.g.: $\begin{cases}
                x_1 = \text{RAM in use} \\
                x_2 = \text{Disk accesses/sec} \\
                x_3 = \text{CPU load} \\
                x_4 = \text{Network traffic} \\
                x_5 = \dfrac{x_3\ ^2}{x_4} \\
                \end{cases}$
## Multivariate Gaussian Distribution (Optional)
1. Multivariate Gaussian Distribution
    * Using the *Density Estimation*, we assume that all features are completely independent. If not, we may not include some important information about combination of features.
        * E.g.: Normal examples of $2$ features lay close to a $45º$ sloped line, from $(-1,-1)$ towards $(1,1)$. An anomaly may lay on $(0,0.9)$ and is clearly far from normal examples, but with the current implementation, it lays in normal ranges of features $(x1, x2)$ independently.
    * To include these combinations and make our model more robust, compute $p(x)$ all in one go rather than $\{p(x_1), p(x_2), \dots, p(x_n)\}$ spearately.
        * Parameters: $\mu \in \mathbb{R}^n,\ \Sigma \in \mathbb{R}^{n \times n}$ (covariance matrix).
        * $p(x;\ \mu,\ \Sigma) = \dfrac{1}{\sqrt{(2\pi)^n |\Sigma|}} \ e^{-\dfrac{(x−\mu)^T \ \Sigma^{-1} \ (x−\mu)}{2}} \quad$ where $|\Sigma|$ is the determinant of the covariance matrix $\Sigma$.
        * This way, $p(x)$ looks more like a Gaussian distributed probability (similar to the one seen on PCA) where features may not be completely independent to each other, and where combinations of features make sense.
            * The main diagonal correspondends to the correlation of each feature itself.
            * Other values corresponds to the cross-correlation of the features against each other.
2. Anomaly Detection using the Multivariate Gaussian Distribution
    * *Density Estimation (original model)*
        * Computationally cheaper (scales better for datasets with large $n$).
        * It is a particular case of *Multivariate Gaussian Distribution* where all features are considered completely independent (convariance matrix is a diagonal matrix).
        * Mostly used
    * *Multivariate Gaussian Distribution*
        * No need to create new features based on the already existing ones, as it automatically captures correlations between them.
        * Must have $m>n$ and non-redundant features, or else $\Sigma$ is non-invertible.
        * Rule of thumb: Use this if $m \geq 10\ n$ and if $n$ is not too large.
# Week9 [Part 2] - Recommender Systems
## Predicting Movie Ratings
1. Problem Formulation
    * Application examples
        * What to buy next: Amazon, eBay, etc.
        * What to watch next: Netflix, etc.
    * E.g.: Predicting movie ratings ($0\sim5$ stars)
        * Matrix of ratings $\in \mathbb{R}^{n_m \times n_u}$, where:
            * Rows are ratings of a specific movie across $n_u$ users.
            * Columns are ratings of a specific user across $n_m$ movies.
            * Values are: $(0 \sim 5)$ and $(?)$ (for unseen movie / unknown rating).
2. Content Based Recommendations
    * Use embeddings to map from a space of movie titles to a continuous vector space with conceptual features (like romance, comedy, action, etc.)
    * We could use Linear Regression
        * $r(i,j)=1$ if user $j$ has rated movie $i$, else $0$
        * $y^{(i,j)} =$ Rating by user $j$ on movie $i$ (if defined)
        * $\theta^{(j)} =$ Parameter vector for user $j$
        * $x^{(i)} =$ Feature vector for movie $i$
        * Predicted rating $= (\theta^{(j)})^T (x^{(i)})$
        * $m^{(j)} =$ No. of movies rated by user $j$
    * The objective is to, given movies features $x$, minimize the MSE of this Linear Regression task across all users, to obtain parameters $\{\theta^{(1)}, \dots, \theta^{(n_u)}\}$ to predict on new / unseen movies to recommend.
        * $\displaystyle \min_{\theta^{(1)},\dots,\theta^{(n_u)}} \frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}\left((\theta^{(j)})^T x^{(i)}-y^{(i,j)}\right)^2 + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n(\theta_k^{(j)})^2$
## Collaborative Filtering
1. Collaborative Filtering
    * Sometimes it is hard to get all the features for each movie when $n_m$ is large, so it may be convienent to, given the users parameters $\theta^{(j)}$, estimate the features of all movies upoon this formula:
        * $\displaystyle \min_{x^{(1)},\dots,x^{(n_m)}} \frac{1}{2}\sum_{i=1}^{n_m}\sum_{j:r(i,j)=1}\left((\theta^{(j)})^T x^{(i)}-y^{(i,j)}\right)^2 + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n(x_k^{(i)})^2$
    * This is somehow a game of *who goes first* between $x$ features and $\theta$ paramters
        * One thing to do is try to guess $\theta$ based on some $x$, then guess $x$ based on new $\theta$, and iterate over and over again until convergence.
    * Users ratings are *collaboratively* helping to improve the recommender system.
2. Collaborative Filtering Algorithm
    * So far we may want to:
        * Given $x$, estimate $\theta$
        * Given $\theta$, estimate $x$
    * Instead of iterating $x \to \theta \to x \to \theta \to x \to \dots,$ minimize $x$ and $\theta$ simultaneously in the same optimization objective:
        * $\begin{cases}
            \displaystyle J_{({x^{(1)},\dots,x^{(n_m)}}, {\theta^{(1)},\dots,\theta^{(n_u)}})} = \frac{1}{2} \sum_{(i,j):r(i,j)=1}\left((\theta^{(j)})^T x^{(i)}-y^{(i,j)}\right)^2 + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n(x_k^{(i)})^2 + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n(\theta_k^{(j)})^2 \\ \\
            \displaystyle \min_{\substack{x^{(1)},\dots,x^{(n_m)} \\ \theta^{(1)},\dots,\theta^{(n_u)}}} \quad J_{({x^{(1)},\dots,x^{(n_m)}}, {\theta^{(1)},\dots,\theta^{(n_u)}})} \\
            \end{cases}$
    * Algorithm:
        1. Initilize $x$ and $\theta$ with low random values $\to$ symmetry breaking (similar to NN) ensuring the algorithm learns features $x^{(1)}, \dots, x^{(n_m)}$ that are different from each other.
        1. Minimize $J_{({x^{(1)},\dots,x^{(n_m)}}, {\theta^{(1)},\dots,\theta^{(n_u)}})}$
        2. Predict the rating for any movie and any user.
    * This makes the features not easy to understand by a human, but they usually are what best captures the content or properties of different movies that casuses users to like or dislike them.
## Low Rank Matrix Factorization
1. Vectorization: Low Rank Matrix Factorization
    * Vectorization
        * $X \in \mathbb{R}^{n_m \times n} \to x^{(i)}$ are stored in rows
        * $\Theta \in \mathbb{R}^{n_u \times n} \to \theta^{(j)}$ are stored in rows
        * $X \Theta^T \to$ *Low Rank Matrix*
    * Finding related movies based on learned features
        * Get movies whose distance $\|x^{(i_\text{current})} - x^{(i_\text{other})}\|$ is small, so it brings similar movies
2. Implementational Detail: Mean Normalization
    * For users who haven't rated any movie yet, this algortihm will predict a rating of $0$ for every movie (regularization term) $\to$ not very useful
    * Mean Normalization:
        * $Y \in \mathbb{R}^{n_m \times n_u}$
        * $\mu_\text{movie} \in \mathbb{R}^{n_m} =$ movie-wise average across all users
        * Train using $Y_\text{norm} = Y - \mu_\text{movie}$
        * Predict using $(\theta^{(j)})^T x^{(i)} + \mu_i$
        * For non-rated-movies users, predictions will be the average ratings across all other users $\to$ much more useful
    * Note: All the movie ratings are already comparable ($0 \sim 5$ stars), so they don't need feature scaling.


---
# Week10 - Large Scale Machine Learning
## Gradient Descent with Large Datasets
1. Learning With Large Datasets
    * With large datasets, every training step becomes too expensive $\to$ new techniques
    * How to know if all that data will help?
        * Plot learning curves: $J_\text{train}$ and $J_\text{cv}$ as a function of $m$
2. Stochastic Gradient Descent
    * Batch Gradient Descent $\to$ compute the whole training set to update parameters $\to J_\text{train}$ is strictly decreasing
    * Stochastic Gradient Descent $\to$ shuffle the dataset and update parameters for every single training example $\to J_\text{train}$ is very noisly decreasing
3. Mini-Batch Gradient Descent
    * Use batches of $b$ mini-batch size $(2 \sim 100)$ examples $\to$ shuffle the dataset and update parameters for every $b$ training examples $\to J_\text{train}$ is noisly decreasing
    * It is a balance between the learning speed (updating parameters more often than *BGD*) and the computational speed (more vectorization, and thus, faster processing than SGD)
4. Stochastic Gradient Descent Convergence
    * Some advices:
        * Plot $J_\text{train}$ using a big number of averaged examples
        * If too noise, the $J_\text{train}$ plot may seem to not be converging $\to$ increase the number of averaged examples to verify that
        * If $J_\text{train}$ diverges $\implies$ decrease $\alpha$
        * Use *Learning Rate Decay* to help to reduce oscillations near the minimum
            * E.g.: $\alpha = \dfrac{\text{const1}}{\text{iterationNumber} + \text{const2}}$
## Advanced Topics
1. Online Learning
    * Applicable where the data comes from a sort of endless source, a continuous stream of data
        * Models learn from examples once, and then get rid of them
        * Models can adapt to new circumstances over time
    * Application examples
        * Users preferences on purchasing the shipping of a product they just bought $\to$ estimate the probability of user accepting shipping
        * *CTR* (*Click Through Rate*): Prodcut search problem, where users search for products and we can learn what to show them $\to p(y=1|\ \ x;\theta)$: estimate the probability of making click on the returned results
        * Special offers to show to users
        * Customized selection of new articles
        * Product recommendations
2. Map Reduce and Data Parallelism
    * Originally thought to be used to speed up training over very large datasets by distributing portions of the dataset to workers, and then centralize and reduce the results in a master server.
    * May speed up to $N$ times the training, where $N$ is the amount of workers.
    * The requirement is to be able to reduce somehow your training in the computing, such as a summation.
    * Also applicable to multi-core CPU, or even multi GPU in only one server.


---
# Week11 - Application Example: Photo OCR
## Photo OCR
1. Problem Description and Pipeline
    * Photo Optical Character Recognition (OCR) is an application that gives extra information about pictures
    * Application examples:
        * Searching images
        * Blind people assistance
        * Car navigation systems
    * Pipeline (usually different steps are developed independently)
        1. Text Detection $\to$ find regions where there is text in it
        2. Character Semgentation $\to$ split the ROI into individual characters
        3. Character Classification $\to$ recognize each character
        4. Spelling Correction System $\to$ fix missrecognized character by context
2. Sliding Windows
    * The *Text Detection* algorithm is based in running a binary classifier (presence of text) as a sliding window on the original image, using different sizes and aspect ratios
        * The output is a map of probability of text presence detected, with the stride of the sliding window as the new resolution
        * Apply a dilation of this output map (expansion of detections)
        * Detect which blobs are most probable to be text (use aspect ratio, size, etc.)
    * The *Character Semgentation* may also be implemented as a binary classifier by a 1D sliding window algorithm, to classify as positive centered divisions between 2 consecutive characters on the outputs of the *Text Detection* system.
3. Getting Lots of Data and Artificial Data
    * Data Synthesis / Data Augmentation
        * From scratch
        * From actual examples by some distortion
    * Adding random noise doesn't seem to help much (why?)
    * Why don't we get more data? Is it worth it?
        1. Make sure you have a low bias model first
        2. If variance is high and you decide you need more data, estimate the time taken and compare for:
            1. Collect more data
            2. Design a data augmentation system
            3. Crowd source (e.g. Amazon Mechanical Turk) $\to$ hire a 3rd party service to label data for you
4. Ceiling Analysis: What Part of the Pipeline to Work on Next
    * Spend time to design metrics of the overall system performance
        1. Start making the first block simulating perfect performance, and measure the overall performance
        2. Go on and make the next blocks, one by one, simulate perfect performance and record the overall performance for each (cummulative) simulation
        3. Place the overall performances in a table and compute the improvements on each step
        4. Compare and choose which block to work on, knowing what is the ceil of the overall improvement that can be made working on that specific block
## Conclusion
1. Summary and Thank You
    * Toppics summary
        * Supervised Learning
            * Linear Regression
            * Logistic Regression
            * NNs
            * SVMs
        * Unsupervised Learning
            * K-means
            * PCA
            * Anomaly Detection
        * Special Applications
            * Recommender Systems
            * Large Scale ML
        * Advice on ML Systems
            * Bias/variance
            * Regularization
            * Deciding what to work on next
            * Evaluation Metrics
            * Learning Curves
            * Error Analysis
            * Ceiling Analysis
    * Thank you Andrew! :)