## Portfolio Optimization
- This codebase focuses on mean-variance optimization. 
- We attempt to minimize risk (variance) subject to a constraint on the expected return. Conversely, we can maximize expected returns subject to a constraint on the variance. 
- Combining the above problems, we get the Markowitz Mean Variance Optimization problem:

$$ \text{minimize} \ -\mathbf{w}^T \Sigma \mathbf{w} + \lambda \mu^T \mathbf{w}$$

Subject to the following constraints:
  1. $\mu^T \mathbf{w} = \mu_t$
  2. $\sum_{i=1}^{N} w_i = 1$
  3. $w_j \geq 0 \ \forall \ j$

Where:
 - $\lambda$ is the risk aversion factor
 - $\mathbf{w}$ is the vector of weights
 - $\mu$ is the vector of expected returns
 - $\Sigma$ is the covariance matrix of returns


### Calculating Returns
- Expected returns are defined either through historical returns or using the Fama-French 3 factor model found in [fama_french.py](code/fama_french.py).


![Flow Chart](img/Concept%20map.png)