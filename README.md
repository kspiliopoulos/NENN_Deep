# NENN_Deep
Normalization Effects on Deep Neural Networks

Authors of this repository are Konstantinos Spiliopoulos and Jiahui Yu.

This repository contains code supporting the article

Konstantinos Spiliopoulos and Jiahui Yu, "Normalization effects on deep neural networks", 2022, https://arxiv.org/abs/2209.01018.

ArXiv preprint: https://arxiv.org/abs/2209.01018.

First read the Read Me file to run the code.

To report bugs encountered in running the code, please contact Konstantinos Spiliopoulos at kspiliop@bu.edu or Jiahui Yu at jyu32@bu.edu


# Short exposition--Achieving good accuracy with less need for fine tuning. 

Consider for example the three layer neural network (conclusions are similar for DNNs of arbitrary depth, see [9])

$$
\begin{align}
g_{\theta}^{N_1,N_2,N_3}(x) = \frac{1}{N_3^{\gamma_3}} \sum_{i=1}^{N_3} C^i \sigma\left(\frac{1}{N_2^{\gamma_2}}\sum_{j=1}^{N_2}W^{3,i,j}\sigma\left(\frac{1}{N_1^{\gamma_1}}\sum_{\nu=1}^{N_1} W^{2,j,\nu}\sigma(W^{1,\nu}x)\right)\right),
\end{align}
$$

where $\gamma_{1},\gamma_{2},\gamma_{3}\in[1/2,1]$ and
$\theta = (C^i, W^{1,\nu}, W^{2,j,\nu}, W^{3,i,j})_{i\leq N_3, j\leq N_2, \nu\leq N_1} $
are the parameters to be learned. Here the two extreme cases are $\gamma_1=\gamma_2=\gamma_3=1/2$
that corresponds to the popular Xavier initialization, see [2], and $\gamma_1=\gamma_2=\gamma_3=1$ corresponds to the mean-field normalization, see [1,3,4,5,6,7]. 

Let the objective function be

$$
\begin{align}
\mathcal{L}(\theta) &= \frac{1}{2} E_{X,Y} (Y-g_{\theta}^{N_1,N_2,N_3}(x))^2.
\end{align}
$$

We learn the parameters $\theta$ using stochastic gradient descent 

$$
\begin{align}
C^i_{k+1} &= C^i_k + \frac{\alpha_c^{N_1,N_2,N_3}}{N_3^{\gamma_3}} \left(y_k - g_k^{N_1,N_2,N_3}(x_k)\right)H^{3,i}_k(x_k),\\
W^{1,\nu}_{k+1} &= W^{1,\nu}_k + \frac{\alpha_1^{N_1,N_2,N_3}}{N_1^{\gamma_1}}\left(y_k - g_k^{N_1,N_2,N_3}(x_k)\right)\left(\frac{1}{N_3^{\gamma_3}}\sum_{i=1}^{N_3}C^i_k\sigma'(Z^{3,i}_k(x_k))\left(\frac{1}{N_2^{\gamma_2}}\sum_{j=1}^{N_2} W^{3,i,j}_k\sigma'(Z^{2,j}(x_k))W^{2,j,\nu}_k\right)\right)\\
&\qquad \qquad \times \sigma'(W^{1,\nu}_k x_k)x_k,\\
W^{2,i,\nu}_{k+1} &= W^{2,i,\nu}_k + \frac{\alpha_2^{N_1,N_2,N_3}}{N_1^{\gamma_1}N_2^{\gamma_2}}\left(y_k - g_k^{N_1,N_2,N_3}(x_k)\right)\frac{1}{N_3^{\gamma_3}}\sum_{i=1}^{N_3}C^i_k \sigma'(Z^{3,i}_k(x_k))W^{3,i,j}_k\sigma'(Z^{2,j}_k(x_k))H^{1,\nu}_k(x_k),\\
W^{3,i,j}_{k+1} &= W^{3,i,j}_k + \frac{\alpha_3^{N_1,N_2,N_3}}{N_2^{\gamma_2}N_3^{\gamma_3}}\left(y_k - g_k^{N_1,N_2,N_3}(x_k)\right)C^i_k \sigma'(Z^{3,i}_k(x_k))H^{2,j}_k(x_k),
\end{align}
$$

where

$$
\begin{align}
H^{1,\nu}_k(x) &= \sigma(W^{1,\nu}_k x),\\
Z^{2,j}_k(x) &= \frac{1}{N_1^{\gamma_1}}\sum^{N_1}_{\nu=1} W^{2,j,\nu}_k H^{1,\nu}_k(x),\\
H^{2,j}_k(x) &= \sigma(Z^{2,j}_k(x)),\\
Z^{3,i}_k(x) &= \frac{1}{N_2^{\gamma_2}} \sum^{N_2}_{j=1} W^{3,i,j}_k H^{2,j}_k(X_k),\\
H^{3,i}_k(x) &= \sigma(Z^{3,i}_k(x)),\\
g_k^{N_1,N_2,N_3}(x) &= g^{N_1,N_2,N_3}_{\theta_k}(x) = \frac{1}{N_3^{\gamma_3}} \sum^{N_3}_{i=1} C^i_k H^{3,i}_k(x)
\end{align}
$$

and

$$
\begin{align}
\alpha_c^{N_1,N_2,N_3}, \alpha_1^{N_1,N_2,N_3}, \alpha_2^{N_1,N_2,N_3}, \alpha_3^{N_1,N_2,N_3}
\end{align}
$$

are the learning rates. The choice of the learning rate is theoretically linked to the $\gamma_i$ parameters. In particular, the theory developed in [9] suggests that for this case and in order to gaurantee ***statistical robustness*** of the neural network as $N_1,N_2,N_3$ increase to infinity, the learning rate should be chosen to be of the order of

$$
\begin{align}
&\alpha_c^{N_1,N_2,N_3} = \frac{1}{N_3^{2-2\gamma_3}}, &&\quad \alpha_1^{N_1,N_2,N_3} = \frac{1}{N_1^{1-2\gamma_1}N_2^{2-2\gamma_2}N_3^{3-2\gamma_3}},\\
&\alpha_2^{N_1,N_2,N_3} = \frac{1}{N_1^{1-2\gamma_1}N_2^{1-2\gamma_2}N_3^{3-2\gamma_3}}, &&\quad \alpha_3^{N_1,N_2,N_3} = \frac{1}{N_2^{1-2\gamma_2}N_3^{2-2\gamma_3}}
\end{align}
$$


The goal of the paper [9] is to study  the performance of neural networks scaled by $1/N_{i}^{\gamma_{i}}$
with $\gamma_{i} \in [1/2, 1]$. 

The theoretical results of [9] derive an asymptotic expansion of the neural network's output with respect to $N_{3}$ in such a way that the variance of the network is kept bounded even as $N_1,N_2,N_3$ increase to infinity. 
This expansion demonstrates the effect of the choice of  $\gamma_{i}$   on bias and variance.
In particular, for large and fixed $N_{i}$,
the variance goes down monotonically as the $\gamma_{i}$ (and specifically as the outer layer normalization $\gamma_3$) increases to $1$.

The numerical results of [9], done on MNIST [10], demonstrate that train and test accuracy monotonically increase as all the $\gamma_i$ increases to $1$. 

The conclusion is that being close to the mean-field normalization  $\gamma_1=\gamma_2=\gamma_3=1$  is clearly the optimal choice!! But, for this to be realized the learning rate has to be chosen appropriately based on a theoretically informed choice as demonstrated above. 

The paper [9] has the formulas for the learning rates for deep neural networks of arbitrary depth.

![plot_mnist_mlp_gIII07_hI100_hII100_hIII100_e1500_b20_test](https://user-images.githubusercontent.com/106413949/191279543-bdc0f43f-e09a-4324-add5-9e8fd2a6b14a.png)

![plot_mnist_mlp_gIII10_hI100_hII100_hIII100_e1500_b20_test](https://user-images.githubusercontent.com/106413949/191278660-5e5a2b62-c9cb-4942-bb10-32c70496325c.png)

Performance of scaled neural networks on MNIST test dataset: cross entropy loss, $N_1=N_2=N_3=100$, batch size $=20$, Number of Epoch $=1500$. Each subfigure plots various $\gamma_1, \gamma_2$ for a fixed $\gamma_3$.


***No further parameter tuning was done in these examples, just plug in the mathematically informed formulas for the learning rates, choose*** $\gamma_{1}, \gamma_{2}, \gamma_{3}$ ***and that's it! Of course, further parameter tuning will improve the accuracy more but the point is that correct choice of the learning rates and correct normalization immediately give great results!***

**References**

[1] L. Chizat, and F. Bach. On the global convergence of gradient descent for over-parameterized models
using optimal transport. Advances in Neural Information Processing Systems (NeurIPS). pp. 3040-3050,
2018.

[2] X. Glorot and Y. Bengio. Understanding the diffculty of training deep feedforward neural networks.
Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249-
256. 2010.

[3] S. Mei, A. Montanari, and P. Nguyen. A mean field view of the landscape of two-layer neural networks
Proceedings of the National Academy of Sciences, 115 (33) E7665-E767, 2018.

[4] G. M. Rotskoff and E. Vanden-Eijnden. Neural Networks as Interacting Particle Systems: Asymptotic
Convexity of the Loss Landscape and Universal Scaling of the Approximation Error. arXiv:1805.00915,
2018.

[5] J. Sirignano and K. Spiliopoulos. Mean Field Analysis of Neural Networks: a law of large numbers.
SIAM Journal on Applied Mathematics, 80(2), 725-752, 2020.

[6] J. Sirignano and K. Spiliopoulos. Mean Field Analysis of Neural Networks: A Central Limit Theorem.
Stochastic Processes and their Applications, 130(3), 1820-1852, 2020.

[7] J. Sirignano and K. Spiliopoulos. Mean Field Analysis of Deep Neural Networks. Mathematics of
Operations Research, 47(1):120-152, 2021.

[8] K. Spiliopoulos and Jiahui Yu, Normalization effects on shallow neural networks and related asymptotic expansions, 2021, AIMS Journal on Foundations of Data Science, June 2021, Vol. 3, Issue 2, pp. 151-200.

[9] K. Spiliopoulos and Jiahui Yu, Normalization effects on deep neural networks, arXiv: https://arxiv.org/abs/2209.01018, 2022.

[10] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition.
Proceedings of the IEEE, 86(11):2278-2324, 1998.


