# Neural Network with InnerProduct Deployment for CTR Prediction

## background 
CTR prediction is a very common task for ad evaluation and analysis.
This program provides a simple neural network with inner product layer proposed by Y. Qu in the paper [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144)

## Neural Netword Structure
<p align="center">
<img src=".\images\NN_Structure.jpg" style="width:60%; border:0;">
</p>

## Formulas in the Inner Product Layer
$l_1 \in \mathbb R^{D_1}$ is the output of the inner product layer, where $D_1$ is the dimension of the layer. The formulation of **$l_1$** is $$l_1 = relu(l_z + l_p + b_1)$$
herein, with $l_z$ the linear signals, $l_p$ the quadratic signals and $b_1$ the bias. \
The linear signals can be obtained by below. $$l_z^n = W_z^n\odot z = \displaystyle\sum_{i=1}^N \displaystyle\sum_{j=1}^M W_{z_{i, j}}^n z_{i,j}$$
Quadratic signals can be obtained by below. $$l_p^n = W_p^n\odot p = \displaystyle\sum_{i-1}^N\displaystyle\sum_{j=1}^M\theta_i^n \theta_j^n \langle f_i, f_j\rangle = \langle \displaystyle\sum_{i=1}^N\delta_i^n,\displaystyle\sum_{i=1}^N\delta_i^n \rangle$$

## Evaluation
Evaluation is omitted as the paper approves that the AUC performs better than those without product layer.

## input data
Train and target data are provided in the data folder.
They are preprocessed data in percentage of display percentage and CTR for a certain category.

This program has a MIT license.
