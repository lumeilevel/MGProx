# MGProx
Reproduction for 2302.04077 which aims to minimize the sum of a $\mu_0$-strongly convex, $L_0$-smooth $f_0$ and a convex, nonsmooth, lower semi-continuous and seperable $g_0$.
$$
x^*=\mathop{\arg\min}_{x\in\R^n}\ F_0(x):=f_0(x)+g_0(x)\tag{1}
$$
proximal mapping:
$$
x_{k+1}=\mathop{\arg\min}_u\ \frac 1 2\Vert u-x_k\Vert_2^2+g_0(u)\tag{2}
$$


 ## EOP

Elastic Obstacle Problem(EOP) describes the shape of an elastic membrane covering an obstacle $\phi$. We discretize a 2-dimensional shifted aEOP on truncated sine wave.
$$
\min_x \frac 1 2 \Braket{Q_0x,x}+\Braket{p_0,x}+i_+(x)\label{eq1}\tag{3}
$$
proximal mapping:
$$
\begin{aligned}
x_{k+1}&=\mathop{\arg\min}_u\ \frac 1 2\Vert u-x_k\Vert_2^2+i_+(u)\\
&=\max(x_k,0)
\end{aligned}\tag{4}
$$

## LASSO

problem
$$
\min_x\ \frac12\Vert Ax-b\Vert_2^2+\lambda\Vert x\Vert_1\tag{5}
$$
proximal mapping
$$
\begin{aligned}
x_{k+1}&=\mathop{\arg\min}_u\ \frac 1 2\Vert u-x_k\Vert_2^2+\lambda\Vert u\Vert_1\\
&=\operatorname{sign}(x_k)\odot\max(|x_k|-\lambda,0)
\end{aligned}\tag{6}
$$


