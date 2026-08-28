# Mathematical background

For a linear parameterization,

`F_theta(s, x) = theta^T phi(s, x)`,

an observed optimizer generates the inequalities

`theta^T [phi(s_i, x_hat_i) - phi(s_i, x)] <= 0` for every feasible alternative `x`.

The normalized incenter maximizes the minimum normalized slack over a bounded parameter space.
The Augmented Suboptimality Loss is

`max_x theta^T[phi(s, x_hat) - phi(s, x)] + d(x_hat, x)`.

The package stores the maximizing alternative because it supplies both the loss value and a
subgradient. All geometry and loss calculations use minimization as the standard convention.

