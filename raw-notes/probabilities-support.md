## Q: what is support of a distribution
The support of a distribution is the set of values where its probability (or density) is greater than zero. It's everywhere the random variable can actually land.

Examples:

Uniform(0, θ): The support is [0, θ], and it changes as θ changes.
Gaussian: The support is all of ℝ, whatever the mean and variance.
Softmax policy: The support is every action, since each one gets probability > 0.

So when I said "the support moves with θ," I meant that changing θ changes which values are possible at all, not just how likely they are.