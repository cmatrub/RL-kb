## Q: importance sampling. explain this
This slide is walking through how to turn the (on-policy) policy gradient into an **off-policy** one using importance sampling, and why the naïve version breaks down. Let me build it up.

## The core trick: importance sampling

Importance sampling estimates an expectation under one distribution $p$ using samples drawn from a *different* distribution $q$:

$$\mathbb{E}_{x\sim p}[f(x)] = \mathbb{E}_{x\sim q}\left[\frac{p(x)}{q(x)}f(x)\right]$$

You draw from the "wrong" distribution $q$, then reweight each sample by the ratio $p(x)/q(x)$ to correct for the mismatch. That ratio is the *importance weight*.

## Why we want this here

Policy gradient is on-policy: to estimate $\nabla J(\theta')$ you need trajectories sampled from $\pi_{\theta'}$ itself. Every time you update the parameters, your old data becomes stale. That's wasteful. The goal here is to update the new policy $\pi_{\theta'}$ while reusing samples collected under the old policy $\pi_\theta$.

## Applying it to whole trajectories (first equation)

The objective is $J(\theta') = \mathbb{E}_{\tau\sim p_{\theta'}(\tau)}[r(\tau)]$, but we only have trajectories from $p_\theta(\tau)$. So we importance-sample with weight $p_{\theta'}(\tau)/p_\theta(\tau)$.

The probability of a trajectory factors as $p_\theta(\tau) = p(s_1)\prod_t \pi_\theta(a_t|s_t)\, p(s_{t+1}|s_t,a_t)$. The initial-state distribution and the dynamics $p(s_{t+1}|s_t,a_t)$ **don't depend on the policy parameters**, so they cancel in the ratio, leaving just the product of action-probability ratios:

$$\frac{p_{\theta'}(\tau)}{p_\theta(\tau)} = \prod_{t=1}^{T}\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)}$$

That's the green-highlighted term in the top equation.

## The problem (first arrow)

This weight is a **product of $T$ terms**. Even if each individual ratio is close to 1, multiplying many of them compounds:

- If the ratios sit slightly below 1, the product decays exponentially toward 0 → **vanishes**.
- If slightly above 1, it grows exponentially → **explodes**.

So for any long horizon $T$, the importance weight has enormous variance and the estimate becomes useless. This is the classic failure mode of importance sampling over trajectories.

## The fix attempt (second equation): expectation over timesteps

Instead of treating a whole trajectory as one sample, treat each individual $(s_{i,t}, a_{i,t})$ as a sample. Now the importance weight only needs to correct for the *marginal* state–action distribution at a single timestep, rather than a product over the whole trajectory:

$$\frac{\pi_{\theta'}(s_{i,t}, a_{i,t})}{\pi_\theta(s_{i,t}, a_{i,t})}$$

This is a **single ratio per timestep** — no product over $T$, so no exponential compounding. Much less likely to blow up or die.

## The catch (second arrow) and the "≈ 1" note

That per-timestep joint ratio factors into two pieces:

$$\underbrace{\frac{p_{\theta'}(s_{i,t})}{p_\theta(s_{i,t})}}_{\text{state marginal ratio}} \cdot \underbrace{\frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}{\pi_\theta(a_{i,t}|s_{i,t})}}_{\text{action ratio}}$$

The action ratio is easy — you can evaluate both policies on the sampled action. But the **state-marginal ratio** is the hard part: it asks "how much more likely is the new policy to *visit* this state?", which requires the full state-visitation distribution under each policy. That depends on rolling the policy through the dynamics and is generally intractable to compute. That's the "hard to measure" the bottom-right note refers to.

The practical move — the "**often approximated as 1**" — is to simply drop that state-marginal ratio, assuming the two policies visit states similarly. This is reasonable *when the policies are close*. What's left is just the action ratio:

$$\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)}$$

This is exactly the surrogate objective at the heart of **TRPO and PPO**, where you keep this single per-timestep ratio and then constrain or clip it so the two policies can't drift too far apart (which is what keeps the approximation valid and the variance controlled).

(The $b$ throughout is just the usual variance-reduction baseline subtracted from the return; it doesn't interact with the importance-sampling story.)

So the arc of the slide is: trajectory-level IS is exact but explodes/vanishes → per-timestep IS is stable but needs an intractable state-distribution ratio → approximate that ratio as 1 → land on the PPO/TRPO-style objective.

## Q: but not all terms in the product are going to be above 1 or below 1 all at once, some above and some below, no? so why do you conclude vanishing or exploding ratio?
You're right to push on this, and my earlier phrasing was sloppy. In a real trajectory the ratios *are* mixed — some above 1, some below — so "they all drift one way" is wrong. The actual reason is subtler, and it's worth getting right.

The key move is to take logs. The product becomes a **sum**:

$$\log \prod_{t=1}^{T}\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} = \sum_{t=1}^{T}\log\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)}$$

So now it's a random walk of $T$ steps, and your intuition ("positives and negatives cancel") is really a claim that this sum stays near 0. Two facts about that sum break the cancellation.

**The sum has negative drift.** The actions are sampled from the *old* policy $\pi_\theta$. The expected value of one log-ratio term under that sampling is

$$\mathbb{E}_{a\sim\pi_\theta}\left[\log\frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)}\right] = -D_{\mathrm{KL}}\big(\pi_\theta \,\|\, \pi_{\theta'}\big) \le 0.$$

It's a KL divergence, so it's negative whenever the two policies differ at all. Each step of the walk has negative expected drift, so the sum trends toward $-T \cdot \overline{D}_{\mathrm{KL}}$, i.e. linearly downward in $T$. Exponentiating, the *typical* (median/geometric-mean) weight shrinks toward zero. That's the vanishing — and it happens even though individual terms are mixed, because the down-steps are, on average, slightly more common or slightly larger. This is just the statement that the geometric mean sits below the arithmetic mean.

**But the mean weight is exactly 1.** You can check directly:

$$\mathbb{E}_{a\sim\pi_\theta}\left[\frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)}\right] = \sum_a \pi_\theta(a|s)\,\frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)} = \sum_a \pi_{\theta'}(a|s) = 1.$$

So the average never moves. If the typical value collapses to ~0 but the average stays pinned at 1, the only way to reconcile that is a **thin tail of enormous weights**: a tiny fraction of trajectories carry astronomically large importance weights that single-handedly hold the mean up. That's the exploding — not most trajectories, just rare ones, but they dominate any finite-sample estimate.

Quantitatively, the variance of the sum of log-ratios grows like $T$, so the log-weight's spread grows like $\sqrt{T}$, and after exponentiating the variance of the weight itself grows roughly like $(1+\sigma^2)^T$ — exponential in the horizon.

So "vanish or explode" isn't the whole product going uniformly one direction. It's both at once: the **typical** trajectory's weight vanishes, while the **variance** explodes because a handful of rare trajectories get gigantic weights.

Why doesn't the mixing save you? Because multiplicative noise doesn't self-cancel the way additive noise does. Take a symmetric toy case: each ratio is $2$ or $\tfrac12$ with equal probability — perfectly "balanced," geometric mean exactly 1. The product after $T$ steps is $2^{(U-D)}$, where $U-D$ is a mean-zero random walk that wanders by $\pm\sqrt{T}$. So the weight is $2^{\pm\sqrt{T}}$ — sometimes vast, sometimes minuscule — and its arithmetic mean is $1.25^T$, which blows up. Even with textbook-balanced up/down terms, the distribution spreads out catastrophically. The cancellation you'd need requires the log-sum to stay pinned near 0, and both the drift and the $\sqrt T$ fluctuations prevent that.

That's exactly why the per-timestep formulation on the slide helps: replacing the length-$T$ product with a single ratio removes the multiplicative compounding entirely, so there's no random walk to run away.

## Q: can you just randomly switch from trajectory sampling to time step sampling. or is there some assumption here that the ratios corresponding to all other time steps in the trajectory are assumed ~1?
Great question — and the honest answer is that no, you can't *just* switch; but the switch that's being made is legitimate, and your "other timesteps ≈ 1" intuition turns out to be *exactly* the right way to see what's really being assumed. Let me connect the two.

## Q: The switch isn't arbitrary — it comes from a different (but exact) way of writing $J$

The first equation writes the objective as an expectation over **whole trajectories**. The second one comes from writing the *same* objective as an expectation over **state–action marginals**, one timestep at a time:

$$J(\theta') = \sum_{t} \mathbb{E}_{(s_t,a_t)\sim p_{\theta'}(s_t,a_t)}\big[r(s_t,a_t)\big]$$

Now importance-sample that marginal using samples from $p_\theta$:

$$= \sum_{t}\mathbb{E}_{(s_t,a_t)\sim p_\theta(s_t,a_t)}\left[\frac{p_{\theta'}(s_t,a_t)}{p_\theta(s_t,a_t)}\,r(s_t,a_t)\right]$$

This is still **exact**. Nothing has been dropped yet. The single marginal weight $\frac{p_{\theta'}(s_t,a_t)}{p_\theta(s_t,a_t)}$ factors cleanly into a state part and an action part:

$$\frac{p_{\theta'}(s_t,a_t)}{p_\theta(s_t,a_t)} = \underbrace{\frac{p_{\theta'}(s_t)}{p_\theta(s_t)}}_{\text{state visitation}}\cdot\underbrace{\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)}}_{\text{action}}$$

## Q: The exact marginal weight and the trajectory product are the same quantity

Here's the part that answers you directly. Where did the length-$T$ product go? It didn't vanish — it's *hiding inside the state ratio*.

Recall that the probability of reaching state $s_t$ differs between the two policies only through the action choices that led there (the dynamics $p(s_{t'+1}|s_{t'},a_{t'})$ cancel, as before). So for a sampled trajectory, the product of ratios over the **leading-up** timesteps

$$\prod_{t'=1}^{t-1}\frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_\theta(a_{t'}|s_{t'})}$$

*is* the sample-level realization of the state-visitation ratio $\frac{p_{\theta'}(s_t)}{p_\theta(s_t)}$. In other words:

$$\underbrace{\prod_{t'=1}^{t}\frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_\theta(a_{t'}|s_{t'})}}_{\text{trajectory product up to }t} \;\longleftrightarrow\; \underbrace{\frac{p_{\theta'}(s_t)}{p_\theta(s_t)}}_{\text{all the past ratios}}\cdot\underbrace{\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)}}_{\text{the ratio at }t}$$

So switching to the marginal form loses *nothing*. The trajectory product and "state-ratio × action-ratio" are literally the same thing written two ways. That's why it's a legitimate re-derivation, not sleight of hand.

## Q: Where the assumption actually enters — and it's exactly what you guessed

The approximation is the **last** step: setting the state-visitation ratio to 1,

$$\frac{p_{\theta'}(s_t)}{p_\theta(s_t)} \approx 1,$$

which leaves only the single action ratio $\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)}$ — the "often approximated as 1" note on the slide.

And given the equivalence above, dropping that state ratio is **precisely** setting the product of all the *other* (leading-up) timesteps' ratios equal to 1. So your instinct is dead on. It's not that each individual other-timestep ratio is assumed to be 1 — it's that their **cumulative product** is assumed to be ≈ 1, i.e. the two policies wander into the same states with nearly the same frequency.

(Strictly, the fully-correct estimator also carries importance weights on the future actions inside the reward-to-go term; those get dropped in the same spirit. Same phenomenon, future side instead of past side.)

## When is that legitimate?

Exactly when $\pi_{\theta'} \approx \pi_\theta$. If the new and old policies are close, the state-visitation distributions they induce are close, so their ratio really is near 1 and throwing it away costs little. If the policies drift far apart, the assumption is bad — the product of those "other" ratios is precisely the runaway quantity from the previous discussion.

This is the whole reason TRPO and PPO exist: they keep only the single per-timestep action ratio (having discarded the state ratio), and then *enforce* $\pi_{\theta'} \approx \pi_\theta$ — via a KL trust region (TRPO) or ratio clipping (PPO) — so that the "product of the other timesteps ≈ 1" assumption you spotted is actually kept true, rather than just hoped for.

## Q: you mean the future step ratios will come in when we calculate the cum future reward term?
Yes — exactly. Let me make the structure explicit, because it splits the importance weights into three neat groups and your question is about the third.

Start from the causal (reward-to-go) policy gradient and importance-sample it. The fully-correct off-policy estimator, after applying causality, is:

$$\nabla_{\theta'} J(\theta') = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=1}^{T}\nabla_{\theta'}\log\pi_{\theta'}(a_t|s_t)\sum_{t'=t}^{T}\left(\prod_{t''=1}^{t'}\frac{\pi_{\theta'}(a_{t''}|s_{t''})}{\pi_\theta(a_{t''}|s_{t''})}\right)r(s_{t'},a_{t'})\right]$$

Look at the weight attached to a single future reward $r(s_{t'}, a_{t'})$ with $t' \ge t$. It's a product of ratios running from step $1$ all the way up to $t'$. Split that product at the current step $t$:

$$\prod_{t''=1}^{t'}\frac{\pi_{\theta'}}{\pi_\theta} = \underbrace{\prod_{t''=1}^{t}\frac{\pi_{\theta'}}{\pi_\theta}}_{\text{past + present}}\cdot\underbrace{\prod_{t''=t+1}^{t'}\frac{\pi_{\theta'}}{\pi_\theta}}_{\text{future}}$$

So the three groups are:

- **Past ratios** ($1$ to $t-1$): correct for *how you arrived* at $s_t$. This is the sample-level realization of the state-visitation ratio $p_{\theta'}(s_t)/p_\theta(s_t)$ — the thing from the last message.
- **Present ratio** (at $t$): the single action ratio $\pi_{\theta'}(a_t|s_t)/\pi_\theta(a_t|s_t)$ — the one PPO/TRPO keep.
- **Future ratios** ($t+1$ to $t'$): these are the ones you're asking about. They ride *inside the reward-to-go term*, one growing product per future reward, because each reward $r(s_{t'},a_{t'})$ was collected along a path whose actions were drawn from $\pi_\theta$. To evaluate that reward-to-go's expectation under $\pi_{\theta'}$, you have to reweight the future actions that led to it.

So your reading is right: the future step ratios enter precisely through the cumulative future reward. Each term $r(s_{t'},a_{t'})$ in the reward-to-go carries its own product of action ratios from $t{+}1$ up to $t'$.

One useful detail on why the product for a reward at $t'$ stops *at* $t'$ and not at $T$: actions taken *after* a reward can't causally affect it, so their ratios are independent of that reward and each has expectation exactly 1 under $\pi_\theta$ (that's the $\sum_a \pi_{\theta'}(a|s) = 1$ identity from before). That's what lets us truncate the product at $t'$ — otherwise every reward would carry the full length-$T$ product.

And the punchline connecting to the slide: the practical estimator drops the past product (state ratio ≈ 1) **and** the future products (reward-to-go ratios ≈ 1), keeping only the present action ratio. Both approximations are the same bet — that $\pi_{\theta'}\approx\pi_\theta$, so any product of "other-timestep" ratios, whether reaching backward to $s_t$ or forward through the reward-to-go, stays near 1. TRPO/PPO then enforce that closeness rather than just assuming it.

## Q: wait, you just proved in that prior discussion that we cannot make this assumption and now you are contradicting yourself?
Not a contradiction — but I see exactly why it reads like one, and it's a fair hit. Let me pin down what the two claims actually are, because they're about *different regimes*.

The earlier "runaway" argument and the later "≈ 1" approximation both talk about the same object — a product of importance ratios over timesteps — but under two different conditions:

**Earlier claim:** *If the two policies are arbitrarily far apart, the length-$T$ product explodes/vanishes.* This is why naïve off-policy policy gradient over full trajectories fails. The product runs away because nothing constrains $\pi_{\theta'}$ from being wildly different from $\pi_\theta$.

**Later claim:** *If the two policies are close, the product of the "other" ratios is ≈ 1, so we can drop it.* This is what PPO/TRPO exploit.

These aren't in tension — they're the two ends of the same knob. The product's behavior depends entirely on how far $\pi_{\theta'}$ has drifted from $\pi_\theta$:

- Policies far apart → each ratio deviates a lot from 1 → the product compounds into the runaway. **Assumption invalid.**
- Policies close → each ratio sits near 1 → their product stays near 1. **Assumption valid.**

So dropping the state/future ratios is **not** an unconditional move. It is *only* licensed when $\pi_{\theta'}\approx\pi_\theta$. The earlier discussion showed what happens when that condition is violated; the later discussion assumes the condition holds. Both are true statements about the same product, conditioned on the closeness of the policies.

Here's the part that I think resolves the apparent contradiction fully: **the approximation would be illegitimate on its own** — you're right that you can't just wave the runaway product away by fiat. What makes it legitimate is that TRPO and PPO don't merely *hope* the policies stay close; they **enforce** it. TRPO adds a hard KL trust-region constraint, $D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta'}) \le \delta$; PPO clips the action ratio so gradients vanish once it strays outside $[1-\epsilon, 1+\epsilon]$. Both are machinery whose entire job is to keep $\pi_{\theta'}\approx\pi_\theta$ within each update — which is precisely the condition that turns the runaway product into a ≈ 1 product you're allowed to drop.

So the logical structure is:

1. The product runs away *in general* (earlier). ✔
2. Therefore you may **not** drop it *in general*. ✔ (your objection, and it's correct)
3. So we *force* the policies to stay close, making the product ≈ 1 *by construction*. ✔
4. *Now* — and only now — dropping it is justified. ✔

The approximation isn't a claim that the runaway problem doesn't exist; it's a claim that we've **spent effort to stay in the regime where the runaway hasn't kicked in yet**. Without the trust region or clipping, your objection stands and the approximation is unjustified. That's not a footnote — it's the whole reason those two algorithms are built the way they are.
