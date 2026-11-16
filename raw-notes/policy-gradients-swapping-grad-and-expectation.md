# CS224R Policy Gradients – Q&A Notes

Source: CS224R Lecture 3 slides (Policy Gradients, 2026), slides 9 and 24.

> Note: Some earlier answers were later corrected or clarified (see Q11–Q14). The corrected summary is at the end.

---

## Q1. Is the per-timestep off-policy equation (slide 24) TD, since it uses timesteps instead of whole trajectories?

No, it's still Monte Carlo, not TD. "Over timesteps" on this slide only changes where the importance weight goes. It doesn't change how the return is estimated.

**What makes something TD:** TD methods bootstrap. They build the target from a learned value estimate, like r(s_t, a_t) + γV(s_{t+1}), instead of waiting for the real rewards.

**What this equation does:** The return term, Σ_{t'=t}^{T} r(s_{i,t'}, a_{i,t'}) − b, adds up the actual rewards from step t to the end of the episode. That's a reward-to-go, which is a Monte Carlo return. There's no learned value function standing in for future rewards, so you still need the full trajectory before you can compute the gradient.

**What "over timesteps" actually changes:** The first equation weights each trajectory by a product of ratios, ∏_{t=1}^{T} π_θ′/π_θ. Multiplying T ratios together makes that weight explode or vanish when T is large. The rewrite treats each (s_t, a_t) pair as a sample and gives it its own ratio, π_θ′(s,a)/π_θ(s,a). That ratio splits into a state part and an action part. The state part is hard to compute, so it's approximated as 1, which leaves the per-step action ratio in the "common final form." So the change is only about how to do importance sampling without the long product. This is the approach behind methods like PPO.

**Where TD would come in:** If you replaced that reward sum with a learned critic, such as a Q-value estimate or a TD-error advantage like r + γV(s′) − V(s), you'd get an actor-critic method. That's when bootstrapping, the TD idea, enters policy gradients.

---

## Q2. Slide 9's last equation uses ∇∫ = ∫∇. Is this always true?

Line 9 swaps the gradient and the integral, going from ∇_θ ∫ p_θ(τ) r(τ) dτ to ∫ ∇_θ p_θ(τ) r(τ) dτ. This is called differentiating under the integral sign (the Leibniz integral rule). It isn't always true, but it holds under mild conditions.

**When the swap is allowed** (from the dominated convergence theorem):

1. **Fixed domain:** The region you integrate over can't depend on θ. If it does, the full Leibniz rule adds extra boundary terms.
2. **Differentiable in θ:** p_θ(τ) r(τ) has to be differentiable in θ for (almost) every τ.
3. **Bounded derivative:** There has to be an integrable g(τ) with |∇_θ p_θ(τ) r(τ)| ≤ g(τ) for all θ in a neighborhood.

**An example where it fails:** Take p_θ(x) = Uniform(0, θ) and r(x) = 1. Then J(θ) = 1 for every θ, so the true gradient is 0. If you swap anyway, you get ∫₀^θ ∇_θ(1/θ) dx = −1/θ, which is wrong. The swap missed the boundary term.

**Why it's fine in this RL setting:**

- **Fixed support:** The policy is stochastic and differentiable (softmax, Gaussian), so the set of possible trajectories doesn't move with θ. The identity ∇ log p = ∇p / p also needs p > 0 wherever ∇p ≠ 0.
- **Only the policy depends on θ:** The dynamics and initial state distribution don't, so ∇_θ log p_θ(τ) = Σ_t ∇_θ log π_θ(a_t|s_t).
- **Bounded integrand:** Bounded rewards with a finite horizon (or discounting) satisfy the domination condition.

**What this buys you:** The reward r(τ) never needs to be differentiable, so the method works with black-box environments.

**When it can break in RL:** deterministic policies (point masses, which is why DPG uses a different derivation), support that depends on θ or collapses to zero, and unbounded returns (undiscounted infinite horizons or heavy-tailed rewards).

---

## Q3. In the example, why integrate from 0 to θ? Shouldn't the domain be whatever x ranges over?

You're right. The expectation is over the whole domain of x. Write the density with an indicator:

p_θ(x) = (1/θ) · 𝟙[0 ≤ x ≤ θ]

Then J(θ) = ∫_{−∞}^{∞} p_θ(x) · 1 dx = 1, so the true gradient is 0. The 0-to-θ limits came from dropping the zero part of the integrand, which gives the same value but moves the θ-dependence into the limits.

**Where the θ-dependence actually sits:** With the fixed domain, the indicator 𝟙[x ≤ θ] jumps from 0 to 1 at θ = x, so it isn't differentiable there.

- **Naive swap:** ∇_θ p_θ(x) = −(1/θ²) 𝟙[0 ≤ x ≤ θ], which integrates to −1/θ. Wrong.
- **Careful swap:** The jump produces a Dirac delta: ∇_θ p_θ(x) = −(1/θ²) 𝟙[0 ≤ x ≤ θ] + (1/θ) δ(x − θ), which integrates to −1/θ + 1/θ = 0. Correct.

The delta term is the Leibniz boundary term in fixed-domain form. "The support moves with θ" and "p_θ(x) is discontinuous in θ" are the same failure. With a fixed domain, the condition that breaks is differentiability in θ. Softmax and Gaussian policies are smooth in θ with no moving edges, so no delta terms appear.

---

## Q4. What does "support" mean?

The support of a distribution is the set of values where its probability (or density) is greater than zero: everywhere the random variable can actually land.

- **Uniform(0, θ):** support is [0, θ], and it changes with θ.
- **Gaussian:** support is all of ℝ, whatever the mean and variance.
- **Softmax policy:** support is every action, since each gets probability > 0.

"The support moves with θ" means changing θ changes which values are possible at all, not just how likely they are.

---

## Q5. In the Leibniz rule, can the gradient variable (θ) and the integration variable (τ) be the same?

They have to be different. The rule is about a parameter (θ) you differentiate with respect to, and a separate integration variable (τ) that gets integrated away:

d/dθ ∫ f(τ, θ) dτ

τ is a dummy variable. After integrating, only a function of θ remains.

θ can appear in the limits, as in d/dθ ∫₀^θ f(τ) dτ = f(θ) (the fundamental theorem of calculus, a special case of Leibniz), but τ and θ are still separate variables. Writing ∫₀^x f(x) dx is sloppy notation; rename the inner variable. On the slide, τ is the trajectory and θ is the policy parameter.

---

## Q6. If the limits of integration involve θ, can I still move the gradient inside?

Not by itself. The full Leibniz rule adds a term for each limit:

d/dθ ∫_{a(θ)}^{b(θ)} f(x, θ) dx = f(b(θ), θ) · b′(θ) − f(a(θ), θ) · a′(θ) + ∫_{a(θ)}^{b(θ)} ∂f/∂θ dx

**Checking it on the uniform example** (f = 1/θ, a = 0, b = θ):

- Upper limit: (1/θ) · 1 = 1/θ
- Lower limit: 0, since a′ = 0
- Inside term: ∫₀^θ (−1/θ²) dx = −1/θ
- Total: 0. Correct.

The boundary terms vanish if the limits don't depend on θ, or if f is zero at whichever limits move. Policy gradients integrate over a fixed space of trajectories, so they're fine.

---

## Q7. With importance sampling, the gradient is over the new policy but the expectation is over the old policy's state visitation. So I can't move the gradient inside?

Actually you can.

**Exact trajectory version (first equation on slide 24):**

J(θ′) = E_{τ~p_θ}[ (p_θ′(τ) / p_θ(τ)) · r(τ) ]

The expectation is over p_θ, and θ is held fixed. θ′ appears only inside the ratio, so the gradient moves inside with no boundary terms:

∇_θ′ J(θ′) = E_{τ~p_θ}[ ∇_θ′ (p_θ′(τ) / p_θ(τ)) · r(τ) ]

One condition remains: p_θ must be positive wherever p_θ′ is.

**Per-timestep common final form:** The exact per-timestep objective takes the expectation over states visited by the new policy, d^{π_θ′}(s), which depends on θ′. The slide replaces it with the old policy's state distribution by setting π_θ′(s)/π_θ(s) ≈ 1. You're then differentiating an approximate objective whose gradient ignores how θ′ changes which states you visit. The approximation is accurate when θ′ is close to θ and exact at θ′ = θ, which is why TRPO and PPO limit how far the policy moves (KL constraint or ratio clipping).

*(This framing was later clarified in Q11–Q14: the swap itself is never the issue.)*

---

## Q8. Explain the concern with the per-timestep case more clearly.

**The true objective:**

J(θ′) = E_{s ~ d^{θ′}} E_{a ~ π_θ′}[ return from (s, a) ]

*(Corrected in Q9: for J itself this should be the per-step reward; the return-from-(s, a) form belongs in the gradient.)*

θ′ affects the actions you pick in a given state, through π_θ′(a|s), and which states you reach, through d^{θ′}(s).

**What importance sampling fixes:** Reweighting by π_θ′(a|s)/π_θ(a|s) corrects for the action choice. That ratio is easy to compute.

**What it can't fix:** Your sampled states came from d^θ, not d^{θ′}. Correcting that needs d^{θ′}(s)/d^θ(s), which is hard because it requires the dynamics and, written out, brings back the product of all earlier action ratios.

**The surrogate you end up optimizing:**

L(θ′) = E_{s ~ d^θ, a ~ π_θ}[ (π_θ′(a|s) / π_θ(a|s)) · return ]

Moving the gradient inside L is legal. The issue is that L(θ′) is a different function from J(θ′): its gradient ignores how changing θ′ changes which states you reach.

**Example:** If the new policy turns left at step 1 instead of right, the surrogate still scores later actions using states from the "right" branch.

**Why it still works:** At θ′ = θ the state ratio is exactly 1 and ∇L = ∇J (the derivative of the state distribution doesn't appear in the policy gradient theorem). As θ′ moves away, the gap grows roughly with the KL divergence between the policies. TRPO and PPO keep updates in the region where the surrogate is accurate.

---

## Q9. How did we get J(θ′) = E_{s ~ d^{θ′}} E_{a ~ π_θ′}[…] from the original equation?

**Step 0: Start from the slide**

J(θ) = E_{τ ~ p_θ(τ)}[ Σ_{t=1}^{T} r(s_t, a_t) ]

where p_θ(τ) = p(s_1) · Π_t π_θ(a_t|s_t) · p(s_{t+1}|s_t, a_t).

**Step 1: Linearity of expectation**

J(θ) = Σ_t E_{τ ~ p_θ(τ)}[ r(s_t, a_t) ]

**Step 2: Each term only cares about (s_t, a_t)**

Integrate out all other variables. Future steps integrate to 1. Past steps leave the marginal p_θ(s_t).

E_τ[ r(s_t, a_t) ] = E_{s_t ~ p_θ(s_t)} E_{a_t ~ π_θ(a_t|s_t)}[ r(s_t, a_t) ]

J(θ) = Σ_t E_{s_t ~ p_θ(s_t)} E_{a_t ~ π_θ(a_t|s_t)}[ r(s_t, a_t) ]

**Step 3: θ shows up twice**

- In π_θ(a_t|s_t): the direct effect (which action you pick).
- In p_θ(s_t): the indirect effect (which states you reach, determined by earlier actions).

d^θ(s) is shorthand for these per-step marginals averaged over t.

**Step 4: Connect to slide 24**

π_θ′(s_t, a_t) / π_θ(s_t, a_t) = [ p_θ′(s_t) / p_θ(s_t) ] · [ π_θ′(a_t|s_t) / π_θ(a_t|s_t) ]

The action ratio survives in the common final form. The state ratio is set to 1, which drops the indirect effect.

**Where "return from (s, a)" came from:** It belongs in the gradient. ∇ log π_θ(a_t|s_t) gets multiplied by every reward from t onward (earlier rewards don't depend on a_t), giving the reward-to-go.

---

## Q10. (Notation)

Plain-text LaTeX-style subscripts are hard to read when not rendered. A plain-notation version was provided; the user then asked to return to LaTeX-style notation.

---

## Q11. Is linearity of expectation still true if the summation bounds depend on the random variable?

Not directly. E[Σ_{t=1}^{T} X_t] = Σ_{t=1}^{T} E[X_t] only when T is fixed. If T is random, the right-hand side doesn't make sense.

**Fix: move the bound into the summand with an indicator**

Σ_{t=1}^{T} X_t = Σ_{t=1}^{∞} X_t · 𝟙[t ≤ T]

E[Σ_{t=1}^{T} X_t] = Σ_{t=1}^{∞} E[X_t · 𝟙[t ≤ T]]

Caveats:

1. Swapping E with an infinite sum needs a Fubini/Tonelli condition (e.g., X_t ≥ 0, or E[Σ_t |X_t| 𝟙[t ≤ T]] < ∞). Bounded rewards with E[T] < ∞ suffice.
2. In general E[X_t 𝟙[t ≤ T]] ≠ E[X_t] · P(t ≤ T).

**Counterexample:** X_1, X_2 i.i.d. fair coins in {0, 1}; T = 1 if X_2 = 1, else T = 2. The sum always equals X_1, so the true value is 1/2. The naive split gives 1/2 + 1/2 · 1/2 = 3/4. It breaks because T peeks at the future. If T is a stopping time, things behave better (Wald's identity: E[Σ_{t=1}^{T} X_t] = E[T] · μ for i.i.d. X_t with mean μ and E[T] < ∞).

**In RL:** Termination is a stopping time, so 𝟙[t ≤ T] depends only on s_1, …, s_t, and the marginalization in Q9 Step 2 still works. The common shortcut is an absorbing terminal state with zero reward, which gives a fixed T (or infinite horizon with discount γ), so plain linearity applies.

---

## Q12. If the gradient can't move inside without importance sampling and proximity, how did it move inside on slide 9?

You can always move the gradient inside the integral when the regularity conditions hold. The swap is never the problem.

**Slide 9: inside the integral, not the expectation**

∇_θ ∫ p_θ(τ) r(τ) dτ = ∫ ∇_θ p_θ(τ) r(τ) dτ

The domain is fixed and p_θ is smooth, so this is legal. The gradient lands on the distribution. You can't naively write ∇_θ E_{τ ~ p_θ}[r(τ)] = E_{τ ~ p_θ}[∇_θ r(τ)], since the right side is 0. The log trick turns ∇_θ p_θ back into p_θ ∇_θ log p_θ.

**The on-policy gradient is exact, including state visitation**

∇_θ log p_θ(τ) = Σ_t ∇_θ log π_θ(a_t|s_t) covers every action, so it already accounts for earlier actions changing later states. That's why the reward-to-go multiplies ∇ log π(a_t|s_t) by all rewards from t onward.

**The off-policy problem is about samples, not swapping**

1. Exact trajectory importance sampling: exact, no proximity needed, high variance.
2. Per-timestep form with the state ratio set to 1: a surrogate that matches J only to first order around θ′ = θ.

The proximity assumption makes L(θ′) a good stand-in for J(θ′); it isn't what permits the swap.

---

## Q13. Is this accurate? "The top equation on slide 24 needs nothing for the swap because of the log trick; the second needs importance sampling to move the gradient in."

Not quite.

**The log trick doesn't justify the swap.** It's a rewrite applied after the gradient is inside. Regularity justifies the swap.

**Top equation:**

1. Importance sampling (exact if π_θ > 0 wherever π_θ′ > 0): J(θ′) = E_{τ ~ p_θ}[ (p_θ′(τ) / p_θ(τ)) · r(τ) ]
2. Move the gradient in (regularity; easy since p_θ doesn't depend on θ′).
3. Log trick: ∇_θ′ (p_θ′/p_θ) = (p_θ′/p_θ) · ∇_θ′ log p_θ′.
4. Causality gives the reward-to-go.

**Second equation:**

1. Start from the on-policy gradient at θ′ (slide 9's argument, per timestep):
   ∇_θ′ J(θ′) = Σ_t E_{(s_t, a_t) ~ p_θ′}[ ∇_θ′ log π_θ′(a_t|s_t) · Q̂_t ]
2. Importance sampling changes the sampling distribution from p_θ′ to p_θ with weight p_θ′(s_t, a_t) / p_θ(s_t, a_t). The gradient is already inside.

**Where the approximation comes in:** The second equation is still exact as written. The approximation enters in the common final form (state ratio set to 1). Using the old policy's reward-to-go is a second approximation of the same kind.

**Corrected summary:** The gradient moves inside by regularity in every case. Importance sampling is only about using old samples, and it's exact. Only dropping the state ratio needs θ′ close to θ.

---

## Q14. Then why was the common final form described as "where your concern is valid"?

That was a mistake. The concern that the gradient can't move inside doesn't hold anywhere, including the common final form. The swap is justified by regularity and happens before importance sampling or any approximation.

The real, separate issue is that setting the state ratio to 1 means you're no longer computing the exact gradient of J(θ′). That's approximation error, not swap legality.

From the TRPO/PPO side, you write the surrogate L(θ′) first and then differentiate it. That swap is legal too, since p_θ doesn't depend on θ′. The cost is that you're optimizing an approximation of J.

---

## Q15. From the TRPO/PPO side, does importance sampling make the swap valid?

Not quite. It makes the swap simpler, but the swap was already valid.

**Without importance sampling:** J(θ′) = ∫ p_θ′(τ) r(τ) dτ. The swap is legal by regularity; the gradient lands on p_θ′, so you need the log trick.

**With importance sampling:**

L(θ′) = E_{(s,a) ~ p_θ}[ (π_θ′(a|s) / π_θ(a|s)) · Â(s, a) ]

The sampling distribution doesn't depend on θ′, so the gradient goes straight onto a smooth ratio. No log trick needed. It's like differentiating an average over a fixed dataset. You still need mild regularity: π_θ′ differentiable, π_θ > 0 wherever π_θ′ > 0, and a bounded integrand.

Importance sampling changes what the gradient acts on (distribution → integrand). In PPO, you compute the ratio on a stored batch and backpropagate through it like any loss.

---

## Final Summary

- **Gradient swap (∇∫ = ∫∇):** Valid by regularity (fixed domain, differentiable in θ, bounded derivative) on slide 9 and both equations on slide 24.
- **Log trick:** A rewrite after the swap that turns ∇p_θ back into a sampleable expectation. It doesn't justify the swap.
- **Importance sampling:** Exact (given support coverage). It lets you use old-policy samples and moves θ′-dependence from the distribution into the integrand.
- **The only approximation:** Setting the state ratio p_θ′(s_t)/p_θ(s_t) to 1 in the common final form (plus using the old policy's reward-to-go). This is accurate only when θ′ ≈ θ, which motivates TRPO/PPO trust regions.