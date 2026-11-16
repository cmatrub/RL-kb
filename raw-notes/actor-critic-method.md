# Actor-Critic Q&A — Stanford CS224R Lecture 4 (Actor-Critic Methods)

---

## Q1: When using the critic, do you run it once for sₜ and then for sₜ₊₁, in sequence?

Yes, you evaluate the same critic twice, once on sₜ and once on sₜ₊₁. But the two calls don't have to happen in any order.

The advantage estimate on the board:

A^π(sₜ, aₜ) ≈ r(sₜ, aₜ) + V^π(sₜ₊₁) − V^π(sₜ)

So you need two critic outputs: V(sₜ), the value of the state you were in, and V(sₜ₊₁), the value of the state you ended up in.

**Why order doesn't matter:** Both calls use the same network with the same weights, and neither one depends on the other's result. In practice you put all the sₜ's and sₜ₊₁'s into one tensor, run one forward pass, and split the outputs.

Practical details:
- **Stop-gradient on the target:** When training the critic, the target r + γV(sₜ₊₁) is treated as a constant. Gradients flow only through V(sₜ).
- **Discount:** With a discount factor, the formula becomes r + γV(sₜ₊₁) − V(sₜ).
- **Terminal states:** If sₜ₊₁ is terminal, set V(sₜ₊₁) = 0.

---

## Q2: So take all time steps across all trajectories and run them in batches?

Yes, that's the usual approach in batch actor-critic. Since sₜ₊₁ for one step is sₜ for the next, you can evaluate every state once (including each trajectory's final state) and then shift by one:

```python
V = critic(all_states)              # one batched forward pass
V_t    = V[:-1]                     # values of s_t
V_next = V[1:]                      # values of s_{t+1}
V_next[terminal] = 0                # zero out where the episode ended
A = r + gamma * V_next - V_t
```

Track trajectory boundaries so the last state of one trajectory isn't paired with the first state of the next.

In the **online** version, you update after every transition, so you just evaluate V(sₜ) and V(sₜ₊₁) for that one step. People often run several environments in parallel to get a small batch.

---

## Q3: Is the loop training or inference? Why rerun the critic to compute advantages?

The whole loop is training. The critic does two jobs inside it:
- **Inference (forward pass, no learning):** produces V values for targets and advantages.
- **Training:** regresses V(sₜ) toward its target.

Rerunning the critic after it's updated is **optional**. It gives slightly fresher values, but many implementations just reuse the earlier ones.

---

## Q4: The correction: targets come from the trajectories, not the value network

The user pointed out that you can use reward-to-go from the rollout as the target. Both methods are standard:

1. **Monte Carlo targets:** yₜ = Σ rₜ′ from the rollout. Fit the critic to these, then use it for advantages. **Unbiased but high-variance.**
2. **Bootstrapped (TD) targets:** yₜ = r(sₜ, aₜ) + γV̂(sₜ₊₁). **Lower variance but biased** while the critic is still inaccurate.

Loop with Monte Carlo targets:
1. Collect rollouts.
2. Compute reward-to-go targets from the rewards.
3. Fit the critic to those targets.
4. Run the critic on sₜ and sₜ₊₁ (one batch), compute advantages, and update the actor.

---

## Q5: In TD targets, does r(s, a) come from the environment?

Yes. It's the actual reward the environment returned, stored with the transition (sₜ, aₜ, rₜ, sₜ₊₁). Only V(sₜ₊₁) comes from the critic.

A TD target is **one real reward plus the critic's estimate of everything after that**. A Monte Carlo target uses real rewards all the way to the end.

---

## Q6: What is bias?

Bias is a **consistent error in one direction**: even with infinitely many samples, the average lands on the wrong value. Variance is the scatter around the average. Bias is the average itself being off.

- **Monte Carlo:** Individual returns are noisy, but their average converges to the true V. High variance, no bias.
- **TD:** If the critic overestimates V(sₜ₊₁) by 5, every target that uses sₜ₊₁ is 5 too high, and more samples don't help. That's bias. Low variance, since there's only one random reward.
- **Effect on the actor:** Advantages from a biased critic can systematically push the policy toward actions that only look good because of the critic's error. The bias shrinks as the critic improves.

> Analogy: variance = shaky hand on a scale; bias = scale miscalibrated by 5 kg. Averaging fixes the first, not the second.

---

## Q7: Why is V(s_{i,t+1}) indexed by i? Shouldn't it be independent of i?

The **function** V is the same for every i, but its **input** depends on i.

- i = trajectory index, t = time step.
- s_{i,t+1} = the state trajectory i actually reached at step t+1.
- Different rollouts visit different states, so they get different values from the same network φ.

Each (i, t) pair is its own training example: (s_{i,t}, y_{i,t}).

---

## Q8: Doesn't the target r + V̂(s_{t+1}) look like Q(s_{i,t}, a_{i,t})?

Yes. It's a **one-sample estimate of Q(sₜ, aₜ)**, since it's tied to the specific action taken. The Monte Carlo target is too.

It still trains **V** because the network's input is **only the state**. Actions were sampled from π, so the same state can appear with different actions and different targets. Squared-error regression learns the average target for each input:

V̂(s) → E_{a∼π}[Q(s, a)] = V^π(s)

Each target is a noisy Q sample, and averaging over the policy's actions turns it into V. This only holds if the data came from the current policy π, which is one reason this setup is **on-policy**.

---

## Q9: "Propagating through time" vs. backpropagation

- **Propagating through time:** With TD targets, reward information spreads backward across states over repeated fitting iterations, one step per round.
- **Backpropagation:** The ordinary gradient computation through the network's layers, used to update φ within one update. Unrelated to time steps.
- **Link:** You don't backpropagate through the target. V̂(sₜ₊₁) inside y is treated as a fixed number. Value spreads across time only because you refit over and over.

---

## Q10: With TD, all time steps are in the batch with no ordering, so how is it "backward"?

Correct, all time steps are trained together in no particular order. The "backward" part is about **what each target can know at a given point**, not processing order.

Example: a 4-step episode, +10 reward only at the last step, critic starts at about 0 everywhere.

- **Iteration 1:** The last step's target = 10 (correct). Earlier targets = 0 + V(next) ≈ 0. Only the state right before the reward learns its value.
- **Iteration 2:** The state two steps before the end gets 0 + V(next) ≈ 10, so it learns.
- **Iteration 3:** One more step back learns, and so on.

Each target contains only **one real reward plus the critic's current guess**, so correct values spread backward about one step per fitting round.

Monte Carlo targets don't have this delay. Every target already includes the +10 directly, so they're correct from iteration 1, just noisier.