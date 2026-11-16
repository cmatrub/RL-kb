# Value Functions with Finite Horizon ($\gamma = 1$)

## Prompt: let's set gamma to 1 and use finite trajectories to T

With $\gamma=1$ and a finite horizon, linearity applies directly to a finite sum (no convergence condition needed). Index future steps by $t'\in\{t,\dots,T\}$. One subtlety: with a finite horizon the value functions depend on time-to-go, so strictly they are $V_t^\pi$, $Q_t^\pi$, $A_t^\pi$; the subscript is implied by $s_t$ below.

### The $t'$-step state distribution

For $t'>t$:

$$
p^\pi(s_{t'}\mid s_t)=\int\!\cdots\!\int\prod_{j=t}^{t'-1}\pi(a_j\mid s_j)\,p(s_{j+1}\mid s_j,a_j)\;da_t\,ds_{t+1}\cdots ds_{t'-1}\,da_{t'-1},
$$

with $p^\pi(s_{t}\mid s_t)=\delta(s_t)$. Conditioning also on $a_t$:

$$
p^\pi(s_{t'}\mid s_t,a_t)=\int_{\mathcal S}p(s_{t+1}\mid s_t,a_t)\,p^\pi(s_{t'}\mid s_{t+1})\,ds_{t+1},\qquad t'\ge t+1.
$$

### State value $V^\pi(s_t)$

**Expectation form**

$$
V^\pi(s_t)=\mathbb{E}\left[\sum_{t'=t}^{T}r(s_{t'},a_{t'})\;\middle|\;s_t\right]
=\sum_{t'=t}^{T}\mathbb{E}_{\,s_{t'}\sim p^\pi(\cdot\mid s_t),\;a_{t'}\sim\pi(\cdot\mid s_{t'})}\big[r(s_{t'},a_{t'})\big]
$$

**Integral form**

$$
V^\pi(s_t)=\sum_{t'=t}^{T}\int_{\mathcal S}\int_{\mathcal A}p^\pi(s_{t'}\mid s_t)\,\pi(a_{t'}\mid s_{t'})\,r(s_{t'},a_{t'})\,da_{t'}\,ds_{t'}
$$

### Action value $Q^\pi(s_t,a_t)$

**Expectation form**

$$
Q^\pi(s_t,a_t)=\mathbb{E}\left[\sum_{t'=t}^{T}r(s_{t'},a_{t'})\;\middle|\;s_t,a_t\right]
=r(s_t,a_t)+\sum_{t'=t+1}^{T}\mathbb{E}_{\,s_{t'}\sim p^\pi(\cdot\mid s_t,a_t),\;a_{t'}\sim\pi(\cdot\mid s_{t'})}\big[r(s_{t'},a_{t'})\big]
$$

**Integral form**

$$
Q^\pi(s_t,a_t)=r(s_t,a_t)+\sum_{t'=t+1}^{T}\int_{\mathcal S}\int_{\mathcal A}p^\pi(s_{t'}\mid s_t,a_t)\,\pi(a_{t'}\mid s_{t'})\,r(s_{t'},a_{t'})\,da_{t'}\,ds_{t'}
$$

### Advantage $A^\pi(s_t,a_t)$

**Expectation form**

$$
A^\pi(s_t,a_t)=\sum_{t'=t}^{T}\Big(\mathbb{E}\big[r(s_{t'},a_{t'})\mid s_t,a_t\big]-\mathbb{E}\big[r(s_{t'},a_{t'})\mid s_t\big]\Big)
$$

**Integral form**

$$
\begin{aligned}
A^\pi(s_t,a_t)={}&\Big[r(s_t,a_t)-\int_{\mathcal A}\pi(a\mid s_t)\,r(s_t,a)\,da\Big]\\
&+\sum_{t'=t+1}^{T}\int_{\mathcal S}\int_{\mathcal A}\Big[p^\pi(s_{t'}\mid s_t,a_t)-p^\pi(s_{t'}\mid s_t)\Big]\,\pi(a_{t'}\mid s_{t'})\,r(s_{t'},a_{t'})\,da_{t'}\,ds_{t'}
\end{aligned}
$$

### Boundary condition and Bellman recursion

The finite horizon gives a terminal condition that the infinite case lacks:

$$
Q^\pi(s_T,a_T)=r(s_T,a_T),\qquad V^\pi(s_T)=\int_{\mathcal A}\pi(a_T\mid s_T)\,r(s_T,a_T)\,da_T,
$$

and for $t<T$ the one-step recursion is

$$
Q^\pi(s_t,a_t)=r(s_t,a_t)+\int_{\mathcal S}p(s_{t+1}\mid s_t,a_t)\,V^\pi(s_{t+1})\,ds_{t+1},
$$

so the values can be computed exactly by backward induction from $T$.

---

## Prompt: why isn't $s_{t'}$ distributed over $p^\pi(\cdot\mid s_t, a_t)$? you are missing $a_t$

It isn't missing; it's integrated out. Which conditioning appears depends on which function you're looking at:

- In $Q^\pi(s_t,a_t)$, $a_t$ is **given**, so future states are drawn from $p^\pi(s_{t'}\mid s_t,a_t)$. That is what the $Q$ formula uses.
- In $V^\pi(s_t)$, $a_t$ is **not given**. It is itself a random variable, $a_t\sim\pi(\cdot\mid s_t)$, so the correct distribution for $s_{t'}$ is the one that averages over $a_t$:

$$
p^\pi(s_{t'}\mid s_t)=\int_{\mathcal A}\pi(a_t\mid s_t)\,p^\pi(s_{t'}\mid s_t,a_t)\,da_t .
$$

You can also see this in the definition above: the product $\prod_{j=t}^{t'-1}\pi(a_j\mid s_j)\,p(s_{j+1}\mid s_j,a_j)$ starts at $j=t$, so it contains $\pi(a_t\mid s_t)$, and the integral includes $da_t$. So $a_t$ is in there, just marginalized.

To make $a_t$ explicit in $V$, substitute the identity above:

$$
V^\pi(s_t)=\int_{\mathcal A}\pi(a_t\mid s_t)\left[r(s_t,a_t)+\sum_{t'=t+1}^{T}\int_{\mathcal S}\int_{\mathcal A}p^\pi(s_{t'}\mid s_t,a_t)\,\pi(a_{t'}\mid s_{t'})\,r(s_{t'},a_{t'})\,da_{t'}\,ds_{t'}\right]da_t,
$$

and the bracket is exactly $Q^\pi(s_t,a_t)$, giving $V^\pi(s_t)=\mathbb{E}_{a_t\sim\pi}\big[Q^\pi(s_t,a_t)\big]$.

The same applies to the advantage: the term $p^\pi(s_{t'}\mid s_t,a_t)-p^\pi(s_{t'}\mid s_t)$ deliberately compares "commit to this specific $a_t$" against "draw $a_t$ from $\pi$". If both were conditioned on $a_t$, the difference would be zero and $A^\pi$ would vanish.
