# Stochastic Thinking, Simulation and Sampling

## 1 Stochastic Process

A **stochastic process** is a collection of random variables representing a process that evolves over time.

Here is the code of simulation of rolling a dice:

```python
def simulate_dice(goal, num_trials, txt):
    total = 0
    for i in range(num_trials):
        result = ''
        for j in range(len(goal)):
            result += str(np.random.choice([1, 2, 3, 4, 5, 6]))
        if result == goal:
            total += 1\
    print(f'Actual Probability of {txt} = {1/6**len(goal)}')
    print(f'Estimated Probability of {txt} = {total/num_trials}')
    
# Actual Prob. 0.0001286
# Estimated Prob. 0.0
```

## 2 Simulation and Sampling

Well, however, simulations are often useful, compared to analytical algorithm.

### 2.1 The Birthday Problem

::: tip The Birthday Problem

What is the probability of at least two people in a group having the same birthday, given that the group has $m$ people?

> If $m>366$, then, $P(m)=1$, for pigeonhole principle.
>
> If $m\in[1,365]$, the answer is
> $$
> 1-\frac{365!}{(365-n)!}\times\frac{1}{365^n}.
> $$
> And we have
> $$
> \begin{gather}
> P_{365}(1) = 0, P_{365}(2) = 1/365,\cdots, P_{365}(22) = 47.6\%, \\P_{365}(23) = 50.7\%,\cdots P_{365}(100)=99.999969\%
> \end{gather}
> $$

:::

What if at least $n$ people having the same birthday, given that the group has $m$ people?

```python
def birthday_problem(n, m, num_trials):
    total = 0
    for i in range(num_trials):
        birthdays = np.random.choice(range(365), m)
        if len(set(birthdays)) < m-n:
            total += 1
    print(f'Estimated Probability of at least {n} people having the same birthday = {total/num_trials}')
```

## 2.2 Simulation

1. **Simulation models** are computational frameworks that simulate the behavior of a system, offering valuable insights into its potential dynamics and outcomes.

## 2.3 Monte Carlo Simulation

1. **Monte Carlo simulation** is a computational technique that generates random variables for modeling risk or uncertainty of a system. It uses random sampling to obtain numerical results, based on principles of inferential statistics:
   1. **Random Sampling**: Randomly sample from a distribution.
   2. **Popluation**: The entire set of possible outcomes.
   3. **Sample**: A subset of the population.

2. **Example**: Estimate the value of $\pi$ using Monte Carlo simulation.

```python
def estimate_pi(num_points):
    points = np.random.rand(num_points, 2)
    inside = np.sum(np.linalg.norm(points, axis=1) < 1)
    return 4*inside/num_points
```

The estimated value of $\pi$ varys with the number of points:

```python
print(estimate_pi(10)) # 3.6
print(estimate_pi(100)) # 3.08
print(estimate_pi(1000)) # 3.144
print(estimate_pi(10000)) # 3.1416
print(estimate_pi(100000)) # 3.14152
```

:::tip Another Step-by-step Example: Toss coins

1. Consider one flip. How confident would you be about answering $1.0$?

	> Half.

2. Consider two flips.  Assume you know nothing about probability, what do you think the next flip will show heads?

    > Half.

3. Consider 100 flips, 1000 flips, 10000 flips...... What do you think the next flip will show heads?

4. Confidence in our estimate depends on two factors:
   1. Size of the sample.
   2. Variance of the sample. As the variance grows, we need larger size of samples to maintain the same level of confidence.

:::

5. Not prefect but precise: Never possible to guarantee perfect accuracy through sampling, but some estimates are precise enough.

	::: danger Question

	How many simulations do we need to have justified confidence on our answer?

	> It depends on the  variability of the results. Theories that supports it are the Large Number Theorem, Central Limit Theorem and Confidence Interval.

	:::

## 2.4 LNT, CLT and CI

1. **Large Number Theorem (LNT)**: As the number of trials increases, the average of the results will converge to the expected value. 

2. **Central Limit Theorem (CLT)**: As the number of trials increases, the distribution of the results will converge to a normal distribution.

3. **Confidence Interval (CI)**: A range of values that is likely to contain the true value of an unknown population parameter. It is calculated from the sample data, and provides a range of values that is likely to contain the true value of the population parameter.
   $$
    \text{CI} = \text{mean} \pm \text{margin of error}
   $$
    where the margin of error is calculated as
    $$
    \text{margin of error} = \text{critical value} \times \text{standard error}
    $$
    and the standard error is calculated as
    $$
    \text{standard error} = \frac{\text{standard deviation}}{\sqrt{\text{sample size}}}
    $$
    The critical value is determined by the desired confidence level. For example, for a 95% confidence level, the critical value is 1.96.
