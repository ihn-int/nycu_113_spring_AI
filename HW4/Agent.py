import random


class Agent:
    # Part 6: alpha
    def __init__(self, k, epsilon, alpha=None):
        self._k = k
        self._epsilon = epsilon
        self._alpha = alpha
        
        # Reward exception
        self._expects = [0.0 for _ in range(k)]

        # Arm count
        self._k_counts = [0 for _ in range(k)]

    def select_action(self) -> int:
        # Test if to select new option
        if random.random() < self._epsilon:
            # Explore
            return random.randint(0, self._k - 1)
        else:
            # Exploit
            return self._expects.index(max(self._expects))
        
    def update_q(self, action, reward) -> None:
        if not self._alpha is None:
            # Part 6: constant step-size update
            # Q(a) = Q(a) + alpha * (R-Q(a))
            self._expects[action] += self._alpha * (reward - self._expects[action])
        else:
            # Sample average method
            self._k_counts[action] += 1
            n = self._k_counts[action]
            old_q = self._expects[action]

            # Update expection value
            new_q = old_q + (reward - old_q) / n
            self._expects[action] = new_q

    def reset(self) -> None:
        self._expects = [0.0 for _ in range(self._k)]
        self._k_counts = [0 for _ in range(self._k)]