import random

class BanditEnv:
    def __init__(self, k, stationary = True):
        self._k = k
        self._steps = []
        self._rewards = []
        self._means = [random.gauss(0, 1) for _ in range(k)]

        # Part 4: stationary
        self._stationary = stationary

        # Record optimal arm in stationary mode to improve efficiency
        if stationary:
            self._opt_k = self._means.index(max(self._means))

    def reset(self) -> None:
        self._steps = []
        self._rewards = []
        self._means = [random.gauss(0, 1) for _ in range(self._k)]
        
        if self._stationary:
            self._opt_k = self._means.index(max(self._means))

    def step(self, action: int) -> float:
        # Check input
        if action < 0 or action >= self._k:
            return -100 # A dummy value
        
        # Generate reward
        mean = self._means[action]
        reward = random.gauss(mean, 1)

        # Update history
        self._steps.append(action)
        self._rewards.append(reward)

        # Part 4: update means
        if not self._stationary:
            for i in range(self._k):
                self._means[i] += random.gauss(0, 0.01)

        # Return
        return reward

    def export_history(self) -> tuple:
        return self._steps, self._rewards
    
    # The function which is not in SPEC
    # Use this to get the index of optimal arm
    def get_optimal_action(self) -> int:
        if self._stationary:
            return self._opt_k
        else:
            return self._means.index(max(self._means))
