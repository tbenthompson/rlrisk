import numpy as np
import risk_ext

def strategy(state):
    # print(state)
    return (0,1)

winner = risk_ext.run_py_vs_dumb_game(strategy)
print(winner)

n = 1000
winners = [risk_ext.run_py_vs_dumb_game(strategy) for i in range(n)]
print(np.unique(winners, return_counts=True))
