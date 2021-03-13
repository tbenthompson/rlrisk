import time
import numpy as np
import risk_ext

start = time.time()
for i in range(1000000):
    risk_ext.tester()
print('runtime', time.time() - start)
#
# def strategy(state):
#     # print(state)
#     return (0,1)
#
# winner = risk_ext.run_py_vs_dumb_game(strategy, True)
# print(winner)
#
# n = 1000
# winners = [risk_ext.run_py_vs_dumb_game(strategy, False) for i in range(n)]
# print(np.unique(winners, return_counts=True))
