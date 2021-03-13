import time
import numpy as np
import risk_ext
print(dir(risk_ext))
#
# start = time.time()
# for i in range(100):
#     risk_ext.tester()
# print('runtime', time.time() - start)

game = risk_ext.start_game()
def game_matrix():
    return game.board_state.reshape((game.n_territories, 1 + game.n_players))

print(game, game.board_state)
game.step(0, 1)
print(game, game.board_state)
game.step(0, 1)
print(game, game.board_state)
game.step(0, 1)
print(game, game.board_state)
__import__('ipdb').set_trace()
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
