from tictactoe import TicTacToeEnv
from agent_player import AgentPlayer

def play(rounds=10000):
    """
    The agent plays against itself for a number of rounds
    """
    player1 = AgentPlayer('Bot1')
    player2 = AgentPlayer('Bot2')

    for _ in range(rounds):
        env = TicTacToeEnv()
        env.reset()
            
        terminal = False

        if _ % 1000 == 0:
            print('Playing...')
            print('Round:', _+1)

        while not terminal:

            # Player 1
            action = player1.choose_action(env)

            _, winner, terminal, _, _ = env.step(action)
            player1.addState(tuple(env.get_observation(1).flatten()))
            env.render()

            if terminal:
                result = env.get_result(1)
                if result == 1:
                    player1.feedReward(1)
                elif result == -1:
                    player1.feedReward(-1)
                else:
                    player1.feedReward(0)
                break

            # Player 2
            action = player2.choose_action(env)

            _, winner, terminal, _, _ = env.step(action)
            player2.addState(tuple(env.get_observation(-1).flatten()))
            env.render()

            if terminal:
                result = env.get_result(-1)
                if result == 1:
                    player2.feedReward(1)
                elif result == -1:
                    player2.feedReward(-1)
                else:
                    player2.feedReward(0)
                break

        # At the end of the game, show the policy
        # print('Player 1 policy:', player1.showPolicy())
        # print('Player 2 policy:', player2.showPolicy())

    # Save the policy
    player1.savePolicy('policy1.pkl')
    player2.savePolicy('policy2.pkl')
            
play()


