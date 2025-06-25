from tictactoe import TicTacToeEnv
from agent_player import AgentPlayer
from tqdm import tqdm

epsilon = 1.0
min_epsilon = 0.01
decay = 0.995

def play(rounds=50000, show=False):
    """
    The agent plays against itself for a number of rounds
    """
    player1 = AgentPlayer('Bot1', epsilon=epsilon)
    player2 = AgentPlayer('Bot2', epsilon=epsilon)

    for _ in tqdm(range(rounds)):
        env = TicTacToeEnv()
        env.reset()
            
        terminal = False

        while not terminal:

            # Player 1
            action = player1.choose_action(env)

            _, winner, terminal, _, _ = env.step(action)
            player1.addState(tuple(env.get_observation(1).flatten()))
            env.render(show)

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
            env.render(show)

            if terminal:
                result = env.get_result(-1)
                if result == 1:
                    player2.feedReward(1)
                elif result == -1:
                    player2.feedReward(-1)
                else:
                    player2.feedReward(0)
                break

        player1.epsilon = max(min_epsilon, player1.epsilon * decay)
        player2.epsilon = max(min_epsilon, player2.epsilon * decay)

        # Show epsilon value
        if _ % 1000 == 0:
            print(f'Epsilon for Player 1: {player1.epsilon}, Player 2: {player2.epsilon}')

    # Save the policy
    player1.savePolicy('policy1.pkl')
    player2.savePolicy('policy2.pkl')
            
play()


