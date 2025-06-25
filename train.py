from tictactoe import TicTacToeEnv
from agent_player import AgentPlayer
from tqdm import tqdm
from collections import deque
import pickle

def play(rounds=50000, show=False):
    """
    The agent plays against itself for a number of rounds
    """

    epsilon = 1.0
    min_epsilon = 0.01
    decay = 0.995

    results = deque(maxlen=1000)
    last_winrate = None
    threshold = 0.1

    rewards = deque(maxlen=1000)
    avg_rewards = []

    player1 = AgentPlayer('Bot1', epsilon=epsilon)
    player2 = AgentPlayer('Bot2', epsilon=epsilon)

    for episode in tqdm(range(rounds)):
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
        
        rewards.append(result)

        results.append(result)
        if len(results) >= 1000 and episode % 1000 == 0:
            winrate = results.count(1) / 1000
            if(last_winrate is not None):
                print(f'\nLast Winrate: {last_winrate:.2f}, Current Winrate: {winrate:.2f}')
                diff = abs(winrate - last_winrate)
                print(f'Difference: {diff:.4f}')
                if diff < threshold and winrate >= last_winrate:
                    print(f'\nWinrate stabilized at {winrate:.2f}, stopping training.')
                    break
            last_winrate = winrate
            avg_rewards.append(sum(rewards) / 1000)

        if episode % 1000 == 0:
            print(f'\nRound {episode}, Winrate: {results.count(1) / len(results):.2f}, Epsilon: {player1.epsilon:.4f}')

    # Save the policy
    player1.savePolicy('policy1.pkl')
    player2.savePolicy('policy2.pkl')

    with open('avg_rewards.pkl', 'wb') as f:
        pickle.dump(avg_rewards, f)
            
play()


