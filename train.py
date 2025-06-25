from tictactoe import TicTacToeEnv
from agent_player import AgentPlayer
from tqdm import tqdm
from collections import deque
import pickle
import random
from board_transformations import canonical_board

def play(rounds=50000, show=False):
    """
    The agent plays against itself for a number of rounds
    """

    epsilon = 1.0
    min_epsilon = 0.01
    decay = 0.995

    results = deque(maxlen=1000)
    last_winrate = None
    threshold = 0.01

    rewards = deque(maxlen=1000)
    avg_rewards = []

    # Agent will play against itself
    agent = AgentPlayer('Bot', epsilon=epsilon)

    for episode in tqdm(range(rounds)):
        env = TicTacToeEnv()
        env.reset()
            
        terminal = False

        # Define agent role for this episode
        if episode % 2 == 0:
            agent_role = 1  # Agent is 1
            opponent_role = -1  
        else:
            agent_role = -1  # Agent is -1
            opponent_role = 1

        while not terminal:
            current_player = env._player

            if current_player == agent_role:
                action = agent.choose_action(env)
            else:
                action = random.choice(env.get_actions())
            
            _, winner, terminal, _, _ = env.step(action)

            if current_player == agent_role:
                board = tuple(env.get_observation(agent_role).flatten())
                canon = canonical_board(board)
                agent.addState(canon)
                
            if terminal:
                result = env.get_result(agent_role)
                if result == 1:
                    agent.feedReward(1)
                elif result == -1:
                    agent.feedReward(-1)
                else:
                    agent.feedReward(0)

                agent.states = [] 
                break

        agent.epsilon = max(min_epsilon, agent.epsilon * decay)

        rewards.append(result)
        results.append(result)

        if len(results) >= 1000 and episode % 1000 == 0:
            winrate = results.count(1) / 1000
            if(last_winrate is not None):
                print(f'\nEPISODE {episode}')
                print(f'\nLast Winrate: {last_winrate:.2f}, Current Winrate: {winrate:.2f}')
                diff = abs(winrate - last_winrate)
                print(f'Difference: {diff:.4f}')
                print(f'Epsilon: {agent.epsilon:.4f}, Rewards: {sum(rewards) / 1000:.2f}')
                if diff < threshold and winrate >= last_winrate and winrate >= 0.8:
                    print(f'\nWinrate stabilized at {winrate:.2f}, stopping training.')
                    break
            last_winrate = winrate
            avg_rewards.append(sum(rewards) / 1000)

    # Save the policy
    agent.savePolicy('policy.pkl')

    with open('avg_rewards.pkl', 'wb') as f:
        pickle.dump(avg_rewards, f)
            
play()


