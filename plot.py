"""
Plotting average rewards from training a Tic Tac Toe agent.
"""
import matplotlib.pyplot as plt
import pickle

def plot_average_rewards(file_path='avg_rewards.pkl'):
    """
    Load average rewards from a pickle file and plot them.
    
    :param file_path: Path to the pickle file containing average rewards.
    """
    with open(file_path, 'rb') as f:
        avg_rewards = pickle.load(f)

    print(f'Average rewards loaded from {file_path}. Total episodes: {len(avg_rewards)}')
    print(f'Last 100 average rewards: {avg_rewards[-10:]}')

    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards, label='Average Rewards', color='blue')
    plt.title('Average Rewards Over Training Episodes')
    plt.xlabel('Episodes (in groups of 1000)')
    plt.ylabel('Average Reward')
    plt.grid()
    plt.legend()
    plt.savefig('average_rewards_plot.png')
    plt.show()
    print('Plot saved as average_rewards_plot.png')

plot_average_rewards()