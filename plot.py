"""
Plotting average rewards from training a Tic Tac Toe agent.
"""
import matplotlib.pyplot as plt
import pickle

def get_average_rewards(list_of_rewards):
    """
    Calculate the average rewards from a list of rewards from episodes
    """
    average_rewards = []
    sum_rewards = 0
    for i, el in enumerate(list_of_rewards):
        sum_rewards += el
        average_rewards.append(sum_rewards / (i + 1))
    return average_rewards

def plot_average_rewards():
    """
    Load list of rewards from a file and plot the average rewards of 3 files
    """
    file_path1 = 'total_rewards_ep1.pkl'
    file_path2 = 'total_rewards_ep2.pkl'
    file_path3 = 'total_rewards_ep3.pkl'

    with open(file_path1, 'rb') as f:
        rewards1 = pickle.load(f)
    
    with open(file_path2, 'rb') as f:
        rewards2 = pickle.load(f)

    with open(file_path3, 'rb') as f:
        rewards3 = pickle.load(f)

    average_rewards1 = get_average_rewards(rewards1)
    average_rewards2 = get_average_rewards(rewards2)
    average_rewards3 = get_average_rewards(rewards3)

    plt.figure()
    plt.plot(average_rewards1, label='Epsilon 0.01', color='blue')
    plt.plot(average_rewards2, label='Epsilon 0.05', color='orange')
    plt.plot(average_rewards3, label='Epsilon 0.1', color='green')
    plt.title('Average Rewards of Agents')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid()
    plt.savefig('average_rewards_plot.png')
    plt.show()

def plot_one_agent():
    """
    Load list of rewards from a file and plot the average rewards of one agent
    """
    file_path = 'total_rewards_ep3.pkl'

    with open(file_path, 'rb') as f:
        rewards = pickle.load(f)

    average_rewards = get_average_rewards(rewards)

    plt.figure()
    plt.plot(average_rewards, label='Epsilon inicial 0.1', color='green')
    plt.title('Average Rewards of One Agent')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid()
    plt.savefig('average_rewards_one_agent_plot.png')
    plt.show()

plot_one_agent()
