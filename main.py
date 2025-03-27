from tictactoe_gym.envs.tictactoe_env import TicTacToeEnv
from human_player import HumanPlayer
from agent_player import AgentPlayer

def main():
    env = TicTacToeEnv()
    env.reset()

    terminal = False

    player1 = HumanPlayer('Human1')
    player2 = AgentPlayer('Bot')

    
    while not terminal:
        action = player1.choose_action(env)
        _, winner, terminal, _, _ = env.step(action)
        env.render()

        if terminal:
            print('Game Over')
            if winner == 0:
                print('Draw')
            elif winner > 0:
                print(f'Winner: {player1.name}')

        action = player2.choose_action(env)
        _, winner, terminal, _, _= env.step(action) 
        env.render()

        if terminal:
            print('Game Over')
            if winner == 0:
                print('Draw')
            elif winner < 0:
                print(f'Winner: {player2.name}')
       

if __name__ == '__main__':
    main()