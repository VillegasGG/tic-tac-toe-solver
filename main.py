from tictactoe_gym.envs.tictactoe_env import TicTacToeEnv
from human_player import HumanPlayer

def main():
    env = TicTacToeEnv()
    env.reset()

    terminal = False

    human1 = HumanPlayer('Human1')
    human2 = HumanPlayer('Human2')

    
    while not terminal:
        action = human1.choose_move(env)
        _, winner, terminal, _, _ = env.step(action)
        env.render()

        if terminal:
            print('Game Over')
            if winner == 0:
                print('Draw')
            elif winner > 0:
                print(f'Winner: {human1.name}')

        action = human2.choose_move(env)
        _, winner, terminal, _, _= env.step(action) 
        env.render()

        if terminal:
            print('Game Over')
            if winner == 0:
                print('Draw')
            elif winner < 0:
                print(f'Winner: {human2.name}')
       

if __name__ == '__main__':
    main()