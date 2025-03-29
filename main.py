from tictactoe import TicTacToeEnv
from human_player import HumanPlayer
from agent_player import AgentPlayer

def human_vs_human():
    env = TicTacToeEnv()
    env.reset()

    terminal = False

    player1 = HumanPlayer('Human1')
    player2 = HumanPlayer('Human2')

    
    while not terminal:
        action = player1.choose_action(env)
        obs, winner, terminal, _, _ = env.step(action)
        env.render()
        print('terminal:', terminal)    

        if terminal:
            print('Game Over')
            if winner == 0:
                print('Draw')
                print('Row winner:', env._row_winner(obs))
                print('Col winner:', env._col_winner(obs))
                print('Main diag winner:', env._main_diag_winner(obs))
            elif winner > 0:
                print(f'Winner: {player1.name}')
                print('Row winner:', env._row_winner(obs))
                print('Col winner:', env._col_winner(obs))
                print('Main diag winner:', env._main_diag_winner(obs))

            break

        action = player2.choose_action(env)
        obs, winner, terminal, _, _= env.step(action) 
        env.render()

        if terminal:
            print('Game Over')
            if winner == 0:
                print(winner)
                print('Draw')
                print('Row winner:', env._row_winner(obs))
                print('Col winner:', env._col_winner(obs))
                print('Main diag winner:', env._main_diag_winner(obs))
            elif winner < 0:
                print(f'Winner: {player2.name}')
                print('Row winner:', env._row_winner(obs))
                print('Col winner:', env._col_winner(obs))
                print('Main diag winner:', env._main_diag_winner(obs))

            break


def main():
    
    human_vs_human()

    # human vs bot
    # env = TicTacToeEnv()
    # env.reset()

    # terminal = False

    # player1 = HumanPlayer('Human1')
    # player2 = AgentPlayer('Bot')

    # player2.loadPolicy('policy1.pkl')

    # while not terminal:
    #     # Human
    #     action = player1.choose_action(env)
    #     _, winner, terminal, _, _ = env.step(action)
    #     env.render()

    #     if terminal:
    #         print('Game Over -Human')
    #         print('Winner:', winner)
    #         print('Row winner:', env._row_winner)
    #         print('Col winner:', env._col_winner)
    #         print('main diag winner:', env._main_diag_winner)
    #         print('Winner:', env.get_result(1))
    #         if winner > 0:
    #             print(f'Winner: {player1.name}')
    #         else:
    #             print('Draw')
    #         break

    #     # Bot
    #     action = player2.choose_action(env)
    #     _, winner, terminal, _, _ = env.step(action)
    #     env.render()

    #     if terminal:
    #         print('Game Over -Bot')
    #         print('Winner:', winner)
    #         if winner < 0:
    #             print(f'Winner: {player2.name}')
    #         else:
    #             print('Draw')
    #         break


       

if __name__ == '__main__':
    main()