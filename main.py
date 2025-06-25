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
            elif winner > 0:
                print(f'Winner: {player1.name}')
            break

        action = player2.choose_action(env)
        obs, winner, terminal, _, _= env.step(action) 
        env.render()

        if terminal:
            print('Game Over')
            if winner == 0:
                print(winner)
                print('Draw')
            elif winner < 0:
                print(f'Winner: {player2.name}')
            break

def human_vs_bot():
    env = TicTacToeEnv()
    env.reset()

    terminal = False

    player1 = HumanPlayer('Human1')
    player2 = AgentPlayer('Bot')

    player2.loadPolicy('policy2.pkl')

    while not terminal:
        # Human
        action = player1.choose_action(env)
        _, winner, terminal, _, _ = env.step(action)
        env.render()

        if terminal:
            if winner > 0:
                print(f'Winner: {player1.name}')
            else:
                print('Draw')
            break

        # Bot
        action = player2.choose_action(env)
        _, winner, terminal, _, _ = env.step(action)
        env.render()

        if terminal:
            if winner < 0:
                print(f'Winner: {player2.name}')
            else:
                print('Draw')
            break

def main():
    
    #human_vs_human()

    human_vs_bot()
   
       

if __name__ == '__main__':
    main()