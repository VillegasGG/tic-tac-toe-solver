class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def choose_move(self, env):
        actions = env.get_actions()
        action = input(f"Enter action ({', '.join(map(str, actions))}): ")
        return int(action)
    
    