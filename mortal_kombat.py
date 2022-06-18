import gym
import numpy as np
import retro


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class MortalKombatIIDiscretizer(Discretizer):
    """
    Use Sonic-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=[['RIGHT','RIGHT', 'A'],['LEFT','LEFT', 'A'], ['LEFT','LEFT', 'X'], ['RIGHT','RIGHT', 'X'], ['LEFT','LEFT', 'Z'],['RIGHT','RIGHT', 'Z'],['A'],['B'],['C'],['X'],['Y'],['Z'],['START']])


def main():
    env = retro.make(game='MortalKombatII-Genesis', use_restricted_actions=retro.Actions.DISCRETE)
    print('retro.Actions.DISCRETE action_space', env.action_space)
    env.close()

    env = retro.make(game='MortalKombatII-Genesis')
    env = MortalKombatIIDiscretizer(env)
    print('MortalKombatIIDiscretizer action_space', env.action_space)

    score = 0
    env.reset()
    for i in range(200000):
        env.render()
        action =  env.action_space.sample() #possible_actions[np.random.randint(len(possible_actions))]
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            print("Your Score at end of game is: ", score)
            # break
    env.reset()
    env.render(close=True)
    env.close()


if __name__ == '__main__':
    main()