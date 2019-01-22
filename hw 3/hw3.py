import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

try:
    import atari_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install Atari dependencies by running 'pip install gym[atari]'.)".format(e))

def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size),dtype=np.uint8)
    ale.getRAM(ram)
    return ram


class preprocessing:
     def check_agent_position(x,y,MonsterArea):

        if (x,y) in MonsterArea:
            preprocessing.check_agent_position(x,y,MonsterArea)
        else:
            return "True"

     def performactionENV(env,AgentPos,MosterArea,action):


        if action == 0:
            #want to move toward right 
            print("Move right")
            print("Current Position of the Agent ",AgentPos[0])

            x,y = AgentPos[0]

            if y < env.shape[1]:
                env[x][y] = '_'
                

                if (x,y+1) in MosterArea:
                    print("Agent Died Because Moster in position ",(x,y+1))
                    return;
                else:
                    env[x][y+1] = 'A'


                    AgentPos = []
                    AgentPos.append((x,y+1))

            print("Moved Right")
            print("Current Position of the Agent ",AgentPos[0]) 

        elif action == 1:
            print("Move Left")
            print("Current Position of the agent is ",AgentPos[0])

            x,y = AgentPos[0]

            if y >0 :
                env[x][y] = '_'

                if (x,y-1) in MosterArea:
                    print("Agent Died Because Moster in position ",(x,y-1))
                    return;
                else:
                    env[x][y-1] = 'A'
                    AgentPos = []
                    AgentPos.append((x,y-1))

            print("Moved Left")
            print("Current Position of the agent is ",AgentPos[0])
        elif action == 2:
            print("Move up")
            print("Current Position of the agent is ",AgentPos[0])

            x,y = AgentPos[0]

            if x >0 :
                env[x][y] = '_'

                if (x-1,y) in MosterArea:
                    print("Agent Died Because Moster in position ",(x,y-1))
                    return;
                else:

                    env[x-1][y] = 'A'
                    AgentPos = []
                    AgentPos.append((x-1,y))

            print("Moved up")
            print("Current Position of the agent is ",AgentPos[0])
        elif action == 3:
            print("Move down")
            print("Current Position of the agent is ",AgentPos[0])

            x,y = AgentPos[0]

            if x >0 :
                env[x][y] = '_'
                if (x+1,y) in MosterArea:
                    print("Agent Died Because Moster in Position",(x+1,y))
                else:

                    env[x+1][y] = 'A'
                    AgentPos = []
                    AgentPos.append((x+1,y))

            print("Moved down")
            print("Current Position of the agent is ",AgentPos[0])

        return env , AgentPos,  MosterArea


class AtariEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game='pong', obs_type='ram', frameskip=(2, 5), 
            repeat_action_probability=0., full_action_space=False):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(self, game, obs_type, frameskip, repeat_action_probability)
        assert obs_type in ('ram', 'image')

        self.game_path = atari_py.get_game_path(game)
        if not os.path.exists(self.game_path):
            raise IOError('You asked for game %s but path %s does not exist'%(game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.ale = atari_py.ALEInterface()
        self.viewer = None

        self.MosterArea = []
        self.AgentArea  = []

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(repeat_action_probability, (float, int)), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_probability)

        self.seed()

        self._action_set = (self.ale.getLegalActionSet() if full_action_space 
                            else self.ale.getMinimalActionSet())
        self.action_space = spaces.Discrete(len(self._action_set))

        (screen_width,screen_height) = self.ale.getScreenDims()
        if self._obs_type == 'ram':
            self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(128,))
        elif self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)
        self.ale.loadROM(self.game_path)
        return [seed1, seed2]

    def step(env,AgentPos,MonsterArea,action):
        reward = 0
        env,AgentPos,MosterArea = preprocessing.performactionENV(env,AgentPos,MonsterArea,action)
        return env,AgentPos,MosterArea

    def _get_image(self):
        return self.ale.getScreenRGB2()

    def _get_ram(self):
        return to_ram(self.ale)

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        if self._obs_type == 'ram':
            return self._get_ram()
        elif self._obs_type == 'image':
            img = self._get_image()
        return img


   

    # return: (states, observations)
    def reset():
        
        Matrix  = np.chararray((10,10))
        Matrix[:] = '_'


        MonsterArea = []
        AgentPos    = []
        for row in range(Matrix.shape[0]):
            for column in range(Matrix.shape[1]):

                if ( (row == 4 and column == 5) or (row == 8 and column == 7) or (row == 6 and column == 3) ):
                    print("Moster is at ",(row,column)," is denoted As M")
                    MonsterArea.append((row,column))
                    Matrix[row][column] = "M"
        
        print("complete environment before agent is placed ",Matrix)

        #lets put the agent in the position 

        agent_x  = np.random.randint(10)
        agent_y  = np.random.randint(8)

        
        if preprocessing.check_agent_position(agent_x,agent_y,MonsterArea) == "True":

            Matrix[agent_x][agent_y] = "A"
            AgentPos.append((agent_x,agent_y))


        print("Environment After the Agent placed",Matrix)


        return Matrix,AgentPos,MonsterArea








	

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def clone_state(self):
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """Restore emulator state w/o system state."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """Clone emulator state w/ system state including pseudorandomness.
        Restoring this state will give an identical environment."""
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """Restore emulator state w/ system state including pseudorandomness."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)

ACTION_MEANING = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}
