"""
Linear Adaptive Cruise Control in Relative Coordinates.
The visualization fixes the position of the leader car.
From N. Fulton and A. Platzer,
"Safe Reinforcement Learning via Formal Methods: Toward Safe Control through Proof and Learning",
AAAI 2018.

OpenAI Gym implementation adapted from the classic control cart pole environment.
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random


logger = logging.getLogger(__name__)

class ACCEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def is_crash(self, some_state):
      return some_state[0] <= 1

    def __init__(self):
        # Makes the continuous fragment of the system determinitic by fixing the
        # amount of time that the ODE evolves.
        self.TIME_STEP = 0.1

        # The maximum separation between the leader and follower before the
        # state becomes a terminal state.
        self.MAX_VALUE = 100

        # The rates at which the vehicle's velocities change when increasing
        # and closing the relative distance, respectively. B will be negative
        # when action = 0.
        self.A = 100
        self.B = 100

        # Obsoleted; just need to figure out how the observation space
        # works...
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(3) # acc = -,0,+
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _acc_from_action(self, action):
      """Comptes the choice of acceleration from a discrete sample space -- ACC, 0, DECEL.
         Choice of acceleration will be return_value * TIME_STEP "meters/second".
      """
      assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
      #print "Action is: " , action
      if(action == 0):
        return -self.B
      elif(action == 1):
        return 0
      elif(action == 2):
        return self.A

    FAULT_RATE = None
    ERROR_MAGNITUDE = None
    def _step(self, action):
        assert self.FAULT_RATE != None and self.ERROR_MAGNITUDE != None, "FAULT_RATE and ERROR_MAGNITUDE should be initialized %s, %s" % (self.FAULT_RATE, self.ERROR_MAGNITUDE)

        if (random.uniform(0, 1) >= self.FAULT_RATE):
            return self._stepByModel(action)
        else:
            #print "[env/acc] INJECTING ERROR"
            state, reward, done, infos = self._stepByModel(action)
            state[0] = state[0] - self.ERROR_MAGNITUDE
            self.state = state

            #COPY PASTA
            done = self.is_crash(self.state) or self.state[0] > self.MAX_VALUE
            done = bool(done)
            if not done:
                reward = 1.0
            elif done and self.state[0] <= 1:
                reward = -100.0
            elif done and self.state[0] > self.MAX_VALUE - 0.5:
                reward = -100.0
            else:
                assert False, "Not sure why this should happen, and when it was previously there was a bug in the if/elif guards..."
                reward = 0.0

            return state, reward, done, infos


    def _stepByModel(self, action):
        assert self.action_space.contains(action), "%s (of type %s) invalid" % (str(action), type(action))
        state = self.state

        # x is the relative distance between the leader and the follower.
        pos, vel = state

        # update velocity by integrating the new acceleration over time --
        # vel = acc*t + vel_0, pos = acc*t^2/2 + vel_0*t + pos_0
        t = self.TIME_STEP
        
        # Determine new acceleration based upon the chosen action.
        acc = self._acc_from_action(action)
        #print "Choice of acceleration is: " , acc * self.TIME_STEP , " m/s"

        # x'=v,v'=a
        pos_0 = pos
        vel_0 = vel
        vel = acc*t + vel_0
        pos = acc*t**2/2 + vel_0*t + pos_0

        self.state = (pos, vel)
        #print "[env/acc.py] state after _step is: ", self.state

        done = self.is_crash(self.state) or self.state[0] > self.MAX_VALUE
        done = bool(done)

        if not done:
            reward = 1.0
        elif done and self.state[0] <= 1:
            reward = -100.0
        elif done and self.state[0] > self.MAX_VALUE - 0.5:
            reward = -100.0
        else:
            assert False, "Not sure why this should happen, and when it was previously there was a bug in the if/elif guards..."
            reward = 0.0

        return np.array(self.state), reward, done, {'crash': self.state[0] <= 0}

    def _reset(self):
        pos = self.np_random.uniform(low=5, high=75, size=(1,))[0]
        vel = 0
        self.state = (pos, vel)
        #print "Starting separated by ", pos, " meters moving at ", vel, " m/s."

        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 1000
        screen_height = 800

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 5.0
        cartheight = 30.0

        relativeDistance = cartwidth * 2

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
           
            # Add a follower cart.
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            # Add a leader cart
            l,r,t,b = -cartwidth/2 + relativeDistance, cartwidth/2 + relativeDistance, cartheight/2, -cartheight/2
            cart2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans2 = rendering.Transform()
            cart2.add_attr(self.carttrans2)
            self.viewer.add_geom(cart2)
            
            # Display a track.
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            #TODO screen_width - 100 = fixed position of the leader car.
            self.max_line = rendering.Line((screen_width - 100 - self.MAX_VALUE, 0), (screen_width - 100 - self.MAX_VALUE, 200))
            self.max_line.set_color(0,0,0)
            self.viewer.add_geom(self.max_line)

        if self.state is None: return None

        relativeDistance, relativeVelocity = self.state
        followerx = screen_width - 100 - relativeDistance
        leaderx = screen_width - 100
        #cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(followerx, carty)
        self.carttrans2.set_translation(leaderx, carty)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
