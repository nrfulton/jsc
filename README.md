# Install

You'll need to symlink our environments into you OpenAI Gym installation.

First, install [OpenAI Gym](https://github.com/openai/gym). You may need to checkout an older version of the gym:

`git checkout 52e803f36b3281a575bf98255947a5a74d3c59a0`

Once installed (using `pip`), `cd` into your installation and `ln` the new `env/` file into the OpenAI gym install directory:

    cd /path/to/openaigym/gym/envs/safety;
    ln -s /path/to/this/dir/env/acc.py acc.py

Also register this new environment by adding the following lines to `gym/envs/__init__.py`:

      register(
          id='acc-v0',
          entry_point='gym.envs.safety:ACCEnv',
          max_episode_steps=200,  # todo edit
          reward_threshold=195.0, # todo edit
      )

And add the appropriate import to `gym/envs/safety/__init__.py`:

    from gym.envs.safety.acc import ACCEnv

Finally, add a new method to `gym/spaces/Discrete.py`:

      def sample_restricted(self, filter_fn):
        """ Sample from a subset of the space. """
        subset = filter(filter_fn, range(0,self.n))
        idx = prng.np_random.randint(len(subset))
        return subset[idx]

# Copyright and License

Nathan Fulton 2017. GPLv2
