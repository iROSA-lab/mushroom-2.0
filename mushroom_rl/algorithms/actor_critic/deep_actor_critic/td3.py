import numpy as np
import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DDPG
from mushroom_rl.policy import Policy
from mushroom_rl.rl_utils.parameters import to_parameter


class TD3(DDPG):
    """
    Twin Delayed DDPG algorithm.
    "Addressing Function Approximation Error in Actor-Critic Methods".
    Fujimoto S. et al.. 2018.

    """
    def __init__(self, mdp_info, policy_class, policy_params, actor_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, tau, policy_delay=2,
                 noise_std=.2, noise_clip=.5, squash_actions=False, critic_fit_params=None):
        """
        Constructor.

        Args:
            policy_class (Policy): class of the policy;
            policy_params (dict): parameters of the policy to build;
            actor_params (dict): parameters of the actor approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ([int, Parameter]): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau ([float, Parameter]): value of coefficient for soft updates;
            policy_delay ([int, Parameter], 2): the number of updates of the critic after
                which an actor update is implemented;
            noise_std ([float, Parameter], .2): standard deviation of the noise used for
                policy smoothing;
            noise_clip ([float, Parameter], .5): maximum absolute value for policy smoothing
                noise;
            squash_actions (bool, False): whether to squash the actions to [-1, 1] with tanh;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._noise_std = to_parameter(noise_std)
        self._noise_clip = to_parameter(noise_clip)
        self._squash_actions = squash_actions

        if 'n_models' in critic_params.keys():
            assert(critic_params['n_models'] >= 2)
        else:
            critic_params['n_models'] = 2

        super().__init__(mdp_info, policy_class, policy_params, actor_params, actor_optimizer, critic_params,
                         batch_size, initial_replay_size, max_replay_size, tau, policy_delay, critic_fit_params)

        self._add_save_attr(
            _noise_std='mushroom',
            _noise_clip='mushroom',
            _squash_actions='primitive'
        )

    def _loss(self, state):
        action = self._actor_approximator(state, **self._actor_predict_params)
        if self._squash_actions:
            action = torch.tanh(action) # squash action to [-1, 1]
        q = self._critic_approximator(state, action, idx=0, **self._critic_predict_params)

        return -q.mean()

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a = self._target_actor_approximator(next_state, **self._actor_predict_params)
        if self._squash_actions:
            a = torch.tanh(a) # squash action to [-1, 1]
        a = a.detach().cpu().numpy() # TODO: Handle without casting to numpy

        low = self.mdp_info.action_space.low
        high = self.mdp_info.action_space.high
        eps = np.random.normal(scale=self._noise_std(), size=a.shape)
        eps_clipped = np.clip(eps, -self._noise_clip(), self._noise_clip.get_value())
        a_smoothed = np.clip(a + eps_clipped, low, high)

        q = self._target_critic_approximator.predict(next_state, a_smoothed,
                                                     prediction='min', **self._critic_predict_params)
        q = q.detach().cpu()
        q *= (~absorbing)

        return q
