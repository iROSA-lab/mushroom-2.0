import numpy as np
import torch
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.rl_utils.replay_memory import ReplayMemory
from mushroom_rl.rl_utils.parameters import to_parameter

from mushroom_rl.core.dataset import Dataset
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import TorchUtils
from tqdm import trange
from copy import deepcopy

class IQL(DeepAC):
    """
    IQL Offline-to-Online RL algorithm.
    "Offline Reinforcement Learning with Implicit Q-Learning".
    Kostrikov I. et al.. 2022.
    Reference implementation: https://github.com/corl-team/CORL

    """
    def __init__(self, mdp_info, policy_class, policy_params, actor_params,
                 actor_optimizer, critic_params, value_func_params, value_func_optimizer, 
                 batch_size, initial_replay_size, max_replay_size, tau,
                 squash_actions=False, normalize_states=False, schedule_actor_lr=False,
                 iql_beta=1.0, iql_tau=0.7, max_clamp_adv=100.0,
                 critic_fit_params=None, actor_predict_params=None, critic_predict_params=None):
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
            value_func_params (dict): parameters of the value function approximator to build;
            value_func_optimizer (dict): parameters to specify the value function
                optimizer algorithm;
            batch_size ([int, Parameter]): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau ([float, Parameter]): value of coefficient for soft updates;
            squash_actions (bool, False): whether to squash the actions to [-1, 1] with tanh;
            normalize_states (bool, False): whether to normalize states;
            schedule_actor_lr (bool, False): whether to use a learning rate scheduler for the actor;
            iql_beta ([float, Parameter], 0.25): Inverse temperature. Small beta -> BC, big beta -> maximizing Q; when fitting on the offline dataset;
            iql_tau ([float, Parameter], 0.7): Coefficient for the asymmetric IQL loss;
            max_clamp_adv ([float, Parameter], 100.0): Maximum value considered for the advantage;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator;
            actor_predict_params (dict, None): parameters for the prediction with the
                actor approximator;
            critic_predict_params (dict, None): parameters for the prediction with the
                critic approximator.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params
        self._actor_predict_params = dict() if actor_predict_params is None else actor_predict_params
        self._critic_predict_params = dict() if critic_predict_params is None else critic_predict_params

        if 'n_models' in critic_params.keys():
            assert(critic_params['n_models'] >= 2)
        else:
            critic_params['n_models'] = 2
        
        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator, **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator, **target_critic_params)

        # Add IQL value function approximator & optimizer
        # assert value_func_params['n_models'] == 1 # Single model
        self._value_func_approximator = Regressor(TorchApproximator, **value_func_params)
        value_func_network_params = self._value_func_approximator.model.network.parameters()
        self._value_func_optimizer = value_func_optimizer['class'](value_func_network_params, **value_func_optimizer['params'])

        self._actor_approximator = Regressor(TorchApproximator, **actor_params)

        self._init_target(self._critic_approximator, self._target_critic_approximator)

        policy = policy_class(self._actor_approximator, **policy_params)

        policy_parameters = self._actor_approximator.model.network.parameters()

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

        self._batch_size = to_parameter(batch_size)
        self._tau = to_parameter(tau)
        self._fit_count = 0
        self._actor_last_loss = None # Store actor loss for logging
        self._value_last_loss = None # Store value loss for logging
        self._last_exp_adv = None # Store exp_adv for logging

        self._replay_memory = ReplayMemory(mdp_info, self.info, initial_replay_size, max_replay_size)

        self._squash_actions = squash_actions
        self._normalize_states = normalize_states
        if self._normalize_states:
            self.states_mean = None
            self.states_std = None
        self._schedule_actor_lr = schedule_actor_lr
        if self._schedule_actor_lr:
            max_steps = (max_replay_size * 100) // batch_size # heuristic. TODO: test
            self._actor_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, max_steps)
        else:
            self._actor_lr_scheduler = None

        self._iql_beta = to_parameter(iql_beta)
        self._iql_tau = to_parameter(iql_tau)
        self._max_clamp_adv = max_clamp_adv
        
        self.offline_dataset = None

        self._add_save_attr(
            _critic_fit_params='pickle',
            _critic_predict_params='pickle',
            _actor_predict_params='pickle',
            _batch_size='mushroom',
            _tau='mushroom',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _value_func_approximator='mushroom',
            _value_func_optimizer='torch',
            _actor_approximator='mushroom',
            _squash_actions='primitive',
            _normalize_states='primitive',
            _schedule_actor_lr='primitive',
            _actor_lr_scheduler='torch',
            _iql_beta='mushroom',
            _iql_tau='mushroom',
        )
    
    def load_dataset(self, dataset):
        # load & create mushroom dataset
        self.offline_dataset = Dataset.from_array(dataset['obs'], dataset['action'], dataset['reward'],
                                                  dataset['next_obs'], dataset['absorbing'], dataset['last'],
                                                  backend='torch')
        # copy it over to the replay buffer
        self._replay_memory._initial_size = len(self.offline_dataset) # set initial size to the size of the offline dataset
        if self._replay_memory._max_size < len(self.offline_dataset):
            print('[[Warning: Offline dataset size exceeds max replay memory size. Resizing replay memory to fit dataset.]]')
            self._replay_memory._max_size = len(self.offline_dataset)
        self._replay_memory.add(self.offline_dataset)

        if self._normalize_states:
            self._compute_states_mean_std(dataset['obs'])
    
    def offline_fit(self, n_epochs):
        if self.offline_dataset is None:
            raise ValueError('No offline dataset loaded!. Call load_dataset() first.')
        
        # fit on the dataset (for n_epochs)
        for epoch in trange(n_epochs):
            state, action, reward, next_state, absorbing, _ = self._replay_memory.get(self._batch_size())

            if self._normalize_states:
                state_fit = self._norm_states(state)
                next_state_fit = self._norm_states(next_state)
            else:
                state_fit = state
                next_state_fit = next_state

            self.iql_fit(state_fit, action, reward, next_state_fit, absorbing)
    
    def fit(self, dataset): # Online
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = self._replay_memory.get(self._batch_size())

            self.iql_fit(state, action, reward, next_state, absorbing)

    def iql_fit(self, state, action, reward, next_state, absorbing):
        with torch.no_grad():
            next_v = self._value_func_approximator(next_state, **self._critic_predict_params)
            next_v = next_v.cpu()
        # Update value function
        adv = self._update_v(state, action)
        # Update Q function
        self._update_q(next_v, state, action, reward, absorbing)
        # Update actor
        self._update_actor(adv, state, action)

    def _update_v(self, state, action):
        # Update value function
        with torch.no_grad():
            target_q = self._target_critic_approximator.predict(state, action,
                                                prediction='min', **self._critic_predict_params)

        v = self._value_func_approximator(state, **self._critic_predict_params)
        adv = target_q - v

        v_loss = self._asymmetric_l2_loss(adv, self._iql_tau())
        self._value_func_optimizer.zero_grad()
        v_loss.backward()
        self._value_func_optimizer.step()

        self._value_last_loss = v_loss.detach().cpu().numpy() # for logging
        
        return adv
    
    def _update_q(self, next_v, state, action, reward, absorbing):
        # Compute q value
        q = reward + (~absorbing) * self.mdp_info.gamma * next_v

        # Fit critic
        self._critic_approximator.fit(state, action, q, **self._critic_fit_params)      

        # Update target critic
        self._update_target(self._critic_approximator, self._target_critic_approximator)

    def _update_actor(self, adv, state, action):
        # compute advantage weighted BC loss
        exp_adv = torch.exp(self._iql_beta() * adv.detach()).clamp(max=self._max_clamp_adv)
        
        # target action from data:
        act = torch.as_tensor(action, dtype=torch.float32, device=TorchUtils.get_device())
        act_pred = self._actor_approximator(state, **self._actor_predict_params)
        # TODO: Handle hybrid policy
        if self._squash_actions:
            # Squash the actions to [-1, 1] (Needed if RL policy squashes actions)
            act_pred = torch.tanh(act_pred)
        
        # if actions are probablity distribution for categorical
        #   bc_losses = -act_pred.log_prob(actions).sum(-1, keepdim=False)
        bc_loss = torch.sum((act_pred - act) ** 2, dim=1)
        actor_loss = torch.mean(exp_adv * bc_loss)

        self._optimize_actor_parameters(actor_loss)
        
        if self._schedule_actor_lr:
            self._actor_lr_scheduler.step()

        self._actor_last_loss = actor_loss.detach().cpu().numpy() # Store actor loss for logging
        self._last_exp_adv = exp_adv.detach().cpu().numpy() # Store exp_adv for logging

    def _asymmetric_l2_loss(self, u: torch.Tensor, tau: float):
        # loss is just L2 when u is positive, but (1 - tau) * L2 when u is negative.
        # Eg. When tau = 0.7, the loss is only 0.3 * L2 when u is negative.
        return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

    def _compute_states_mean_std(self, states: np.ndarray, eps: float = 1e-3):
        self.states_mean = states.mean(0)
        self.states_std = states.std(0) + eps

    def _norm_states(self, states: np.ndarray):
        if self.states_mean is None or self.states_std is None:
            raise ValueError('States mean and std not computed yet. Call _compute_states_mean_std() on the dataset first.')
        return (states - self.states_mean) / self.states_std

    def _post_load(self):
        self._actor_approximator = self.policy._approximator
        self._update_optimizer_parameters(self._actor_approximator.model.network.parameters())