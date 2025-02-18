import torch
import numpy as np
from .policy import ParametricPolicy
from collections import deque
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

# from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
# from lerobot.common.policies.normalize import Normalize, Unnormalize
# from lerobot.common.policies.pretrained import PreTrainedPolicy
# from lerobot.common.policies.utils import populate_queues

def populate_queues(queues, batch):
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


class DiffusionPolicy(ParametricPolicy):
    """
    Diffusion-based policy, as used in:

    "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion".
    Chi et al.. 2023.
    Code reference: https://github.com/huggingface/lerobot

    """
    def __init__(self, policy_params, policy_state_shape=None, 
                 draw_random_act=False, draw_deterministic=False,
                 squash_actions=False, discrete_action_dims=0, continuous_action_dims=0,
                 normalize_states=False, normalize_actions=False):
        """
        Constructor.

        Args:
            policy_params (dict): config parameters of the diffusion policy.
            draw_random_act (bool, False): if True, the policy will draw random actions.
            draw_deterministic (bool, False): if True, the policy will draw deterministic actions.
            squash_actions (bool, False): if True, the actions will be squashed to [-1, 1] with a tanh function.
            discrete_action_dims (int, 0): the number of discrete actions in the action space. # Unused for now
            continuous_action_dims (int, 0): the number of continuous actions in the action space. # Unused for now
            normalize_states (bool, False): if True, the states will be normalized before being passed to the regressor.
            normalize_actions (bool, False): if True, the actions will be normalized before being returned.

        """
        super().__init__(policy_state_shape) # TODO: Needed?

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None
        self.config = policy_params
        self._model = policy_params['model_class'](self.config)
        self.reset()

        # self._approximator = mu
        # self._predict_params = dict()
        # self._chol_sigma = torch.linalg.cholesky(sigma)
        self._low = torch.as_tensor(self.config['low'])
        self._high = torch.as_tensor(self.config['high'])
        self._draw_random_act = draw_random_act
        self._draw_deterministic = draw_deterministic
        self._squash_actions = squash_actions
        self._discrete_action_dims = discrete_action_dims
        self._continuous_action_dims = continuous_action_dims
        self._normalize_states = normalize_states
        self._states_mean = None # will be set by the agent class
        self._states_std = None # will be set by the agent class
        self._normalize_actions = normalize_actions
        self._actions_mean = None # will be set by the agent class
        self._actions_std = None # will be set by the agent class

        # debug
        self.debug_replay_states = None
        self.debug_replay_actions = None
        self.debug_replay_index = 0
        self.debug_action_diffs = []

        self._add_save_attr(
            # _model='mushroom',
            # _predict_params='pickle',
            # _chol_sigma='torch',
            _low='torch',
            _high='torch',
            _draw_random_act='primitive',
            _draw_deterministic='primitive',
            _squash_actions='primitive',
            _discrete_action_dims='primitive',
            _continuous_action_dims='primitive',
            _normalize_states='primitive',
            _states_mean='primitive',
            _states_std='primitive',
            _normalize_actions='primitive',
            _actions_mean='primitive',
            _actions_std='primitive',
            debug_replay_states='primitive',
            debug_replay_actions='primitive',
            debug_replay_index='primitive',
            debug_action_diffs='primitive'
        )

    def __call__(self, state, action=None, policy_state=None):
        raise NotImplementedError

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config['n_obs_steps']),
            "action": deque(maxlen=self.config['n_action_steps']),
        }
        if self.config['image_features']:
            raise NotImplementedError("Image features are not implemented yet")
            self._queues["observation.images"] = deque(maxlen=self.config['n_obs_steps'])
        if self.config['env_state_feature']:
            raise NotImplementedError("Environment state feature is not implemented yet")
            self._queues["observation.environment_state"] = deque(maxlen=self.config['n_obs_steps'])

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        # batch = self.normalize_inputs(batch) # Assuming inputs are normalized already if needed
        if self.config['image_features']:
            raise NotImplementedError("Image features are not implemented yet")
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config['image_features']], dim=-4
            )
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self._model.generate_actions(batch)

            # TODO(rcadene): make above methods return output dictionary?
            # actions = self.unnormalize_outputs({"action": actions})["action"]
            # TODO: check if action normalization/unnormalization is needed

            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor], squash_actions: bool) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        # batch = self.normalize_inputs(batch) # Assuming inputs are normalized already if needed
        if self.config['image_features']:
            raise NotImplementedError("Image features are not implemented yet")
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config['image_features']], dim=-4
            )
        # batch = self.normalize_targets(batch)
        # TODO: check if action normalization/unnormalization is needed
        loss = self._model.compute_loss(batch, squash_actions)
        return {"loss": loss}

    def draw_action(self, state, policy_state=None):
        
        if state.type == 'numpy': # TODO test
            state = torch.tensor(state)
        
        if self._draw_random_act is True:
            return self.draw_random_action()
        elif self._draw_deterministic is True:
            return self.draw_deterministic_action(state, policy_state)
        
        with torch.no_grad():
            if self._normalize_states:
                if self._states_mean is None:
                    raise ValueError('States mean is not set by the agent class')
                state_query = (state - self._states_mean) / self._states_std
            else:
                state_query = state
                
            input_batch = {'observation.state': state_query}
            mu = self.select_action(input_batch).cpu()
            # mu = self._model.predict(state_query, **self._predict_params).cpu()
            # mu = np.reshape(self._approximator.predict(np.expand_dims(state_query, axis=0), **self._predict_params), -1)
            if self._squash_actions:
                # Squash the continuous actions to [-1, 1]
                mu[-self._continuous_action_dims:] = torch.tanh(mu[-self._continuous_action_dims:])

            if self._normalize_actions:
                if self._actions_mean is None:
                    raise ValueError('Actions mean is not set by the agent class')
                mu = mu * self._actions_std + self._actions_mean

            # sample continuous actions from distribution
            distribution = torch.distributions.MultivariateNormal(loc=mu[-self._continuous_action_dims:], scale_tril=self._chol_sigma,
                                                                  validate_args=False)
            action_raw = distribution.sample()

            if self._discrete_action_dims > 0:
                # discrete actions from network are logits, so sigmoid them
                action_disc = torch.sigmoid(mu[:self._discrete_action_dims])
                action = torch.cat((action_disc, action_raw), dim=0)
                # print("action: ", torch.round(action*100)/100)
            else:
                action = action_raw

            return torch.clip(action, self._low, self._high), None
    
    def draw_random_action(self):
        return torch.rand(self._low.shape) * (self._high - self._low) + self._low, None

    def draw_deterministic_action(self, state, policy_state=None):
        # fix if loading is not on the correct device
        if self._low.device != state.device:
            self._low = self._low.to(state.device)
            self._high = self._high.to(state.device)
            self._chol_sigma = self._chol_sigma.to(state.device)
        with torch.no_grad():
            if self._normalize_states:
                if self._states_mean is None:
                    raise ValueError('States mean is not set by the agent class')
                state_query = (state - self._states_mean) / self._states_std
            else:
                state_query = state
            
            input_batch = {'observation.state': state_query}
            mu = self.select_action(input_batch).cpu()
            # mu = self._model.predict(state_query, **self._predict_params).cpu()
            
            if self._squash_actions:
                # Squash the continuous actions to [-1, 1]
                mu[-self._continuous_action_dims:] = torch.tanh(mu[-self._continuous_action_dims:])

            if self._normalize_actions:
                if self._actions_mean is None:
                    raise ValueError('Actions mean is not set by the agent class')
                mu = mu * self._actions_std + self._actions_mean
            
            action_raw = mu[-self._continuous_action_dims:]

            if self._discrete_action_dims > 0:
                # discrete actions from network are logits, so sigmoid them
                action_disc = torch.sigmoid(mu[:self._discrete_action_dims])
                action = torch.cat((action_disc, action_raw), dim=0)
                # print("action: ", torch.round(action*100)/100)
            else:
                action = action_raw

            action_clipped = torch.clip(action, self._low, self._high)
            
            return action_clipped, None
    
    def set_weights(self, weights):
        self._model.set_weights(weights)

    def get_weights(self):
        return self._model.get_weights()

    @property
    def weights_size(self):
        return self._model.weights_size