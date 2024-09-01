import torch
import numpy as np

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.rl_utils.parameters import Parameter, to_parameter
from mushroom_rl.utils.torch import TorchUtils
from tqdm import trange


class BC(DeepAC):
    """
    BEHAVIOR CLONING algorithm that builds on top of the DeepAC class.
    Even though the algorithm is not an actor-critic method, it is implemented
    here to compare against other methods such as TD3+BC.
    Normally uses ClippedGaussianPolicy

    """
    def __init__(self, mdp_info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params=None,
                 batch_size=1, n_epochs_policy=1, patience=1, squash_actions=False,
                 critic_fit_params=None, actor_predict_params=None, critic_predict_params=None):
        """
        Constructor.

        Args:
            policy_class (Policy): class of the policy;
            policy_params (dict): parameters of the policy to build;
            actor_params (dict): parameters of the actor approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): Unused parameter; Left for future
            batch_size ([int, Parameter]): size of minibatches for every optimization step
            n_epochs_policy ([int, Parameter]): number of policy update epochs on the whole dataset at every call;
            patience (float, 1.): (Optional) the number of epochs to wait until stop
                the learning if not improving;
            squash_actions (bool, True): (Optional) whether to squash the actions to [-1, 1] with tanh;
            critic_fit_params (dict, None): Unused parameter; Left for future
            actor_predict_params (dict, None): Unused parameter; Left for future
            critic_predict_params (dict, None): Unused parameter; Left for future

        """

        self._actor_predict_params = dict() if actor_predict_params is None else actor_predict_params

        self._actor_approximator = Regressor(TorchApproximator, **actor_params)

        policy = policy_class(self._actor_approximator, **policy_params)

        policy_parameters = self._actor_approximator.model.network.parameters()

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

        self._batch_size = to_parameter(batch_size)
        self._n_epochs_policy = to_parameter(n_epochs_policy)
        self._patience = to_parameter(patience)
        self._squash_actions = squash_actions

        self._bc_loss_fn = torch.nn.MSELoss()
        self._fit_count = 0
        self._actor_last_loss = None # Store actor loss for logging

        self._add_save_attr(
            _batch_size='mushroom',
            _n_epochs_policy='mushroom',
            _patience='mushroom',
            _squash_actions='primitive',
            _actor_approximator='mushroom',
            _fit_count='primitive',
        )

    def fit(self, demo_dataset, n_epochs=None):
        if n_epochs is None:
            n_epochs = self._n_epochs_policy()
        
        # fit on the dataset (for n_epochs)
        for epoch in trange(n_epochs):
            for obs, act in minibatch_generator(self._batch_size(), demo_dataset['obs'], demo_dataset['action']):
                loss = self._loss(obs, act)
                self._optimize_actor_parameters(loss)

                self._fit_count += 1

                # early stopping: (Optional)
                # check loss reduction, self._patience

        self._actor_last_loss = loss.detach().cpu().numpy() # Store actor loss for logging
    
    def _loss(self, obs, act):
        # loss for behavior cloning
        
        # if isinstance(obs, np.ndarray):
        #     obs = torch.as_tensor(obs, dtype=torch.float32)
        act = torch.as_tensor(act, dtype=torch.float32, device=TorchUtils.get_device())

        # TODO: Handle hybrid policy
        act_pred = self._actor_approximator(obs, **self._actor_predict_params)
        if self._squash_actions:
            # Squash the actions to [-1, 1] (Needed if RL policy squashes actions)
            act_pred = torch.tanh(act_pred)

        return self._bc_loss_fn(act_pred, act) # normally mse loss
        

    def _post_load(self):
        self._actor_approximator = self.policy._approximator
        self._update_optimizer_parameters(self._actor_approximator.model.network.parameters())
