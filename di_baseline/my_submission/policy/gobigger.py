from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Categorical

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from ding.policy.base_policy import Policy
from ding.policy.common_utils import default_preprocess_learn

def gobigger_collate(data):
    r"""
    Arguments:
        - data (:obj:`list`): Lsit type data, [{scalar:[player_1_scalar, player_2_scalar, ...], ...}, ...]
    """
    B, player_num_per_team = len(data), len(data[0]['scalar'])
    data = {k: sum([d[k] for d in data], []) for k in data[0].keys() if not k.startswith('collate_ignore')}
    clone_num = max([x.shape[0] for x in data['clone']])
    thorn_num = max([x.shape[1] for x in data['thorn_relation']])
    food_h = max([x.shape[1] for x in data['food']])
    food_w = max([x.shape[2] for x in data['food']])
    data['scalar'] = torch.stack([torch.as_tensor(x) for x in data['scalar']]).float()
    data['action_space'] = torch.stack([torch.as_tensor(x) for x in data['action_space']]).float()
    data['food'] = torch.stack([F.pad(torch.as_tensor(x), (0, food_w - x.shape[2], 0, food_h - x.shape[1])) for x in data['food']]).float()
    data['food_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[0])) for x in data['food_relation']]).float()
    data['thorn_mask'] = torch.stack([torch.arange(thorn_num) < x.shape[1] for x in data['thorn_relation']]).float()
    data['thorn_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, thorn_num - x.shape[1], 0, clone_num - x.shape[0])) for x in data['thorn_relation']]).float()
    data['clone_mask'] = torch.stack([torch.arange(clone_num) < x.shape[0] for x in data['clone']]).float()
    data['clone'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[0])) for x in data['clone']]).float()
    data['clone_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[1], 0, clone_num - x.shape[0])) for x in data['clone_relation']]).float()
    return B, player_num_per_team, data

def gobigger_preprocess_learn(
        data: List[Any],
        use_priority_IS_weight: bool = False,
        use_priority: bool = False,
        use_nstep: bool = False,
        ignore_done: bool = False,
) -> dict:
    # data collate
    obs = [d['obs'] for d in data]
    next_obs = [d['next_obs'] for d in data]
    for i in range(len(data)):
        data[i] = {k: v for k, v in data[i].items() if not 'obs' in k}
    data = default_collate(data)
    B, player_num_per_team, data['obs'] = gobigger_collate(obs)
    B, player_num_per_team, data['next_obs'] = gobigger_collate(next_obs)
    # data process

    data['action'] = data['action'].reshape(-1, *data['action'].shape[2:])

    done = data['done']
    if ignore_done:
        done = torch.zeros_like(done).float()
    else:
        done = done.float()
    done = done.reshape(-1, 1).repeat(1, player_num_per_team).reshape(-1)
    data['done'] = done

    if use_priority_IS_weight:
        assert use_priority, "Use IS Weight correction, but Priority is not used."
    if use_priority and use_priority_IS_weight:
        weight = data['IS']
    else:
        weight = data.get('weight', None)
    if weight is not None:
        weight = weight.reshape(-1, 1).repeat(1, player_num_per_team).reshape(-1)
    data['weight'] = weight

    reward = data['reward']
    if use_nstep:
        # Reward reshaping for n-step
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        # reward: (batch_size, nstep) -> (nstep, batch_size)
        reward = reward.permute(1, 0).contiguous()
    shape = reward.shape
    reward = reward.reshape(-1, 1).repeat(1, player_num_per_team).reshape(*shape[:-1], -1)
    data['reward'] = reward

    return B, player_num_per_team, data


@POLICY_REGISTRY.register('gobigger')
class GoBiggerPolicy(Policy):
    r"""
       Overview:
           Policy class of SAC algorithm.

       Config:
           == ====================  ========    =============  ================================= =======================
           ID Symbol                Type        Default Value  Description                       Other(Shape)
           == ====================  ========    =============  ================================= =======================
           1  ``type``              str         td3            | RL policy register name, refer  | this arg is optional,
                                                               | to registry ``POLICY_REGISTRY`` | a placeholder
           2  ``cuda``              bool        True           | Whether to use cuda for network |
           3  | ``random_``         int         10000          | Number of randomly collected    | Default to 10000 for
              | ``collect_size``                               | training samples in replay      | SAC, 25000 for DDPG/
              |                                                | buffer when training starts.    | TD3.
           4  | ``model.policy_``   int         256            | Linear layer size for policy    |
              | ``embedding_size``                             | network.                        |
           5  | ``model.soft_q_``   int         256            | Linear layer size for soft q    |
              | ``embedding_size``                             | network.                        |
           6  | ``model.value_``    int         256            | Linear layer size for value     | Defalut to None when
              | ``embedding_size``                             | network.                        | model.value_network
              |                                                |                                 | is False.
           7  | ``learn.learning``  float       3e-4           | Learning rate for soft q        | Defalut to 1e-3, when
              | ``_rate_q``                                    | network.                        | model.value_network
              |                                                |                                 | is True.
           8  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to 1e-3, when
              | ``_rate_policy``                               | network.                        | model.value_network
              |                                                |                                 | is True.
           9  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to None when
              | ``_rate_value``                                | network.                        | model.value_network
              |                                                |                                 | is False.
           10 | ``learn.alpha``     float       0.2            | Entropy regularization          | alpha is initiali-
              |                                                | coefficient.                    | zation for auto
              |                                                |                                 | `\alpha`, when
              |                                                |                                 | auto_alpha is True
           11 | ``learn.repara_``   bool        True           | Determine whether to use        |
              | ``meterization``                               | reparameterization trick.       |
           12 | ``learn.``          bool        False          | Determine whether to use        | Temperature parameter
              | ``auto_alpha``                                 | auto temperature parameter      | determines the
              |                                                | `\alpha`.                       | relative importance
              |                                                |                                 | of the entropy term
              |                                                |                                 | against the reward.
           13 | ``learn.-``         bool        False          | Determine whether to ignore     | Use ignore_done only
              | ``ignore_done``                                | done flag.                      | in halfcheetah env.
           14 | ``learn.-``         float       0.005          | Used for soft update of the     | aka. Interpolation
              | ``target_theta``                               | target network.                 | factor in polyak aver
              |                                                |                                 | aging for target
              |                                                |                                 | networks.
           == ====================  ========    =============  ================================= =======================
       """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='gobigger',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in SAC.
        on_policy=False,
        # (bool type) priority: Determine whether to use priority in buffer sample.
        # Default False in SAC.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 10000 in SAC.
        random_collect_size=10000,
        model=dict(
        ),
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=4,
            # (int) Minibatch size for gradient descent.
            batch_size=128,
            # (float type) learning_rate_q: Learning rate for soft q network.
            # Default to 3e-4.
            # Please set to 1e-3, when model.value_network is True.
            learning_rate_q=3e-4,
            # (float type) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            ignore_done=False,
        ),
        collect=dict(
            # You can use either "n_sample" or "n_episode" in actor.collect.
            # Get "n_sample" samples per collect.
            # Default n_sample to 1.
            n_sample=128,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # (float type) alpha: Entropy regularization coefficient.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # If auto_alpha is set  to `True`, alpha is initialization for auto `\alpha`.
            # Default to 1.0.
            alpha=1.0,
        ),
        eval=dict(),
        other=dict(
            replay_buffer=dict(
                # (int type) replay_buffer_size: Max size of replay buffer.
                replay_buffer_size=20000,
                # (int type) max_use: Max use times of one data in the buffer.
                # Data will be removed once used for too many times.
                # Default to infinite.
                # max_use=256,
            ),
        ),
    )
    r"""
    Overview:
        Policy class of SAC algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target models.
        """
        # Init
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight

        # Optimizers
        self._optimizer_q = Adam(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate_q,
        )

        # Algorithm config
        self._gamma = self._cfg.learn.discount_factor

        # Main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr, loss, target_q_value and other \
                running information.
        """
        loss_dict = {}
        B, player_num_per_team, data = gobigger_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        obs = data['obs']
        action = data['action']
        reward = data['reward']
        next_obs = data['next_obs']
        done = data['done']
        weight = data['weight']

        # 0. encoder obs
        encode = self._learn_model.forward(obs, mode='compute_encoder')['encode']
        with torch.no_grad():
            next_encode = self._target_model.forward(next_obs, mode='compute_encoder')['encode']

        # 1. predict q value
        q_value = self._learn_model.forward({'encode':encode, 'action':action}, mode='compute_critic')['q_value']
        with torch.no_grad():
            target_q_value = self._target_model.forward({'encode':next_encode, 'action':next_obs['action_space']}, mode='compute_critic')['q_value'].max(-1).values
        # 2. compute q loss
        q_data = v_1step_td_data(q_value, target_q_value, reward, done, weight)
        loss_dict['critic_loss'], td_error_per_sample = v_1step_td_error(q_data, self._gamma)

        # 3. update q network
        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward()
        self._optimizer_q.step()

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.detach().mean().item(),
            'q_value': q_value.detach().mean().item(),
            **loss_dict
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        ret = {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_q': self._optimizer_q.state_dict(),
        }
        return ret

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_q.load_state_dict(state_dict['optimizer_q'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
            Use action noise for exploration.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='base')
        self._collect_model.reset()
        self._alpha = self._cfg.collect.alpha

    def _forward_collect(self, data: dict, eps: float) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        B, player_num_per_team, data = gobigger_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)

        self._collect_model.eval()

        with torch.no_grad():
            encode = self._collect_model.forward(data, mode='compute_encoder')['encode']
            q_value = self._collect_model.forward({'encode':encode, 'action':data['action_space']}, mode='compute_critic')['q_value']
            dist = Categorical(logits=self._alpha * q_value)
            action_sample = dist.sample()
            action_argmax = q_value.max(-1).indices
            mask = (torch.rand(len(action_sample)) > eps).to(action_sample)
            action = action_sample * (1 - mask) + action_argmax * mask
            action = data['action_space'][torch.arange(action.shape[0]), action]
            output = {'action': action.view(B, player_num_per_team, -1)}

        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, policy_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - policy_output (:obj:`dict`): Output of policy collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': policy_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model. Unlike learn and collect model, eval model does not need noise.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        B, player_num_per_team, data = gobigger_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)

        self._collect_model.eval()

        with torch.no_grad():
            encode = self._collect_model.forward(data, mode='compute_encoder')['encode']
            q_value = self._collect_model.forward({'encode':encode, 'action':data['action_space']}, mode='compute_critic')['q_value']
            action = q_value.max(-1).indices
            action = data['action_space'][torch.arange(action.shape[0]), action]
            output = {'action': action.view(B, player_num_per_team, -1)}

        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        return 'qac', ['ding.model.template.qac']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return [
            'critic_loss',
            'cur_lr_q',
            'q_value',
            'td_error',
        ]
