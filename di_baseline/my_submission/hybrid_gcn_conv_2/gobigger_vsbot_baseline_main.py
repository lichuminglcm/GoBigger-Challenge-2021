import os
import numpy as np
import copy
from tensorboardX import SummaryWriter
import sys
import time
import torch
sys.path.append('..')

from ding.config import compile_config
from ding.worker import BaseLearner, BattleSampleSerialCollector, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import SyncSubprocessEnvManager
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from gobigger.agents import BotAgent

from envs import GoBiggerHybridEnv
from model import GoBiggerHybridAction
from policy import GoBiggerPolicy
from config.gobigger_hybrid_action_gcn_conv_config_1 import main_config

def action_space(angle_num):
    angle = 2 * np.pi * np.arange(angle_num) / angle_num

    angle_direction = np.stack([np.cos(angle), np.sin(angle)], axis=1) #[angle_num, 2]
    direction_action = np.stack([angle_direction, angle_direction, angle_direction, angle_direction], axis=0) #[4, angle_num, 2]

    discrete_action = np.zeros((4, angle_num, 4))
    discrete_action[np.array([0, 1, 2, 3]), :, np.array([0, 1, 2, 3])] = 1

    action = np.concatenate([direction_action, discrete_action], axis=2).reshape(4 * angle_num, 6)
    return action

class RandomPolicy:

    def __init__(self, angle_num: int, player_num: int):
        self.action_space = action_space(angle_num)
        self.player_num = player_num

    def forward(self, data: dict) -> dict:
        return {
            env_id: {
                'action': self.action_space[np.random.randint(0, len(self.action_space), size=(self.player_num))]
            }
            for env_id in data.keys()
        }

    def reset(self, data_id: list = []) -> None:
        pass

def act_inverse_transform(act):
    action = np.zeros((6,))
    if act[0] is not None and act[1] is not None:
        action[:2] = np.array(act[:2])
    action[2 + int(act[2] + 1)] = 1
    return action

class RulePolicy:

    def __init__(self, team_id: int, player_num_per_team: int):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = [BotAgent(str(i)) for i in range(start, end)]

    def forward(self, data: dict, **kwargs) -> dict:
        ret = {}
        for env_id in data.keys():
            action = []
            for bot, raw_obs in zip(self.bot, data[env_id]['collate_ignore_raw_obs']):
                action.append(act_inverse_transform(bot.step(raw_obs)))
            ret[env_id] = {'action': np.array(action)}
        return ret

    def reset(self, data_id: list = []) -> None:
        pass


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        GoBiggerPolicy,
        BaseLearner,
        BattleSampleSerialCollector,
        BattleInteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env_cfg = copy.deepcopy(cfg.env)
    collector_env_cfg.train = True
    evaluator_env_cfg = copy.deepcopy(cfg.env)
    evaluator_env_cfg.train = False
    collector_env = SyncSubprocessEnvManager(
        env_fn=[lambda: GoBiggerHybridEnv(collector_env_cfg) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    random_evaluator_env = SyncSubprocessEnvManager(
        env_fn=[lambda: GoBiggerHybridEnv(evaluator_env_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )
    rule_evaluator_env = SyncSubprocessEnvManager(
        env_fn=[lambda: GoBiggerHybridEnv(evaluator_env_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    collector_env.seed(seed)
    random_evaluator_env.seed(seed, dynamic_seed=False)
    rule_evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = GoBiggerHybridAction(**cfg.policy.model)
    # model = torch.nn.DataParallel(model)
    policy = GoBiggerPolicy(cfg.policy, model=model)
    team_num = cfg.env.team_num
    rule_collect_policy = [RulePolicy(team_id, cfg.env.player_num_per_team) for team_id in range(1, team_num)]
    random_eval_policy = RandomPolicy(
        4, cfg.env.player_num_per_team
    )
    rule_eval_policy = [RulePolicy(team_id, cfg.env.player_num_per_team) for team_id in range(1, team_num)]
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(
        cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name, instance_name='learner'
    )
    collector = BattleSampleSerialCollector(
        cfg.policy.collect.collector,
        collector_env, [policy.collect_mode] + rule_collect_policy,
        tb_logger,
        exp_name=cfg.exp_name
    )
    random_evaluator = BattleInteractionSerialEvaluator(
        cfg.policy.eval.evaluator,
        random_evaluator_env, [policy.eval_mode] + [random_eval_policy for _ in range(team_num - 1)],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='random_evaluator'
    )
    rule_evaluator = BattleInteractionSerialEvaluator(
        cfg.policy.eval.evaluator,
        rule_evaluator_env, [policy.eval_mode] + rule_eval_policy,
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='rule_evaluator'
    )
    replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, exp_name=cfg.exp_name)

    for _ in range(max_iterations):
        if random_evaluator.should_eval(learner.train_iter + 1):
            random_stop_flag, random_reward, _ = random_evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep
            )
            rule_stop_flag, rule_reward, _ = rule_evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep
            )
            if random_stop_flag and rule_stop_flag:
                break
        eps = epsilon_greedy(collector.envstep)
        # Sampling data from environments
        # t1 = time.time()
        # print('begin collecting')
        new_data, _ = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        # t2 = time.time()
        # print( 'collect time: {:.4f}'.format( (t2 - t1) / len(new_data[0]) ) )
        replay_buffer.push(new_data[0], cur_collector_envstep=collector.envstep)
        # t1 = time.time()
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            learner.train(train_data, collector.envstep)
        # t2 = time.time()
        # print( 'train time: {:.4f}'.format( (t2 - t1) / cfg.policy.learn.update_per_collect ) )


if __name__ == "__main__":
    main(main_config)
