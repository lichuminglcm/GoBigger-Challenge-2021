from typing import Any, List, Union, Optional, Tuple
import time
import copy
import math
from collections import OrderedDict
import cv2
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.utils import ENV_REGISTRY
from gobigger.server import Server
from gobigger.render import EnvRender


def one_hot_np(value: int, num_cls: int):
    ret = np.zeros(num_cls)
    ret[value] = 1
    return ret


@ENV_REGISTRY.register('gobigger')
class GoBiggerEnv(BaseEnv):
    config = dict(
        player_num_per_team=2,
        team_num=3,
        match_time=1200,
        map_height=1000,
        map_width=1000,
        resize_height=160,
        resize_width=160,
        spatial=False,
        speed = False,
        all_vision = False,
        train=True,
    )

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._player_num_per_team = cfg.player_num_per_team
        self._team_num = cfg.team_num
        self._player_num = self._player_num_per_team * self._team_num
        self._match_time = cfg.match_time
        self._map_height = cfg.map_height
        self._map_width = cfg.map_width
        self._resize_height = cfg.resize_height
        self._resize_width = cfg.resize_width
        self._spatial = cfg.spatial
        self._speed = cfg.speed
        self._all_vision = cfg.all_vision
        self._cfg['obs_settings'] = dict(
                with_spatial=self._spatial,
                with_speed=self._speed,
                with_all_vision=self._all_vision)
        self._train = cfg.train
        self._last_team_size = None
        self._init_flag = False

    def _launch_game(self) -> Server:
        server = Server(self._cfg)
        server.start()
        render = EnvRender(server.map_width, server.map_height)
        server.set_render(render)
        self._player_names = sum(server.get_player_names_with_team(), [])
        return server

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = self._launch_game()
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            # self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            pass
            # self._env.seed(self._seed)
        self._final_eval_reward = [0. for _ in range(self._team_num)]
        self._env.reset()
        raw_obs = self._env.obs()
        obs = self._obs_transform(raw_obs)
        rew = self._get_reward(raw_obs)
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def render(self) -> None:
        if not self._init_flag:
            self._env = self._launch_game()
            self._init_flag = True

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def _obs_transform(self, obs: tuple) -> list:
        global_state, player_state = obs
        player_state = OrderedDict(player_state)
        # global
        # 'border': [map_width, map_height] fixed map size
        total_time_feat = one_hot_np(round(min(1200, global_state['total_time']) / 100), 13)
        last_time_feat = one_hot_np(round(min(1200, global_state['last_time']) / 100), 13)
        # only use leaderboard rank
        leaderboard_feat = np.zeros((self._team_num, self._team_num))
        for idx, (team_name, team_size) in enumerate(global_state['leaderboard'].items()):
            team_name_number = int(team_name[-1])
            leaderboard_feat[idx, team_name_number] = 1
        leaderboard_feat = leaderboard_feat.reshape(-1)
        global_feat = np.concatenate([total_time_feat, last_time_feat, leaderboard_feat])
        # player
        obs = []
        for n, value in player_state.items():
            if self._spatial:
                player_spatial_feat = []
                for c, item in enumerate(value['feature_layers']):
                    # cv2.imwrite('before_{}_{}.jpg'.format(n, c), item*255)
                    one_channel_item = item[..., np.newaxis].astype(np.float32)
                    resize_item = cv2.resize(one_channel_item, (self._resize_width, self._resize_height))
                    player_spatial_feat.append(resize_item)
                    # cv2.imwrite('after_{}_{}.jpg'.format(n, c), resize_item.astype(np.uint8)*255)
                player_spatial_feat = np.stack(player_spatial_feat, axis=-1).transpose(2, 0, 1)

            team_name_feat = one_hot_np(int(value['team_name'][-1]), self._team_num)
            ori_left_top_x, ori_left_top_y, ori_right_bottom_x, ori_right_bottom_y = value['rectangle']
            left_top_x, right_bottom_x = ori_left_top_x / self._map_width, ori_right_bottom_x / self._map_width
            left_top_y, right_bottom_y = ori_left_top_y / self._map_height, ori_right_bottom_y / self._map_height
            rectangle_feat = np.stack([left_top_x, left_top_y, right_bottom_x, right_bottom_y])
            player_scalar_feat = np.concatenate([rectangle_feat, team_name_feat])

            player_unit_feat = []
            unit_type_mapping = {'food': 0, 'thorn': 1, 'spore': 2, 'clone': 3}
            raw_overlap = {}
            for unit_type in value['overlap']:
                raw_overlap_one_type = list(value['overlap'][unit_type])
                if raw_overlap_one_type is None:
                    raw_overlap_one_type = []
                raw_overlap[unit_type] = copy.deepcopy(raw_overlap_one_type)
                for unit in raw_overlap_one_type:
                    if unit_type == 'clone':
                        position, radius = unit['position'], unit['radius']
                        player_name, team_name = unit['player'], unit['team']
                        player_number, team_nubmer = int(player_name[-1]), int(team_name[-1])
                    else:
                        position, radius = unit['position'], unit['radius']
                        player_number, team_nubmer = self._player_num, self._team_num  # placeholder
                    radius_feat = one_hot_np(round(min(10, math.sqrt(radius))), 11)
                    position = [
                        (position[0] - ori_left_top_x) / (ori_right_bottom_x - ori_left_top_x),
                        (position[1] - ori_right_bottom_y) / (ori_left_top_y - ori_right_bottom_y)
                    ]
                    position_feat = np.stack(position)
                    player_feat = one_hot_np(player_number, self._player_num + 1)
                    team_feat = one_hot_np(team_nubmer, self._team_num + 1)
                    player_unit_feat_item = np.concatenate([position_feat, radius_feat, player_feat, team_feat])
                    player_unit_feat.append(player_unit_feat_item)
            if len(player_unit_feat) <= 200:
                padding_num = 200 - len(player_unit_feat)
                padding_player_unit_feat = np.zeros((padding_num, player_unit_feat[0].shape[0]))
                player_unit_feat = np.stack(player_unit_feat)
                player_unit_feat = np.concatenate([player_unit_feat, padding_player_unit_feat])
            else:
                player_unit_feat = np.stack(player_unit_feat)[-200:]

            obs.append(
                {
                    'scalar_obs': np.concatenate([global_feat, player_scalar_feat]).astype(np.float32),
                    'unit_obs': player_unit_feat.astype(np.float32),
                    'unit_num': len(player_unit_feat),
                    'collate_ignore_raw_obs': copy.deepcopy({'overlap': raw_overlap}),
                }
            )
            if self._spatial:
                obs[-1]['spatial_obs'] = player_spatial_feat.astype(np.float32)
        team_obs = []
        for i in range(self._team_num):
            team_obs.append(obs[i * self._player_num_per_team:(i + 1) * self._player_num_per_team])
        return team_obs

    def _act_transform(self, act: list) -> dict:
        act = [item.tolist() for item in act]
        act = sum(act, [])
        # the element of act can be int scalar or structed object
        return {n: self._to_raw_action(a) if np.isscalar(a) else a for n, a in zip(self._player_names, act)}

    @staticmethod
    def _to_raw_action(act: int) -> Tuple[float, float, int]:
        assert 0 <= act < 16
        # -1, 0, 1, 2(noop, eject, split, gather)
        # 0, 1, 2, 3(up, down, left, right)
        action_type, direction = act // 4, act % 4
        action_type = action_type - 1
        if direction == 0:
            x, y = 0, 1
        elif direction == 1:
            x, y = 0, -1
        elif direction == 2:
            x, y = -1, 0
        elif direction == 3:
            x, y = 1, 0
        return [x, y, action_type]

    def _get_reward(self, obs: tuple) -> list:
        global_state, _ = obs
        if self._last_team_size is None:
            team_reward = [np.array([0.]) for __ in range(self._team_num)]
        else:
            reward = []
            for n in self._player_names:
                team_name = str(int(n) // self._player_num_per_team)
                last_size = self._last_team_size[team_name]
                cur_size = global_state['leaderboard'][team_name]
                reward.append(np.array([cur_size - last_size]))
            team_reward = []
            for i in range(self._team_num):
                team_reward_item = sum(reward[i * self._player_num_per_team:(i + 1) * self._player_num_per_team])
                if self._train:
                    team_reward_item = np.clip(team_reward_item / 2, -1, 1)
                team_reward.append(team_reward_item)
        self._last_team_size = global_state['leaderboard']
        return team_reward

    def step(self, action: list) -> BaseEnvTimestep:
        action = self._act_transform(action)
        done = self._env.step(action)
        raw_obs = self._env.obs()
        obs = self._obs_transform(raw_obs)
        rew = self._get_reward(raw_obs)
        info = [{} for _ in range(self._team_num)]

        for i, team_reward in enumerate(rew):
            self._final_eval_reward[i] += team_reward
        if done:
            for i in range(self._team_num):
                info[i]['final_eval_reward'] = self._final_eval_reward[i]
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=self._player_num,
            obs_space=T(
                {
                    'spatial': (self._player_num + 3, self._resize_width, self._resize_height),
                    'scalar': (42, ),
                    'unit': (188, 21),  # unit is dynamic list
                },
                {
                    'min': 0,
                    'max': 1,
                    'dtype': np.float32,
                },
            ),
            # [min, max)
            act_space=T(
                (1, ),
                {
                    'min': 0,
                    'max': 16,
                    'dtype': int,
                },
            ),
            rew_space=T(
                (1, ),
                {
                    'min': -1000.0,
                    'max': 1000.0,
                    'dtype': np.float32,
                },
            ),
            use_wrappers=None,
        )

    def __repr__(self) -> str:
        return "DI-engine GoBigger Env"

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        raise NotImplementedError

def action_space(angle_num):
    angle = 2 * np.pi * np.arange(angle_num) / angle_num

    angle_direction = np.stack([np.cos(angle), np.sin(angle)], axis=1) # [n, 2]
    zero_direction = np.zeros((angle_num, 2)) # [n, 2]
    direction_action = np.stack([zero_direction, angle_direction, angle_direction, angle_direction, zero_direction], axis=0) # [5, n, 2]

    discrete_action = np.zeros((5, angle_num, 4))
    discrete_action[np.array([0, 1, 2, 3, 4]), :, np.array([0, 0, 1, 2, 3])] = 1

    action = np.concatenate([direction_action, discrete_action], axis=2).reshape(5 * angle_num, 6)
    return action

def obs_pad(obs, pad_num, shape):
    obs_num = len(obs)
    if obs_num < pad_num:
        obs_ = np.zeros((pad_num - obs_num, *shape))
        if obs_num > 0:
            obs = np.concatenate([obs, obs_])
        else:
            obs = obs_
    obs = obs[:pad_num]
    mask = np.zeros(pad_num)
    mask[:obs_num] = 1.0
    return obs, mask

def unit_id(unit_player, unit_team, ego_player, ego_team, team_size):
    unit_player, unit_team, ego_player, ego_team = int(unit_player) % team_size, int(unit_team), int(ego_player) % team_size, int(ego_team)
    # The ego team's id is always 0, enemey teams' ids are 1,2,...,team_num-1
    # The ego player's id is always 0, allies' ids are 1,2,...,player_num_per_team-1
    if unit_team != ego_team:
        player_id = unit_player
        team_id = unit_team if unit_team > ego_team else unit_team + 1
    else:
        if unit_player != ego_player:
            player_id = unit_player if unit_player > ego_player else unit_player + 1
        else:
            player_id = 0
        team_id = 0

    return [team_id, player_id]

def food_encode(clone, food, left_top_x, left_top_y, right_bottom_x, right_bottom_y):
    w = (right_bottom_x - left_top_x) // 16 + 1
    h = (right_bottom_y - left_top_y) // 16 + 1
    food_map = np.zeros((2, h, w))

    w_ = (right_bottom_x - left_top_x) // 8 + 1
    h_ = (right_bottom_y - left_top_y) // 8 + 1
    food_grid = [ [ [] for j in range(w_) ] for i in range(h_) ]
    food_relation = np.zeros((len(clone), 7 * 7 + 1, 3))

    for p in food:
        x = min(max(p[0], left_top_x), right_bottom_x) - left_top_x
        y = min(max(p[1], left_top_y), right_bottom_y) - left_top_y
        r = p[2]
        # encode food density map
        i, j = int(y // 16), int(x // 16)
        food_map[0, i, j] += r * r
        # encode food fine grid
        i, j = int(y // 8), int(x // 8)
        food_grid[i][j].append([(x - 8 * j) / 8, (y - 8 * i) / 8, r])

    for c_id, p in enumerate(clone):
        x = min(max(p[0], left_top_x), right_bottom_x) - left_top_x
        y = min(max(p[1], left_top_y), right_bottom_y) - left_top_y
        r = p[2]
        # encode food density map
        i, j = int(y // 16), int(x // 16)
        if int(p[3]) == 0 and int(p[4]) == 0:
            food_map[1, i, j] += r * r
        # encode food fine grid
        i, j = int(y // 8), int(x // 8)
        t, b, l, r = max(i - 3, 0), min(i + 4, h_), max(j - 3, 0), min(j + 4, w_)
        for ii in range(t, b):
            for jj in range(l, r):
                for f in food_grid[ii][jj]:
                    food_relation[c_id][(ii - t) * 7 + jj - l][0] = f[0]
                    food_relation[c_id][(ii - t) * 7 + jj - l][1] = f[1]
                    food_relation[c_id][(ii - t) * 7 + jj - l][2] += f[2] * f[2]

        food_relation[c_id][-1][0] = (x - j * 8) / 8
        food_relation[c_id][-1][1] = (y - i * 8) / 8
        food_relation[c_id][-1][2] = r / 10

    food_map[0, :, :] = np.sqrt(food_map[0, :, :]) / 2
    food_map[1, :, :] = np.sqrt(food_map[1, :, :]) / 10
    food_relation[:, :-1, 2] = np.sqrt(food_relation[:, :-1, 2]) / 2
    food_relation = food_relation.reshape(len(clone), -1)
    return food_map, food_relation

def clone_encode(clone):
    pos = clone[:, :2] / 100
    rds = clone[:, 2:3] / 10
    ids = np.zeros((len(clone), 12))
    ids[np.arange(len(clone)), (clone[:, 3] * 3 + clone[:, 4]).astype(np.int64)] = 1.0
    slpit = (clone[:, 2:3] - 10) / 10
    eject = (clone[:, 2:3] - 10) / 10
    clone = np.concatenate([pos, rds, ids, slpit, eject], axis=1)
    return clone

def relation_encode(point_1, point_2):
    pos_rlt_1 = point_2[None,:,:2] - point_1[:,None,:2] # relative position
    pos_rlt_2 = np.linalg.norm(pos_rlt_1, ord=2, axis=2, keepdims=True) # distance
    pos_rlt_3 = point_1[:,None,2:3] - pos_rlt_2 # whether source collides with target
    pos_rlt_4 = point_2[None,:,2:3] - pos_rlt_2 # whether target collides with source
    pos_rlt_5 = (2 + np.sqrt(0.5)) * point_1[:,None,2:3] - pos_rlt_2 # whether source's split collides with target
    pos_rlt_6 = (2 + np.sqrt(0.5)) * point_2[None,:,2:3] - pos_rlt_2 # whether target's split collides with source
    rds_rlt_1 = point_1[:,None,2:3] - point_2[None,:,2:3] # whether source can eat target
    rds_rlt_2 = np.sqrt(0.5) * point_1[:,None,2:3] - point_2[None,:,2:3] # whether source's split can eat target
    rds_rlt_3 = np.sqrt(0.5) * point_2[None,:,2:3] - point_1[:,None,2:3] # whether target's split can eat source
    rds_rlt_4 = point_1[:,None,2:3].repeat(len(point_2), axis=1) # target radius
    rds_rlt_5 = point_2[None,:,2:3].repeat(len(point_1), axis=0) # target radius
    relation = np.concatenate([pos_rlt_1 / 100, pos_rlt_2 / 100, pos_rlt_3 / 100, pos_rlt_4 / 100, pos_rlt_5 / 100, pos_rlt_6 / 100, rds_rlt_1 / 10, rds_rlt_2 / 10, rds_rlt_3 / 10, rds_rlt_4 / 10, rds_rlt_5 / 10], axis=2)
    return relation

def team_obs_stack(team_obs):
    result = {}
    for k in team_obs[0].keys():
        result[k] = [o[k] for o in team_obs]
    return result

@ENV_REGISTRY.register('gobigger_hybrid')
class GoBiggerHybridEnv(GoBiggerEnv):
    config = dict(
        player_num_per_team=2,
        team_num=3,
        match_time=1200,
        map_height=1000,
        map_width=1000,
        spatial=False,
        speed = False,
        all_vision = False,
        train=True,
        food_pad_num=24,
        thorn_pad_num=16,
        angle_num=32,
    )

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._player_num_per_team = cfg.player_num_per_team
        self._team_num = cfg.team_num
        self._player_num = self._player_num_per_team * self._team_num
        self._match_time = cfg.match_time
        self._map_height = cfg.map_height
        self._map_width = cfg.map_width

        self._last_team_size = None
        self._init_flag = False
        self._spatial = cfg.spatial
        self._speed = cfg.speed
        self._all_vision = cfg.all_vision
        self._cfg['obs_settings'] = dict(
                with_spatial=self._spatial,
                with_speed=self._speed,
                with_all_vision=self._all_vision)
        self._train = cfg.train

        self._action_space = action_space(cfg.angle_num)

    def _obs_transform(self, obs: tuple) -> list:
        global_state, player_state = obs
        player_state = OrderedDict(player_state)
        # global
        # 'border': [map_width, map_height] fixed map size
        total_time = global_state['total_time']
        last_time = global_state['last_time']
        rest_time = total_time - last_time
        # only use leaderboard rank
        # leaderboard_feat = np.zeros((self._team_num, self._team_num))
        # for idx, (team_name, team_size) in enumerate(global_state['leaderboard'].items()):
        #     team_name_number = int(team_name[-1])
        #     leaderboard_feat[idx, team_name_number] = 1
        # leaderboard_feat = leaderboard_feat.reshape(-1)
        # global_feat = np.concatenate([total_time_feat, last_time_feat, leaderboard_feat])

        # player
        obs = []
        collate_ignore_raw_obs = []
        for n, value in player_state.items():
            # scalar feat
            # get margin
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = value['rectangle']
            center_x, center_y = (left_top_x + right_bottom_x) / 2, (left_top_y + right_bottom_y) / 2
            left_margin, right_margin = left_top_x, self._map_width - right_bottom_x
            top_margin, bottom_margin = left_top_y, self._map_height - right_bottom_y
            # get scalar feat
            scalar_obs = np.array([rest_time / 1000, left_margin / 1000, right_margin / 1000, top_margin / 1000, bottom_margin / 1000])

            # unit feat
            overlap = value['overlap']
            # load units
            food = np.array(overlap['food'] + overlap['spore']) if len(overlap['food'] + overlap['spore']) > 0 else np.array([[center_x, center_y, 0]])
            thorn = np.array(overlap['thorns']) if len(overlap['thorns']) > 0 else np.array([[center_x, center_y, 0]])
            clone = np.array([[x[0], x[1], x[2], *unit_id(x[3], x[4], n, value['team_name'], self._player_num_per_team)] for x in overlap['clone']]) if len(overlap['clone']) > 0 else np.array([[center_x, center_y, 0, 0, 0]])
            overlap['clone'] = [[x[0], x[1], x[2], int(x[3]), int(x[4])] for x in overlap['clone']]
            # encode units
            food, food_relation = food_encode(clone, food, left_top_x, left_top_y, right_bottom_x, right_bottom_y)
            thorn_relation = relation_encode(clone, thorn)
            clone_relation = relation_encode(clone, clone)
            clone = clone_encode(clone)

            player_obs = {
                'scalar': scalar_obs.astype(np.float32),
                'food': food.astype(np.float32),
                'food_relation': food_relation.astype(np.float32),
                'thorn_relation': thorn_relation.astype(np.float32),
                'clone': clone.astype(np.float32),
                'clone_relation': clone_relation.astype(np.float32),
                'action_space': self._action_space.astype(np.float32),
                'collate_ignore_raw_obs': {'overlap': overlap},
            }
            obs.append(player_obs)

        team_obs = []
        for i in range(self._team_num):
            team_obs.append(team_obs_stack(obs[i * self._player_num_per_team: (i + 1) * self._player_num_per_team]))
        return team_obs

    def _act_transform(self, act: list) -> dict:
        act = np.concatenate([item for item in act], axis=0)
        return {n: self._to_raw_action(a) for n, a in zip(self._player_names, act)}

    @staticmethod
    def _to_raw_action(act) -> Tuple[float, float, int]:
        # -1, 0, 1, 2(noop, eject, split, gather)
        action_type, direction = act[2:], act[:2]
        action_type = round(np.sum(action_type * np.array([-1, 0, 1, 2])))
        x, y = direction if np.sum(direction * direction) > 1e-5 else [None, None]
        return [x, y, action_type]

# from easydict import EasyDict
# env = GoBiggerHybridEnv(EasyDict(GoBiggerHybridEnv.config))
# obs = env.reset()
# print(obs[0]['food_relation'][0][0])
# action = np.array([[1,0,1,0,0,0],[0,1,1,0,0,0]])
# print(env.step([action,action,action]).reward)
