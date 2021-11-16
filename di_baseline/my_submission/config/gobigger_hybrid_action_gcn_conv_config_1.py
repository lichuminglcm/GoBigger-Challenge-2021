from easydict import EasyDict

player_num_per_team=3
team_num=4

gobigger_config = dict(
    exp_name='gobigger_selfplay_baseline_dqn',
    env=dict(
        collector_env_num=32,
        evaluator_env_num=3,
        n_evaluator_episode=3,
        stop_value=1e10,
        team_num=team_num,
        player_num_per_team=player_num_per_team,
        match_time=200,
        map_height=1000,
        map_width=1000,
        spatial=False,
        speed = False,
        all_vision = False,
        angle_num=32,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=False,
        priority_IS_weight=False,
        model=dict(
            scalar_shape=5,
            food_shape=2,
            food_relation_shape=150,
            thorn_relation_shape=12,
            clone_shape=17,
            clone_relation_shape=12,
            hidden_shape=128,
            encode_shape=32,
            action_shape=6,
        ),
        learn=dict(
            update_per_collect=8,
            batch_size=256,
            learning_rate_q=3e-4,
            target_theta=0.005,
            discount_factor=0.99,
            ignore_done=False,
        ),
        collect=dict(n_sample=512, unroll_len=1, alpha=1.0),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.5,
                decay=100000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
main_config = EasyDict(gobigger_config)
gobigger_create_config = dict(
    env=dict(
        type='gobigger_v5',
        import_names=['dizoo.gobigger.envs.gobigger_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='gobigger'),
)
create_config = EasyDict(gobigger_create_config)
