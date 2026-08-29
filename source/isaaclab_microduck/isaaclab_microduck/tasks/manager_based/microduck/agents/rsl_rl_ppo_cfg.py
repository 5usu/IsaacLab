# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configurations for MicroDuck."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from .symmetry import compute_symmetric_states


@configclass
class MicroDuckPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50_000
    save_interval = 250
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    experiment_name = "velocity"
    run_name = "velocity"
    logger = "wandb"
    wandb_project = "isaaclab_microduck"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MicroDuckVelStandPPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "velstand"
    run_name = "velstand"
    max_iterations = 20_000


@configclass
class MicroDuckStandUpPPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "microduck_stand"
    run_name = "microduck_stand"
    max_iterations = 15_000


@configclass
class MicroDuckSitStandPPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "microduck_sitstand"
    run_name = "microduck_sitstand"
    max_iterations = 15_000


@configclass
class MicroDuckGroundPickPPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "ground_pick"
    run_name = "ground_pick"
    max_iterations = 20_000


@configclass
class MicroDuckBallKickPPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "ball_kick_right"
    run_name = "ball_kick_right"
    max_iterations = 10_000


@configclass
class MicroDuckRollersPPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "velocity_rollers"
    run_name = "velocity_rollers"
    algorithm = MicroDuckPPORunnerCfg().algorithm.replace(entropy_coef=0.03)


@configclass
class MicroDuckSwizzlePPORunnerCfg(MicroDuckRollersPPORunnerCfg):
    experiment_name = "velocity_swizzle"
    run_name = "velocity_swizzle"


@configclass
class MicroDuckRollerCrouchPPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "roller_crouch"
    run_name = "roller_crouch"
    max_iterations = 8_000


@configclass
class MicroDuckRollerSlopePPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "roller_slope"
    run_name = "roller_slope"
    max_iterations = 8_000


@configclass
class MicroDuckRollerStandUpPPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "roller_standup"
    run_name = "roller_standup"
    max_iterations = 15_000


@configclass
class MicroDuckSpinPPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "spin"
    run_name = "spin"
    max_iterations = 8_000


@configclass
class MicroDuckRouladePPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "microduck_roulade"
    run_name = "microduck_roulade"
    max_iterations = 10_000
    algorithm = MicroDuckPPORunnerCfg().algorithm.replace(
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=0.5,
            data_augmentation_func=compute_symmetric_states,
        )
    )


@configclass
class MicroDuckBackflipPPORunnerCfg(MicroDuckRouladePPORunnerCfg):
    experiment_name = "microduck_backflip"
    run_name = "microduck_backflip"


@configclass
class MicroDuckParkourPPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "microduck_parkour"
    run_name = "microduck_parkour"


@configclass
class MicroDuckTestBenchPPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "testbench"
    run_name = "testbench"
    max_iterations = 2_000
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
