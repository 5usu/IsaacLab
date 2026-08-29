# Isaac Lab MicroDuck

IsaacLab assets, actuator dynamics, reinforcement-learning environments, and
RSL-RL configurations for the [MicroDuck robot](https://github.com/pollen-robotics/microduck).

## Setup

From the IsaacLab repository root:

```bash
uv sync --extra rl
uv run -- ./isaaclab.sh -p source/isaaclab_microduck/scripts/convert_assets.py
uv run -- ./isaaclab.sh -p -m pytest source/isaaclab_microduck/test/test_registry.py
```

IsaacLab excludes generated USD files from source control. Run the conversion
command once after cloning and again after editing an MJCF or mesh asset. The
extension uses the generated assets from its local `data/robots/usd` directory.
Add `--force` to rebuild existing assets:

```bash
uv run -- ./isaaclab.sh -p source/isaaclab_microduck/scripts/convert_assets.py --force
```

## Train and play

Train a policy with the maintained IsaacLab CLI:

```bash
uv run -- ./isaaclab.sh train --rl_library rsl_rl \
  --external_callback isaaclab_microduck.register_tasks \
  --task Isaac-Velocity-Flat-MicroDuck-v0 \
  --num_envs 4096 --viz none
```

Play the latest checkpoint for that task by omitting `--checkpoint`:

```bash
uv run -- ./isaaclab.sh play --rl_library rsl_rl \
  --external_callback isaaclab_microduck.register_tasks \
  --task Isaac-Velocity-Flat-MicroDuck-Play-v0 \
  --num_envs 1 --viz kit
```

To select a checkpoint explicitly, add `--checkpoint /path/to/model.pt`.
Playback exports `policy.pt` and `policy.onnx` into an `exported` directory next
to the loaded checkpoint.

## Tasks

Every task has a `-Play-v0` registration with reduced environment count and
randomization. The 21 primary task families are:

| Family | Train task ID |
| --- | --- |
| Walking | `Isaac-Velocity-{Flat,Rough}-MicroDuck-v0` |
| Walking and recovery | `Isaac-VelStand-{Flat,Rough}-MicroDuck-v0` |
| Stand up | `Isaac-StandUp-{Flat,Rough}-MicroDuck-v0` |
| Sit and stand | `Isaac-SitStand-{Flat,Rough}-MicroDuck-v0` |
| Ground pick | `Isaac-GroundPick-{Flat,Rough}-MicroDuck-v0` |
| Ball kick | `Isaac-BallKick-Flat-MicroDuck-v0` |
| Roller velocity | `Isaac-Velocity-Flat-MicroDuck-Rollers-v0` |
| Roller swizzle | `Isaac-Velocity-Swizzle-MicroDuck-v0` |
| Roller crouch | `Isaac-RollerCrouch-Flat-MicroDuck-v0` |
| Roller slope | `Isaac-RollerSlope-Flat-MicroDuck-v0` |
| Roller stand up | `Isaac-RollerStandUp-Flat-MicroDuck-v0` |
| Roller spin | `Isaac-Spin-Flat-MicroDuck-v0` |
| Forward roll | `Isaac-Roulade-Flat-MicroDuck-v0` |
| Standing backflip | `Isaac-Backflip-Flat-MicroDuck-v0` |
| Parkour | `Isaac-Parkour-MicroDuck-v0` |
| XL330 testbench | `Isaac-Position-XL330-TestBench-v0` |

Sixteen matching `Backlash` families model serial gearbox play and feed the
post-backlash position to the encoder observation. Run the environment-listing
command above for their exact IDs.

## Policy contract

Production policies use 61 observations and 14 actions. The observation blocks
are angular velocity `[0:3]`, projected gravity `[3:6]`, velocity command
`[6:9]`, joint position `[9:23]`, joint velocity `[23:37]`, previous action
`[37:51]`, head command `[51:55]`, and body command `[55:61]`.

The action order is:

```text
left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
neck_pitch, head_pitch, head_yaw, head_roll,
right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
```

Passive rollers and backlash joints are never part of the action vector. The
XL330 testbench is intentionally separate and uses a 4-observation, 1-action
contract.

## Development

```bash
uv run -- ./isaaclab.sh -p -m pytest source/isaaclab_microduck/test
```

The extension does not carry private video, plotting, checkpoint-discovery, or
hosted-training wrappers. IsaacLab owns the train, play, checkpoint, and export
workflows so this extension stays focused on robot behavior.

See [migration.md](migration.md) for the validation matrix and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for model provenance.
