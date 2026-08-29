# MicroDuck Migration Matrix

This matrix records the MicroDuck behavior ported from the original MuJoCo task
package and the validation completed in IsaacLab.

## Robot and physics

- [x] Walking, full-collision, passive-roller, and ball models
- [x] Walking, full-collision, and passive-roller backlash models
- [x] Fixed-base XL330 testbench
- [x] BAM M6 voltage, delay, saturation, friction, gain, and battery-sag dynamics
- [x] Encoder-through-backlash feedback
- [x] Stable 14-action and 61-observation policy contracts
- [x] Source MJCF plus reproducible layered-USD conversion

## Environments

- [x] Velocity, VelStand, StandUp, SitStand, and GroundPick on flat and rough terrain
- [x] BallKick, Roulade, strictly standing Backflip, and Parkour
- [x] Roller velocity, Swizzle, RollerCrouch, RollerSlope, RollerStandUp, and Spin
- [x] All 16 backlash variants registered by the source package
- [x] Train and play registrations for every environment

## Shared behavior

- [x] Custom command generators and curricula
- [x] Observations, history terms, rewards, events, resets, and terminations
- [x] RSL-RL mirror-loss symmetry for roll and backflip
- [x] Rough, slope, and parkour terrains
- [x] Non-finite state guards and reset invariants

## Workflow

- [x] RSL-RL configurations for every task
- [x] IsaacLab train and play CLI integration
- [x] Latest-checkpoint selection through the standard play command
- [x] JIT and ONNX export through the standard play command
- [x] Deterministic MJCF-to-USD asset conversion
- [x] Legacy video, plotting, hosted-training, and ad hoc viewer scripts excluded

The old hardware-side sim-to-real runner remains in the MicroDuck runtime
project. It is not duplicated here because it owns hardware I/O rather than an
IsaacLab environment.

## Validation

- [x] All 74 Gym registrations load their environment and RSL-RL configuration
- [x] Asset conversion, actuator, command, and symmetry unit tests
- [x] All 37 training environments reset and step under PhysX with finite rewards
- [x] XL330 play environment reset and step under PhysX
- [x] Short RSL-RL training smoke test
- [x] Checkpoint playback and JIT/ONNX export smoke test
- [x] Formatting, lint, and repository hygiene checks

Newton is not claimed as validated. The converted assets include multi-physics
payloads, but task behavior has only been exercised under PhysX in this port.
