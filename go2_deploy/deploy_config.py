from gym.envs.go2.go2trot_config import Go2TrotCfg
import torch


class DeployConfig:
    task_name = "go2trot"

    ctrl_freq = 100  # Hz

    kp = 30.0  # Stiffness constant
    kd = 2.0  # Damping constant

    phase_frequency = 2.0  # for go2trot env

    # The Go2 remote has no height axis, so the deployed policy is commanded a
    # fixed height: the midpoint of the task's training range.
    command_height = 0.5 * (
        Go2TrotCfg.commands.ranges.height[0] + Go2TrotCfg.commands.ranges.height[1]
    )

    # Apply exponential moving average to actions
    exp_moving_avg = True
    ema_smoothing_factor = 0.1

    # Must mirror Go2TrotRunnerCfg.actor.obs: rl_controller.setup_actor() loads
    # an actor trained on that list, in that order.
    obs_vector = [
        "base_height",
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
    ]  # add later? foot contact

    # Scale observations
    class DeployScaling(Go2TrotCfg.scaling):
        pass

    # Set ranges of joystick commands
    command_limits = {
        "lin_vel_x": 2.0,  # m/s
        "lin_vel_y": 1.0,  # m/s
        "yaw_vel": 3.0,  # rad/s
    }

    # Specify size of observation vector. should not have to modify this
    obs_sizes = {
        "base_height": 1,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "commands": 4,
        "dof_pos_obs": 12,
        "dof_vel": 12,
        "dof_accel": 12,
        "dof_pos_target": 12,
        "phase_obs": 2,
        "phase_frequency": 1,
    }

    default_dof_pos = torch.zeros(12)

    lower_joint_limit = torch.tensor(
        [
            -0.83,
            -1.59,
            -2.72,
            -0.83,
            -1.59,
            -2.72,
            -0.83,
            -0.52,
            -2.72,
            -0.83,
            -0.52,
            -2.72,
        ]
    )
    upper_joint_limit = torch.tensor(4 * [0.83, 3.49, -0.83])

    # s: how often terminal should print ctrl freq, instructions
    terminal_log_period = 2.0

    # Check for unsafe config
    def __init__(self):
        if self.ctrl_freq < 20:
            raise ValueError("ctrl_freq should be > 20")
        if self.kp <= 0:
            raise ValueError("kp should be > 0")
        if self.kd <= 0:
            raise ValueError("kd should be > 0")
        for obs in self.obs_vector:
            if obs not in self.obs_sizes:
                raise KeyError("observation " + obs + " not supported")
