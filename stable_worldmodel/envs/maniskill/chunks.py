"""Compare action chunks that start from the same PushT state.

Groups of GPU envs share one source slot (the expert). Before each chunk every
slot is reset to its source's state, so a candidate chunk is "correct" when
its end state matches the source's end state within a tolerance.
"""

import math

import torch

TOLERANCES = {  # (tee xy mm, tee yaw deg, tcp xy mm)
    'tight': (0.5, 0.5, 1.0),
    'medium': (1.0, 1.0, 2.0),
    'loose': (2.0, 2.0, 5.0),
}


def index_tree(tree, index):
    if isinstance(tree, dict):
        return {key: index_tree(value, index) for key, value in tree.items()}
    return tree[index]


def restore_from_source(env, source):
    """Copy each source slot's full state (incl. controller target) to its group."""
    state = index_tree(env.get_state_dict(), source)
    env.set_state_dict(state)
    # Flat set_state drops the controller target; restore it explicitly.
    env.agent.set_controller_state(state['controller'])
    return env.get_obs()


def snapshot(env):
    tee = env.tee
    return {
        'tee_xy': tee.pose.p[:, :2].clone(),
        'tee_yaw': env.quat_to_z_euler(tee.pose.q).clone(),
        'tee_vel': torch.cat(
            (tee.linear_velocity[:, :2], tee.angular_velocity[:, 2:]), 1
        ).clone(),
        'tcp_xy': env.agent.tcp.pose.p[:, :2].clone(),
        'qvel': env.agent.robot.get_qvel().clone(),
    }


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def end_errors(end, source):
    """Per-slot end-state distance to the source slot (mm, deg, mm)."""
    return {
        'tee_mm': 1e3 * (end['tee_xy'] - end['tee_xy'][source]).norm(dim=1),
        'yaw_deg': torch.rad2deg(
            wrap(end['tee_yaw'] - end['tee_yaw'][source]).abs()
        ),
        'tcp_mm': 1e3 * (end['tcp_xy'] - end['tcp_xy'][source]).norm(dim=1),
    }


def accepted(errors, tolerance):
    tee, yaw, tcp = tolerance
    return (
        (errors['tee_mm'] <= tee)
        & (errors['yaw_deg'] <= yaw)
        & (errors['tcp_mm'] <= tcp)
    )
