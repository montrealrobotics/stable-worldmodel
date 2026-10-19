"""Gymnasium/SWM bridge for RandomGoal and Overhead ManiSkill PushT.

Gymnasium imports this module via swm/ManiSkillPushT-v1. With expert_checkpoint,
goal images are selected from a deterministic PPO trajectory and the environment
is restored to a configurable trajectory start before planning. Without an
expert, synthetic goal rendering places the T on the target without advancing
physics. See wrapper.py for the CPU (one scene per env) and GPU (batched)
simulation backends.
"""

from mani_skill.utils.structs.pose import Pose

from .random_goal_pusht import quaternion_yaw
from .wrapper import ManiSkillSWMEnv, ManiSkillTask


class PushTTask(ManiSkillTask):
    """The T block's pose is the goal; its native target is goal_tee."""

    def object_pose(self, base):
        return base.tee.pose.raw_pose

    def target_pose(self, base):
        return base.goal_tee.pose.raw_pose

    def set_object_pose(self, base, raw_pose):
        base.tee.set_pose(Pose.create(raw_pose))

    def target_object_pose(self, base):
        target = self.target_pose(base).clone()
        # Target marker is on the table; the physical block center is 2 cm higher.
        target[:, 2] = 0.021
        return target

    def yaw(self, quaternion):
        return quaternion_yaw(quaternion)


class PushTSWMEnv(ManiSkillSWMEnv):
    """Expose one ManiSkill PushT scene with SWM observations and goals."""

    task_cls = PushTTask
