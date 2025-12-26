import logging
import random

from src.core.models import Intent, Shot

logger = logging.getLogger("FashionEngine")


class ShotMutator:
    """
    Intelligently mutates shots to increase diversity and quality.
    """

    def mutate(
        self,
        base_shot: Shot,
        intent: Intent,
        failure_notes: list[str],
        mutation_level: int = 1,
    ) -> Shot:
        """
        mutation_level:
        1 → small change
        2 → medium change
        3 → aggressive change
        """

        poses = intent.scene_rules.poses
        envs = intent.scene_rules.environments
        cams = intent.scene_rules.camera_actions
        focus = intent.scene_rules.focus_points

        new_pose = base_shot.pose
        new_env = base_shot.environment
        new_cam = base_shot.camera_action
        new_focus = list(base_shot.focus_points)

        # 🧠 Interpret failure
        if any("visibility" in n.lower() for n in failure_notes):
            new_focus = ["Overall product visibility", "Logo clarity"]

        if any("motion" in n.lower() for n in failure_notes):
            new_cam = random.choice([c for c in cams if "slow" in c.lower()] or cams)

        if any("brand" in n.lower() for n in failure_notes):
            new_env = random.choice(envs)

        # 🎲 Escalate mutation if needed
        if mutation_level >= 2:
            new_pose = random.choice(poses)

        if mutation_level >= 3:
            new_cam = random.choice(cams)
            new_env = random.choice(envs)
            new_focus = random.sample(focus, min(2, len(focus)))

        logger.info(
            f"🔁 Mutated shot | " f"Pose: {new_pose}, Env: {new_env}, Cam: {new_cam}"
        )

        return Shot(
            id=base_shot.id,
            pose=new_pose,
            environment=new_env,
            camera_action=new_cam,
            focus_points=new_focus,
        )
