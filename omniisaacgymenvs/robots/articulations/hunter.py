from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.usd
import omni.kit
import os
from pxr import UsdLux, Sdf, Gf, UsdPhysics
import carb

class Hunter(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "hunter",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        hunter_usd_path = "/home/username/Hybrid_Deep_Reinforcement_Learning_RoughTerrain/OmniIsaacGymEnvs/omniisaacgymenvs/USD_Files/hunter_aim4.usd"
        self._usd_path = hunter_usd_path
        self._name = name
        add_reference_to_stage(self._usd_path, prim_path)
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )
