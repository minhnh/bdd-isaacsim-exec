# SPDX-License-Identifier:  GPL-3.0-or-later
from typing import Any, Optional
import numpy as np
from behave.runner import Context
from rdflib import Namespace, URIRef
from rdflib.namespace import NamespaceManager
from bdd_dsl.execution.common import Behaviour

from bdd_isaacsim_exec.tasks import MeasurementType
from omni.isaac.manipulators.controllers import PickPlaceController as GenPickPlaceController
from omni.isaac.core.controllers.articulation_controller import ArticulationController


NS_FRANKA = Namespace("https://www.franka.de/")
URI_M_AGN_TYPE_PANDA = NS_FRANKA["emika-panda"]
NS_UR = Namespace("https://www.universal-robots.com/products/")
URI_M_AGN_TYPE_UR10 = NS_UR["ur:ur10-robot"]


class IsaacsimPickPlaceBehaviour(Behaviour):
    agn_id: Optional[URIRef]
    obj_id: Optional[URIRef]
    ws_id: Optional[list[URIRef]]
    _fsm: Optional[GenPickPlaceController]
    _art_ctrl: Optional[ArticulationController]

    def __init__(
        self,
        bhv_id: URIRef,
        bhv_types: set[URIRef],
        context: Context,
        ns_manager: NamespaceManager,
        **kwargs: Any,
    ) -> None:
        super().__init__(bhv_id=bhv_id, bhv_types=bhv_types, context=context, **kwargs)
        self._ns_manager = ns_manager
        self._fsm = None

    def is_finished(self, context: Context, **kwargs: Any) -> bool:
        assert (
            self._fsm is not None
        ), f"Behaviour '{self.id}': no FSM loaded, reset() not yest called"
        return self._fsm.is_done()

    def reset(self, context: Context, **kwargs: Any) -> None:
        task_params = context.task.get_params()
        self.agn_id = task_params["agn_id"]["value"]
        self.obj_id = task_params["obj_id"]["value"]
        place_ws_ids = task_params["place_ws_ids"]["value"]
        rand_ws_index = np.random.randint(len(place_ws_ids))
        self.ws_id = place_ws_ids[rand_ws_index]

        # add measurements required by behaviour
        context.task.add_measurement(elem_id=self.obj_id, meas_type=MeasurementType.OBJ_POSE)
        context.task.add_measurement(elem_id=self.ws_id, meas_type=MeasurementType.WS_POSE)
        context.task.add_measurement(
            elem_id=self.agn_id, meas_type=MeasurementType.AGN_JNT_POSITIONS
        )

        agn_prim = context.task.get_agn_prim(self.agn_id)
        self._art_ctrl = agn_prim.get_articulation_controller()

        agn_model = context.task.get_agn_model(self.agn_id)
        if URI_M_AGN_TYPE_PANDA in agn_model.types:
            from omni.isaac.franka.controllers import PickPlaceController as PandaPickPlaceCtrl

            self._fsm = PandaPickPlaceCtrl(
                name="pick_place_controller", gripper=agn_prim.gripper, robot_articulation=agn_prim
            )
        elif URI_M_AGN_TYPE_UR10 in agn_model.types:
            from omni.isaac.universal_robots.controllers import (
                PickPlaceController as URPickPlaceCtrl,
            )

            self._fsm = URPickPlaceCtrl(
                name="pick_place_controller", gripper=agn_prim.gripper, robot_articulation=agn_prim
            )
        else:
            raise RuntimeError(
                f"Behaviour '{self.id}': unrecognized agent types: {agn_model.types}"
            )

    def step(self, context: Context, **kwargs: Any) -> Any:
        assert (
            self.agn_id is not None
            and self.obj_id is not None
            and self.ws_id is not None
            and self._fsm is not None
            and self._art_ctrl is not None
        ), f"Behaviour '{self.id}': params are None, step() expects reset() to be called first"
        obs = context.observations
        assert self.obj_id in obs, f"target obj '{self.obj_id}' not in observations"
        assert self.ws_id in obs, f"target ws '{self.ws_id}' not in observations"
        assert self.agn_id in obs, f"target agn '{self.agn_id}' not in observations"

        ws_obj_list = list(obs[self.ws_id]["objects"])
        assert len(ws_obj_list) == 1, f"unexpected number of ws objects (not 1): {ws_obj_list}"
        ws_obj_id = ws_obj_list[0]
        assert (
            ws_obj_id in obs
        ), f"obj '{ws_obj_id}' of target place ws '{self.ws_id}' not in observations"

        actions = self._fsm.forward(
            picking_position=obs[self.obj_id]["position"],
            placing_position=obs[ws_obj_id]["position"],
            current_joint_positions=obs[self.agn_id]["joint_positions"],
            end_effector_offset=np.array([0, 0.005, -0.015]),
        )
        self._art_ctrl.apply_action(actions)
