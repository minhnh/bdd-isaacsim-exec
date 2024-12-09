# SPDX-License-Identifier:  GPL-3.0-or-later
from typing import Any, Generator, Optional
from numpy import ndarray
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import NamespaceManager
from rdf_utils.naming import get_valid_var_name
from rdf_utils.uri import URL_SECORO_M
from bdd_dsl.behave import parse_str_param
from bdd_dsl.models.agent import AgentModel
from bdd_dsl.models.environment import ObjectModel, WorkspaceModel
from bdd_dsl.models.user_story import ScenarioVariantModel, SceneModel
from bdd_isaacsim_exec.utils import create_rigid_prim_in_scene

from omni.isaac.core import World
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.scenes.scene import Scene as IsaacScene


NS_M_TMPL = Namespace(f"{URL_SECORO_M}/acceptance-criteria/bdd/templates/")
URI_M_BHV_PICKPLACE = NS_M_TMPL["bhv-pickplace"]
URI_M_TASK_PICKPLACE = NS_M_TMPL["task-pickplace"]
URI_M_TASK_SORTING = NS_M_TMPL["task-sorting"]


class PickPlace(BaseTask):
    _ns_manager: NamespaceManager
    _obj_prims: dict[URIRef, RigidPrim]
    _agn_prims: dict[URIRef, Robot]
    _obj_models: dict[URIRef, ObjectModel]
    _ws_models: dict[URIRef, WorkspaceModel]
    _agn_models: dict[URIRef, AgentModel]
    _obj_id: URIRef
    _agent_id: URIRef
    _place_ws_ids: list[URIRef]

    def __init__(
        self, scene_model: SceneModel, task_name: str, ns_manger: NamespaceManager, **kwargs: Any
    ) -> None:
        offset = kwargs.get("offset", None)
        BaseTask.__init__(self, name=task_name, offset=offset)

        self._ns_manager = ns_manger
        # TODO(minhnh): handle different scenes for the same task
        self._scene_model = scene_model
        self._obj_prims = {}
        self._agn_prims = {}
        self._obj_models = {}
        self._ws_models = {}
        self._agn_models = {}

    def get_agn_model(self, agn_id: URIRef) -> AgentModel:
        assert agn_id in self._agn_models, f"Task {self.name}: no modelfor agent {agn_id}"
        return self._agn_models[agn_id]

    def get_agn_prim(self, agn_id: URIRef) -> Robot:
        assert agn_id in self._agn_prims, f"Task {self.name}: no prim for agent {agn_id}"
        return self._agn_prims[agn_id]

    def add_scene_obj_model(self, obj_model: ObjectModel) -> None:
        if obj_model.id in self._obj_models:
            return
        self._obj_models[obj_model.id] = obj_model

    def add_scene_ws_model(self, ws_model: WorkspaceModel, graph: Graph) -> None:
        if ws_model.id in self._ws_models:
            return
        self._ws_models[ws_model.id] = ws_model
        for obj_model in self._scene_model.env_model_loader.load_ws_objects(
            graph=graph, ws_id=ws_model.id
        ):
            self.add_scene_obj_model(obj_model=obj_model)

    def add_scene_agn_model(self, agn_model: AgentModel) -> None:
        if agn_model.id in self._agn_models:
            return
        self._agn_models[agn_model.id] = agn_model

    def get_ws_obj_ids_re(
        self, ws_id: URIRef, ws_path: Optional[set[URIRef]] = None
    ) -> Generator[URIRef, None, None]:
        if ws_path is None:
            ws_path = set()

        assert ws_id not in ws_path, f"loop detected at ws '{ws_id}'"
        ws_path.add(ws_id)

        assert ws_id in self._ws_models, f"ws '{ws_id}' not in scene"
        for obj_id in self._ws_models[ws_id].objects:
            yield obj_id
        for sub_ws_id in self._ws_models[ws_id].workspaces:
            for obj_id in self.get_ws_obj_ids_re(ws_id=sub_ws_id, ws_path=ws_path):
                yield obj_id

    def set_up_scene(self, scene: IsaacScene) -> None:
        super().set_up_scene(scene)

        scene.add_default_ground_plane()

        for obj_id, obj_model in self._obj_models.items():
            print(f"*** loading model for object {obj_id}")
            obj_prim = create_rigid_prim_in_scene(
                scene=scene,
                ns_manager=self._ns_manager,
                model=obj_model,
                prim_prefix="/World/Objects/",
            )
            self._obj_prims[obj_id] = obj_prim

        for agn_id, agn_model in self._agn_models.items():
            print(f"*** loading model for agent {agn_id}")
            agn_prim = create_rigid_prim_in_scene(
                scene=scene,
                ns_manager=self._ns_manager,
                model=agn_model,
                prim_prefix="/World/Agents/",
            )
            assert isinstance(
                agn_prim, Robot
            ), f"Prim for agn '{agn_id}' not a Isaac Robot instance"
            self._agn_prims[agn_id] = agn_prim

    def set_params(self, **kwargs) -> None:
        """Set parameters values.

        Extension of BaseTask.set_params().

        Parameters:
        - obj_id_str: string rep of object URI to be picked and placed
        - agent_id_str: string rep of agent URI to perform pick & place bhv
        - place_ws_id: URI of workspace(s) for placing.
        """
        agn_id_str = kwargs.get("agn_id_str", None)
        assert agn_id_str is not None, f"Isaac Task '{self.name}': arg 'agn_id_str' not specified'"
        obj_id_str = kwargs.get("obj_id_str", None)
        assert obj_id_str is not None, f"Isaac Task '{self.name}': arg 'obj_id_str' not specified'"
        ws_id_str = kwargs.get("ws_id_str", None)
        assert ws_id_str is not None, f"Isaac Task '{self.name}': arg 'ws_id_str not specified'"

        _, agn_uris = parse_str_param(param_str=agn_id_str, ns_manager=self._ns_manager)
        assert (
            len(agn_uris) == 1
        ), f"Isaac Task '{self.name}': expected 1 agent URI, got: {agn_uris}"
        assert isinstance(agn_uris[0], URIRef), f"unexpected agent URI: {agn_uris[0]}"
        self._agn_id = agn_uris[0]

        _, obj_uris = parse_str_param(param_str=obj_id_str, ns_manager=self._ns_manager)
        assert len(obj_uris) == 1, f"Isaac Task '{self.name}': expected 1 obj URI, got: {obj_uris}"
        assert isinstance(obj_uris[0], URIRef), f"unexpected obj URI: {obj_uris[0]}"
        self._obj_id = obj_uris[0]

        _, place_ws_uris = parse_str_param(param_str=ws_id_str, ns_manager=self._ns_manager)
        self._place_ws_ids = []
        for uri in place_ws_uris:
            assert isinstance(
                uri, URIRef
            ), f"Isaac Task '{self.name}': unexpected place ws URI: {uri}"
            self._place_ws_ids.append(uri)

        return

    def get_params(self) -> dict:
        """Return parameters values.

        Extension of BaseTask.get_params().

        Parameters:
        - agn_id: URI of agent performing pick & place
        - obj_id: URI of object to be picked and placed
        - place_ws_id: URI of workspace where the picked object should be placed

        Returns:
            dict: dictionary containing parameters' values and modifiable flag
        """
        params = {}
        params["agn_id"] = {"value": self._agn_id, "modifiable": True}
        params["obj_id"] = {"value": self._obj_id, "modifiable": True}
        params["place_ws_ids"] = {"value": self._place_ws_ids, "modifiable": True}

        return params

    def get_obj_pose(self, obj_id: URIRef) -> tuple[ndarray, ndarray]:
        assert obj_id in self._obj_prims, f"Isaac Task '{self.name}': no prim for obj '{obj_id}'"
        return self._obj_prims[obj_id].get_local_pose()

    def get_ws_pose(self, ws_id: URIRef) -> dict[URIRef, tuple[ndarray, ndarray]]:
        assert ws_id in self._ws_models, f"Isaac Task '{self.name}': ws '{ws_id}' not on record"
        obj_poses = {}
        for obj_id in self.get_ws_obj_ids_re(ws_id=ws_id):
            obj_position, obj_orientation = self.get_obj_pose(obj_id=obj_id)
            obj_poses[obj_id] = {"position": obj_position, "orientation": obj_orientation}
        return obj_poses

    def get_agn_ee_pose(self, agn_id: URIRef) -> tuple[ndarray, ndarray]:
        assert agn_id in self._agn_prims, f"Isaac Task '{self.name}': no prim for agn '{agn_id}'"
        return self._agn_prims[agn_id].end_effector.get_local_pose()

    def get_agn_joint_positions(self, agn_id: URIRef) -> ndarray:
        assert agn_id in self._agn_prims, f"Isaac Task '{self.name}': no prim for agn '{agn_id}'"
        return self._agn_prims[agn_id].get_joint_positions()

    def get_observations(self) -> dict:
        """Return observations"""
        obj_position, obj_orientation = self.get_obj_pose(obj_id=self._obj_id)
        ee_position, ee_orientation = self.get_agn_ee_pose(agn_id=self._agn_id)
        agn_joint_positions = self.get_agn_joint_positions(agn_id=self._agn_id)
        obs = {
            self._obj_id: {"position": obj_position, "orientation": obj_orientation},
            self._agn_id: {
                "ee_position": ee_position,
                "ee_orientation": ee_orientation,
                "joint_positions": agn_joint_positions,
            },
        }
        for ws_id in self._place_ws_ids:
            ws_obj_poses = self.get_ws_pose(ws_id=ws_id)
            obs |= ws_obj_poses
        return obs

    def cleanup_scene_models(self) -> None:
        """Should be called before loading obj and agent models.

        Either in before_scenario or after_scenario
        """
        self._obj_models.clear()
        self._ws_models.clear()
        self._agn_models.clear()
        self._obj_prims.clear()
        self._agn_prims.clear()


def load_isaacsim_task(
    world: World, graph: Graph, scr_var: ScenarioVariantModel, **kwargs: Any
) -> BaseTask:
    task_name = get_valid_var_name(
        scr_var.scenario.task_id.n3(namespace_manager=graph.namespace_manager)
    )
    try:
        return world.get_task(task_name)
    except Exception:
        # task is not added
        pass

    if (
        scr_var.scenario.task_id == URI_M_TASK_PICKPLACE
        or scr_var.scenario.task_id == URI_M_TASK_SORTING
    ):
        task = PickPlace(
            scene_model=scr_var.scene,
            task_name=task_name,
            ns_manger=graph.namespace_manager,
            **kwargs,
        )
        world.add_task(task)
        return task

    raise RuntimeError(f"unhandled task: {scr_var.scenario.task_id}")
