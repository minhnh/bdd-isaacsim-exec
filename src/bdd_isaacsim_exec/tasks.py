# SPDX-License-Identifier:  GPL-3.0-or-later
from typing import Any, Optional
from bdd_dsl.models.agent import AgentModel
from rdflib import Graph, Namespace, URIRef
from rdf_utils.naming import get_valid_var_name
from rdf_utils.uri import URL_SECORO_M
from bdd_dsl.models.environment import ObjectModel, WorkspaceModel
from bdd_dsl.models.user_story import ScenarioVariantModel, SceneModel
from rdflib.namespace import NamespaceManager
from bdd_isaacsim_exec.utils import create_rigid_prim_in_scene

from omni.isaac.core import World
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.scenes.scene import Scene as IsaacScene


NS_M_TMPL = Namespace(f"{URL_SECORO_M}/acceptance-criteria/bdd/templates/")
URI_M_BHV_PICKPLACE = NS_M_TMPL["bhv-pickplace"]
URI_M_TASK_PICKPLACE = NS_M_TMPL["task-pickplace"]
URI_M_TASK_SORTING = NS_M_TMPL["task-sorting"]


class PickPlace(BaseTask):
    _obj_prims: dict[URIRef, RigidPrim]
    _agn_prims: dict[URIRef, RigidPrim]
    _obj_models: dict[URIRef, ObjectModel]
    _ws_models: dict[URIRef, WorkspaceModel]
    _agn_models: dict[URIRef, AgentModel]

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

    def _get_ws_obj_re(
        self, ws_id: URIRef, obj_set: set[URIRef], ws_path: Optional[set[URIRef]] = None
    ) -> None:
        if ws_path is None:
            ws_path = set()

        assert ws_id not in ws_path, f"loop detected at ws '{ws_id}'"
        ws_path.add(ws_id)

        assert ws_id in self._ws_models, f"ws '{ws_id}' not in scene"
        for obj_id in self._ws_models[ws_id].objects:
            obj_set.add(obj_id)
        for sub_ws_id in self._ws_models[ws_id].workspaces:
            self._get_ws_obj_re(ws_id=sub_ws_id, obj_set=obj_set, ws_path=ws_path)

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
            self._agn_prims[agn_id] = agn_prim

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
