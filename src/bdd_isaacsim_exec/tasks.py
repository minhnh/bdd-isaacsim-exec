# SPDX-License-Identifier:  GPL-3.0-or-later
from typing import Any
from rdflib import Graph, Namespace
from rdf_utils.naming import get_valid_var_name
from rdf_utils.uri import URL_SECORO_M
from bdd_dsl.models.user_story import ScenarioVariantModel, SceneModel
from bdd_isaacsim_exec.utils import create_rigid_prim_in_scene

from omni.isaac.core import World
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.scenes.scene import Scene as IsaacScene


NS_M_TMPL = Namespace(f"{URL_SECORO_M}/acceptance-criteria/bdd/templates/")
URI_M_BHV_PICKPLACE = NS_M_TMPL["bhv-pickplace"]
URI_M_TASK_PICKPLACE = NS_M_TMPL["task-pickplace"]
URI_M_TASK_SORTING = NS_M_TMPL["task-sorting"]


class PickPlace(BaseTask):
    def __init__(
        self, graph: Graph, scene_model: SceneModel, task_name: str, **kwargs: Any
    ) -> None:
        offset = kwargs.get("offset", None)
        BaseTask.__init__(self, name=task_name, offset=offset)

        self._ns_manager = graph.namespace_manager
        # TODO(minhnh): handle different scenes for the same task
        self._scene_model = scene_model

        # load object models
        self._obj_models = {}  # object URI -> ObjectModel
        for obj_id in self._scene_model.objects:
            self._obj_models[obj_id] = self._scene_model.load_obj_model(obj_id=obj_id, graph=graph)

        # load agent models
        self._agn_models = {}  # object URI -> AgentModel
        for agn_id in self._scene_model.agents:
            self._agn_models[agn_id] = self._scene_model.load_agn_model(
                agent_id=agn_id, graph=graph
            )

        # TODO(minhnh): load workspace models

        self._obj_prims = {}  # object URI -> RigidPrim
        self._agn_prims = {}  # agent URI -> RigidPrim

    def set_up_scene(self, scene: IsaacScene) -> None:
        super().set_up_scene(scene)

        scene.add_default_ground_plane()

        for obj_id in self._scene_model.objects:
            print(f"*** loading model for object {obj_id}")
            obj_prim = create_rigid_prim_in_scene(
                scene=scene,
                ns_manager=self._ns_manager,
                obj_model=self._obj_models[obj_id],
                prim_prefix="/World/Objects/",
            )
            self._obj_prims[obj_id] = obj_prim

        for agn_id in self._scene_model.agents:
            print(f"*** loading model for agent {agn_id}")
            agn_prim = create_rigid_prim_in_scene(
                scene=scene,
                ns_manager=self._ns_manager,
                obj_model=self._agn_models[agn_id],
                prim_prefix="/World/Agents/",
            )
            self._agn_prims[agn_id] = agn_prim


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
        task = PickPlace(graph=graph, scene_model=scr_var.scene, task_name=task_name, **kwargs)
        world.add_task(task)
        return task

    raise RuntimeError(f"unhandled task: {scr_var.scenario.task_id}")
