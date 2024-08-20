# SPDX-License-Identifier:  GPL-3.0-or-later
from typing import Any
from rdflib import Graph, Namespace, URIRef
from rdf_utils.naming import get_valid_var_name
from rdf_utils.uri import URL_SECORO_M
from bdd_dsl.user_story import SceneModel
from bdd_dsl.simulation.common import ObjModelLoader
from bdd_isaacsim_exec.utils import create_rigid_prim_in_scene

from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.scenes.scene import Scene as IsaacScene


NS_M_TMPL = Namespace(f"{URL_SECORO_M}/acceptance-criteria/bdd/templates/")
URI_M_BHV_PICKPLACE = NS_M_TMPL["bhv-pickplace"]
URI_M_TASK_PICKPLACE = NS_M_TMPL["task-pickplace"]


class PickPlace(BaseTask):
    def __init__(
        self, graph: Graph, scene_model: SceneModel, task_id: URIRef, **kwargs: Any
    ) -> None:
        offset = kwargs.get("offset", None)
        task_id_str = task_id.n3(namespace_manager=graph.namespace_manager)
        BaseTask.__init__(self, name=get_valid_var_name(task_id_str), offset=offset)

        self.task_id = task_id
        self._graph = graph
        # TODO(minhnh): handle different scenes for the same task
        self._scene_model = scene_model
        self._obj_model_loader = ObjModelLoader(graph)
        self._obj_prims = {}  # object URI -> RigidPrim

    def set_up_scene(self, scene: IsaacScene) -> None:
        super().set_up_scene(scene)

        scene.add_default_ground_plane()

        for obj_id in self._scene_model.objects:
            # TODO(minhnh): perhaps make sense for a generator here
            obj_model = self._obj_model_loader.load_object_model(obj_id)
            obj_prim = create_rigid_prim_in_scene(
                scene=scene, graph=self._graph, obj_model=obj_model, prim_prefix="/World/Objects/"
            )
            self._obj_prims[obj_id] = obj_prim


def load_isaacsim_task(graph: Graph, scene: SceneModel, task_id: URIRef, **kwargs: Any) -> BaseTask:
    if task_id == URI_M_TASK_PICKPLACE:
        return PickPlace(graph=graph, scene_model=scene, task_id=task_id, **kwargs)

    raise RuntimeError(f"unhandled task: {task_id}")
