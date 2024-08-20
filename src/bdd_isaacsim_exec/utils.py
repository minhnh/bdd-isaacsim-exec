# SPDX-License-Identifier:  GPL-3.0-or-later
from os.path import exists as os_exists
from rdflib import Graph
from rdf_utils.naming import get_valid_var_name
from rdf_utils.python import URI_PY_TYPE_MODULE_ATTR, import_attr_from_node
from bdd_dsl.models.urirefs import URI_SIM_TYPE_SYS_RES
from bdd_dsl.utils.common import check_or_convert_ndarray
from bdd_dsl.simulation.common import ObjectModel, get_path_of_node
from bdd_isaacsim_exec.uri import URI_SIM_TYPE_ISAAC_RES, URI_TYPE_USD_FILE

from omni.isaac.core.scenes.scene import Scene as IsaacScene
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.prims import is_prim_path_valid


_CACHED_ASSET_ROOT = None


def get_cached_assets_root_path() -> str:
    """Get Isaacsim assets root path and cache.

    Raise exception if asset root directory can't be found.
    Raises:
        RuntimeError: if asset root directory can't befound
    Returns:
        str: Root directory containing assets from Isaac Sim
    """
    global _CACHED_ASSET_ROOT
    if _CACHED_ASSET_ROOT is not None:
        return _CACHED_ASSET_ROOT

    # These imports are assumed to be called after SimulationApp() call,
    # otherwise my cause import errors
    from omni.isaac.core.utils.nucleus import get_assets_root_path

    _CACHED_ASSET_ROOT = get_assets_root_path()
    if _CACHED_ASSET_ROOT is not None:
        return _CACHED_ASSET_ROOT

    raise RuntimeError("Could not find Isaac Sim assets folder")


def create_rigid_prim_in_scene(
    scene: IsaacScene, graph: Graph, obj_model: ObjectModel, prim_prefix: str
) -> RigidPrim:
    id_str = obj_model.id.n3(namespace_manager=graph.namespace_manager)
    id_str = get_valid_var_name(id_str)

    # TODO(minhnh): handle initial poses

    if "scale" in obj_model.configs:
        obj_model.configs["scale"] = (
            check_or_convert_ndarray(obj_model.configs["scale"]) / get_stage_units()
        )
    if "color" in obj_model.configs:
        obj_model.configs["color"] = (
            check_or_convert_ndarray(obj_model.configs["color"]) / get_stage_units()
        )

    prim_path = find_unique_string_name(
        initial_name=prim_prefix + id_str,
        is_unique_fn=lambda x: not is_prim_path_valid(x),
    )
    obj_name = find_unique_string_name(
        initial_name=id_str,
        is_unique_fn=lambda x: not scene.object_exists(x),
    )

    if URI_TYPE_USD_FILE in obj_model.model_types:
        asset_path = get_path_of_node(graph=graph, node_id=obj_model.model_id)

        if URI_SIM_TYPE_ISAAC_RES in obj_model.model_types:
            asset_path = get_cached_assets_root_path() + asset_path

        elif URI_SIM_TYPE_SYS_RES in obj_model.model_types:
            assert os_exists(
                asset_path
            ), f"Path for object model '{obj_model.model_id}' does not exists: {asset_path}"

        else:
            raise RuntimeError(
                f"unhandled types for UsdFile object '{obj_model.model_id}': {obj_model.model_types}"
            )

        add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
        return scene.add(RigidPrim(prim_path=prim_path, name=obj_name, **obj_model.configs))

    if URI_PY_TYPE_MODULE_ATTR in obj_model.model_types:
        obj_cls = import_attr_from_node(graph=graph, uri=obj_model.model_id)
        assert issubclass(
            obj_cls, RigidPrim
        ), f"Python class for object model '{obj_model.model_id}' is not a subclass of Isaacsim RigidPrim: {obj_cls}"

        return scene.add(obj_cls(name=obj_name, prim_path=prim_path, **obj_model.configs))

    raise RuntimeError(f"unhandled types for object'{obj_model.model_id}': {obj_model.model_types}")
