# SPDX-License-Identifier:  GPL-3.0-or-later
from os.path import exists as os_exists
from typing import Union
from numpy.random import uniform
from rdflib.namespace import NamespaceManager
from rdf_utils.naming import get_valid_var_name
from rdf_utils.models.python import URI_PY_TYPE_MODULE_ATTR, import_attr_from_model
from bdd_dsl.models.urirefs import (
    URI_SIM_PRED_HAS_CONFIG,
    URI_SIM_PRED_PATH,
    URI_SIM_TYPE_RES_PATH,
    URI_SIM_TYPE_SYS_RES,
)
from bdd_dsl.utils.common import check_or_convert_ndarray
from bdd_dsl.models.environment import ObjectModel
from bdd_dsl.models.agent import AgentModel
from bdd_isaacsim_exec.uri import URI_SIM_TYPE_ISAAC_RES, URI_TYPE_USD_FILE

from omni.isaac.core.scenes.scene import Scene as IsaacScene
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.prims import is_prim_path_valid


_CACHED_ASSET_ROOT = None
OBJ_POSITION_LOWER_BOUNDS = [0.25, -0.4, 0.12]
OBJ_POSITION_UPPER_BOUNDS = [0.6, 0.4, 0.15]


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
    scene: IsaacScene,
    ns_manager: NamespaceManager,
    model: Union[ObjectModel, AgentModel],
    prim_prefix: str,
) -> RigidPrim:
    id_str = model.id.n3(namespace_manager=ns_manager)
    id_str = get_valid_var_name(id_str)

    # TODO(minhnh): handle initial poses

    obj_configs = model.get_attr(key=URI_SIM_PRED_HAS_CONFIG)
    assert obj_configs is not None and isinstance(obj_configs, dict), f"no configs for {model.id}"

    if "scale" in obj_configs:
        obj_configs["scale"] = check_or_convert_ndarray(obj_configs["scale"]) / get_stage_units()
    if "color" in obj_configs:
        obj_configs["color"] = check_or_convert_ndarray(obj_configs["color"]) / get_stage_units()

    if "position" not in obj_configs:
        obj_position = uniform(OBJ_POSITION_LOWER_BOUNDS, OBJ_POSITION_UPPER_BOUNDS)
        obj_configs["position"] = obj_position / get_stage_units()

    prim_path = find_unique_string_name(
        initial_name=prim_prefix + id_str,
        is_unique_fn=lambda x: not is_prim_path_valid(x),
    )
    obj_name = find_unique_string_name(
        initial_name=id_str,
        is_unique_fn=lambda x: not scene.object_exists(x),
    )

    if URI_TYPE_USD_FILE in model.model_types:
        assert (
            URI_SIM_TYPE_RES_PATH in model.model_types
        ), f"object '{model.id}' has type '{URI_TYPE_USD_FILE}' but not type '{URI_SIM_TYPE_RES_PATH}'"

        asset_path = None
        for path_model_id in model.model_type_to_id[URI_SIM_TYPE_RES_PATH]:
            asset_path = model.models[path_model_id].get_attr(key=URI_SIM_PRED_PATH)
            if asset_path is not None:
                break
        assert (
            asset_path is not None
        ), f"attr '{URI_SIM_PRED_PATH}' not loaded for object model '{model.id}'"

        usd_model_uris = model.model_type_to_id[URI_TYPE_USD_FILE]

        if URI_SIM_TYPE_ISAAC_RES in model.model_types:
            asset_path = get_cached_assets_root_path() + asset_path

        elif URI_SIM_TYPE_SYS_RES in model.model_types:
            assert os_exists(
                asset_path
            ), f"Path in USD model(s) '{usd_model_uris}' does not exists: {asset_path}"

        else:
            raise RuntimeError(
                f"unhandled types for USD model(s) '{usd_model_uris}': {model.model_types}"
            )

        add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
        return scene.add(RigidPrim(prim_path=prim_path, name=obj_name, **obj_configs))

    if URI_PY_TYPE_MODULE_ATTR in model.model_types:
        correct_cls = None
        for model_id in model.model_type_to_id[URI_PY_TYPE_MODULE_ATTR]:
            python_cls = import_attr_from_model(model=model.models[model_id])
            if issubclass(python_cls, RigidPrim) or issubclass(python_cls, Articulation):
                correct_cls = python_cls
                break
        assert (
            correct_cls is not None
        ), f"'{model.id}' has no handled Python class model: {model.models.keys()}"

        return scene.add(correct_cls(name=obj_name, prim_path=prim_path, **obj_configs))

    raise RuntimeError(f"unhandled types for object'{model.id}': {model.model_types}")
