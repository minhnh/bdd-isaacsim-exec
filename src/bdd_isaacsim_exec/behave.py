# SPDX-License-Identifier:  GPL-3.0-or-later
from typing import Any
from rdflib import Graph
from behave import use_fixture
from behave.model import Scenario
from behave.runner import Context
from rdf_utils.models.common import ModelLoader
from rdf_utils.uri import try_expand_curie
from rdf_utils.models.python import (
    URI_PY_PRED_ATTR_NAME,
    URI_PY_PRED_MODULE_NAME,
    URI_PY_TYPE_MODULE_ATTR,
    load_py_module_attr,
)
from bdd_dsl.behave import (
    PARAM_AGN,
    PARAM_EVT,
    PARAM_OBJ,
    PARAM_PICK_WS,
    PARAM_PLACE_WS,
    PARAM_WS,
    load_obj_models_from_table,
    load_agn_models_from_table,
    load_str_params,
    parse_str_param,
)
from bdd_dsl.execution.common import ExecutionModel
from bdd_dsl.models.user_story import UserStoryLoader
from bdd_dsl.simulation.common import (
    URI_SIM_PRED_PATH,
    URI_SIM_TYPE_RES_PATH,
    load_attr_path,
)


def isaacsim_fixture(context: Context, **kwargs: Any):
    from isaacsim import SimulationApp

    headless = kwargs.get("headless", False)
    unit_length = kwargs.get("unit_length", 1.0)

    print("*** STARTING ISAAC SIM ****")
    context.simulation_app = SimulationApp({"headless": headless})

    from omni.isaac.core import World

    context.world = World(stage_units_in_meters=unit_length)

    yield context.simulation_app

    print("*** CLOSING ISAAC SIM ****")
    context.simulation_app.close()


def before_all_isaac(context: Context, headless: bool = False):
    use_fixture(isaacsim_fixture, context, headless=headless, unit_length=1.0)

    g = getattr(context, "model_graph", None)
    assert g is not None, "'model_graph' attribute not found in context"

    context.execution_model = ExecutionModel(graph=g)
    context.us_loader = UserStoryLoader(graph=g)
    context.ws_model_loader = ModelLoader()


def before_scenario(context: Context, scenario: Scenario):
    model_graph = getattr(context, "model_graph", None)
    assert model_graph is not None and isinstance(model_graph, Graph)

    us_loader = getattr(context, "us_loader", None)
    assert us_loader is not None and isinstance(us_loader, UserStoryLoader)

    # scenario outline renders each scenario as
    #   SCHEMA: "{outline_name} -- {examples.name}@{row.id}"
    scr_name_splits = scenario.name.split(" -- ")
    assert len(scr_name_splits) > 0, f"unexpected scenario name: {scenario.name}"
    scr_name = scr_name_splits[0]
    try:
        scenario_var_uri = model_graph.namespace_manager.expand_curie(scr_name)
    except ValueError as e:
        raise RuntimeError(
            f"can't parse behaviour URI '{scr_name}' from scenario '{scenario.name}': {e}"
        )

    scenario_var_model = us_loader.load_scenario_variant(
        full_graph=model_graph, variant_id=scenario_var_uri
    )
    scenario_var_model.scene.obj_model_loader.register_attr_loaders(
        load_attr_path, load_py_module_attr
    )
    scenario_var_model.scene.agn_model_loader.register_attr_loaders(
        load_attr_path, load_py_module_attr
    )
    context.current_scenario = scenario_var_model

    # TODO(minhnh): handles different task? Isaacsim expects a single task
    from bdd_isaacsim_exec.tasks import load_isaacsim_task

    task = load_isaacsim_task(world=context.world, graph=model_graph, scr_var=scenario_var_model)
    print(f"**** Loaded Isaac Sim Task {task.name}")
    context.world.reset()
    # context.behaviour.reset()


def after_scenario(context: Context, scenario: Scenario):
    context.world.clear()


def given_objects_isaac(context: Context):
    assert context.table is not None, "no table added to context, expected a list of object URIs"
    assert context.model_graph is not None, "no 'model_graph' in context, expected an rdflib.Graph"
    assert (
        context.current_scenario is not None
    ), "no 'current_scenario' in context, expected a ScenarioVariantModel"
    for obj_model in load_obj_models_from_table(
        table=context.table, graph=context.model_graph, scene=context.current_scenario.scene
    ):
        if URI_PY_TYPE_MODULE_ATTR in obj_model.model_types:
            for py_model_uri in obj_model.model_type_to_id[URI_PY_TYPE_MODULE_ATTR]:
                py_model = obj_model.models[py_model_uri]
                assert py_model.has_attr(
                    key=URI_PY_PRED_MODULE_NAME
                ), f"Python attribute model '{py_model.id}' for object '{obj_model.id}' missing module name"
                assert py_model.has_attr(
                    key=URI_PY_PRED_ATTR_NAME
                ), f"Python attribute model '{py_model.id}' for object '{obj_model.id}' missing attribute name"

        if URI_SIM_TYPE_RES_PATH in obj_model.model_types:
            for py_model_uri in obj_model.model_type_to_id[URI_SIM_TYPE_RES_PATH]:
                path_model = obj_model.load_first_model_by_type(model_type=URI_SIM_TYPE_RES_PATH)
                assert path_model.has_attr(
                    URI_SIM_PRED_PATH
                ), f"ResourceWithPath model '{path_model.id}' for object '{obj_model.id}' missing attr path"


def given_agents_isaac(context: Context):
    assert context.table is not None, "no table added to context, expected a list of agent URIs"
    assert context.model_graph is not None, "no 'model_graph' in context, expected an rdflib.Graph"
    assert (
        context.current_scenario is not None
    ), "no 'current_scenario' in context, expected an ScenarioVariantModel"
    for agn_model in load_agn_models_from_table(
        table=context.table, graph=context.model_graph, scene=context.current_scenario.scene
    ):
        if URI_PY_TYPE_MODULE_ATTR in agn_model.model_types:
            for py_model_uri in agn_model.model_type_to_id[URI_PY_TYPE_MODULE_ATTR]:
                py_model = agn_model.models[py_model_uri]
                assert py_model.has_attr(
                    key=URI_PY_PRED_MODULE_NAME
                ), f"Python attribute model '{py_model.id}' for agent '{agn_model.id}' missing module name"
                assert py_model.has_attr(
                    key=URI_PY_PRED_ATTR_NAME
                ), f"Python attribute model '{py_model.id}' for agent '{agn_model.id}' missing attribute name"


def is_located_at_isaac(context: Context, **kwargs):
    params = load_str_params(param_names=[PARAM_OBJ, PARAM_WS, PARAM_EVT], **kwargs)

    assert context.model_graph is not None, "no 'model_graph' in context"
    assert (
        context.current_scenario is not None
    ), "no 'current_scenario' in context, expected an ObjModelLoader"

    _, pick_obj_uris = parse_str_param(
        param_str=params[PARAM_OBJ], ns_manager=context.model_graph.namespace_manager
    )
    for obj_uri in pick_obj_uris:
        obj_model = context.current_scenario.scene.load_obj_model(
            graph=context.model_graph, obj_id=obj_uri
        )
        assert obj_model is not None, f"can't load model for object {obj_uri}"
        if URI_PY_TYPE_MODULE_ATTR in obj_model.model_types:
            py_model = obj_model.load_first_model_by_type(URI_PY_TYPE_MODULE_ATTR)
            assert py_model.has_attr(
                key=URI_PY_PRED_MODULE_NAME
            ), f"Python attribute model '{py_model.id}' for object '{obj_model.id}' missing module name"
            assert py_model.has_attr(
                key=URI_PY_PRED_ATTR_NAME
            ), f"Python attribute model '{py_model.id}' for object '{obj_model.id}' missing attribute name"

    _, pick_ws_uris = parse_str_param(
        param_str=params[PARAM_WS], ns_manager=context.model_graph.namespace_manager
    )
    for ws_uri in pick_ws_uris:
        assert ws_uri in context.workspaces, f"workspace '{ws_uri}' unrecognized"

    evt_uri = try_expand_curie(
        curie_str=params[PARAM_EVT], ns_manager=context.model_graph.namespace_manager, quiet=False
    )
    assert evt_uri is not None, f"can't parse '{params[PARAM_EVT]}' as URI"


def behaviour_isaac(context: Context, **kwargs):
    params = load_str_params(
        param_names=[PARAM_AGN, PARAM_OBJ, PARAM_PICK_WS, PARAM_PLACE_WS], **kwargs
    )

    behaviour_model = getattr(context, "behaviour_model", None)

    if behaviour_model is None:
        exec_model = getattr(context, "execution_model", None)
        assert isinstance(
            exec_model, ExecutionModel
        ), f"no valid 'execution_model' added to the context: {exec_model}"

        model_graph = getattr(context, "model_graph", None)
        assert isinstance(
            model_graph, Graph
        ), f"no 'model_graph' of type rdflib.Graph in context: {model_graph}"

        behaviour_model = exec_model.load_behaviour_impl(
            context=context,
            agn_id_str=params[PARAM_AGN],
            obj_id_str=params[PARAM_OBJ],
            pick_ws_str=params[PARAM_PICK_WS],
            place_ws_str=params[PARAM_PLACE_WS],
            ns_manager=model_graph.namespace_manager,
        )
        context.behaviour_model = behaviour_model

    bhv = behaviour_model.behaviour
    assert bhv is not None
    bhv.reset(context=context)
    while not bhv.is_finished(context=context):
        bhv.step(context=context)
