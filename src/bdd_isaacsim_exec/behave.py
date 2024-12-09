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
    PARAM_WS,
    load_obj_models_from_table,
    load_agn_models_from_table,
    load_str_params,
    load_ws_models_from_table,
    parse_str_param,
)
from bdd_dsl.execution.common import ExecutionModel
from bdd_dsl.models.user_story import UserStoryLoader
from bdd_dsl.simulation.common import (
    load_attr_path,
)


def isaacsim_fixture(context: Context, **kwargs: Any):
    from isaacsim import SimulationApp

    unit_length = kwargs.get("unit_length", 1.0)
    headless = context.headless

    print(f"*** STARTING ISAAC SIM, headless={headless}, unit_length={unit_length} ****")
    context.simulation_app = SimulationApp({"headless": headless})

    from omni.isaac.core import World

    context.world = World(stage_units_in_meters=unit_length)

    yield context.simulation_app

    print("*** CLOSING ISAAC SIM ****")
    context.simulation_app.close()


def before_all_isaac(context: Context, headless: bool):
    context.headless = headless
    use_fixture(isaacsim_fixture, context, unit_length=1.0)

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
    scenario_var_model.scene.env_model_loader.register_attr_loaders(
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
    context.task = task
    # context.behaviour.reset()


def after_scenario(context: Context, scenario: Scenario):
    context.task.cleanup_scene_models()
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
        context.task.add_scene_obj_model(obj_model=obj_model)


def given_workspaces_isaac(context: Context):
    assert context.table is not None, "no table added to context, expected a list of object URIs"
    assert context.model_graph is not None, "no 'model_graph' in context, expected an rdflib.Graph"
    assert (
        context.current_scenario is not None
    ), "no 'current_scenario' in context, expected a ScenarioVariantModel"
    for ws_model in load_ws_models_from_table(
        table=context.table, graph=context.model_graph, scene=context.current_scenario.scene
    ):
        context.task.add_scene_ws_model(ws_model=ws_model, graph=context.model_graph)


def given_agents_isaac(context: Context):
    assert context.table is not None, "no table added to context, expected a list of agent URIs"
    assert context.model_graph is not None, "no 'model_graph' in context, expected an rdflib.Graph"
    assert (
        context.current_scenario is not None
    ), "no 'current_scenario' in context, expected an ScenarioVariantModel"
    for agn_model in load_agn_models_from_table(
        table=context.table, graph=context.model_graph, scene=context.current_scenario.scene
    ):
        context.task.add_scene_agn_model(agn_model=agn_model)


def setup_scene_isaac(context: Context):
    # trigger set_up_scene() function in an Isaac Sim BaseTask
    context.world.reset()


def is_located_at_isaac(context: Context, **kwargs):
    params = load_str_params(param_names=[PARAM_OBJ, PARAM_WS, PARAM_EVT], **kwargs)

    assert context.model_graph is not None, "no 'model_graph' in context"
    assert (
        context.current_scenario is not None
    ), "no 'current_scenario' in context, expected a ScenarioVariantModel"

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
        _ = context.current_scenario.scene.load_ws_model(graph=context.model_graph, ws_id=ws_uri)

    evt_uri = try_expand_curie(
        curie_str=params[PARAM_EVT], ns_manager=context.model_graph.namespace_manager, quiet=False
    )
    assert evt_uri is not None, f"can't parse '{params[PARAM_EVT]}' as URI"


def behaviour_isaac(context: Context, **kwargs):
    params = load_str_params(param_names=[PARAM_AGN, PARAM_OBJ, PARAM_WS], **kwargs)
    context.task.set_params(
        agn_id_str=params[PARAM_AGN],
        obj_id_str=params[PARAM_OBJ],
        ws_id_str=params[PARAM_WS],
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
            ns_manager=model_graph.namespace_manager,
        )
        context.behaviour_model = behaviour_model

    bhv = behaviour_model.behaviour
    assert bhv is not None, f"behaviour not processed for {behaviour_model.id}"
    bhv.reset(context=context)
    render = not context.headless
    while context.simulation_app.is_running():
        if bhv.is_finished(context=context):
            break
        context.world.step(render=render)
        context.observations = context.world.get_observations()
        bhv.step(context=context)
