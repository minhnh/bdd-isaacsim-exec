# SPDX-License-Identifier:  GPL-3.0-or-later
from typing import Any
from rdflib import Graph
from behave import given, then, when, use_fixture
from behave.model import Scenario
from behave.runner import Context
from rdf_utils.models.python import (
    URI_PY_PRED_ATTR_NAME,
    URI_PY_PRED_MODULE_NAME,
    URI_PY_TYPE_MODULE_ATTR,
    load_py_module_attr,
)
from bdd_dsl.behave import given_ws_models, load_obj_models_from_table, load_agn_models_from_table
from bdd_dsl.execution.common import ExecutionModel
from bdd_dsl.models.user_story import UserStoryLoader
from bdd_dsl.simulation.common import (
    URI_SIM_PRED_PATH,
    URI_SIM_TYPE_RES_PATH,
    load_attr_has_config,
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
        load_attr_path, load_attr_has_config, load_py_module_attr
    )
    scenario_var_model.scene.agn_model_loader.register_attr_loaders(
        load_attr_path, load_attr_has_config, load_py_module_attr
    )
    context.current_scenario = scenario_var_model

    # TODO(minhnh): handles different task? Isaacsim expects a single task
    from bdd_isaacsim_exec.tasks import load_isaacsim_task

    task = load_isaacsim_task(world=context.world, graph=model_graph, scenario=scenario_var_model)
    print(f"**** Loaded Isaac Sim Task {task.name}")
    context.world.reset()
    # context.behaviour.reset()


def after_scenario(context: Context, scenario: Scenario):
    context.world.clear()


@given("a set of objects")
def given_objects_isaacsim(context: Context):
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


given("a set of workspaces")(given_ws_models)


@given("a set of agents")
def given_agents_mockup(context: Context):
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


@given("specified objects, workspaces and agents are available")
def given_scene_isaacsim(context: Context):
    pass


@given('"{pick_obj}" is located at "{pick_ws}" before event "{evt}"')
@then('"{pick_obj}" is located at "{pick_ws}" after event "{evt}"')
def is_located_at_mockup_given(context: Context, pick_obj: str, pick_ws: str, evt: str):
    try:
        pick_obj_uri = context.model_graph.namespace_manager.expand_curie(pick_obj)
    except ValueError as e:
        raise RuntimeError(f"can't parse pick target obj URI '{pick_obj}': {e}")

    try:
        pick_ws_uri = context.model_graph.namespace_manager.expand_curie(pick_ws)
    except ValueError as e:
        raise RuntimeError(f"can't parse pick workspace URI '{pick_ws}': {e}")

    try:
        _ = context.model_graph.namespace_manager.expand_curie(evt)
    except ValueError as e:
        raise RuntimeError(f"can't parse event URI '{evt}': {e}")

    assert (
        pick_obj_uri in context.current_scenario.scene.objects
    ), f"object '{pick_obj_uri}' unrecognized"
    assert pick_ws_uri in context.workspaces, f"workspace '{pick_ws}' unrecognized"


@when('behaviour "{bhv_name}" occurs')
def behaviour_mockup(context: Context, bhv_name: str):
    _ = getattr(context, "behaviour_model", None)
    for _ in range(200):
        context.world.step()
