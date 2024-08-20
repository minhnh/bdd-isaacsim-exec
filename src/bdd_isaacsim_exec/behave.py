# SPDX-License-Identifier:  GPL-3.0-or-later
from behave import given
from behave.model import Scenario
from behave.runner import Context
from rdflib import Graph
from bdd_dsl.user_story import UserStoryLoader
from bdd_dsl.execution.common import ExecutionModel


def before_all_isaac(context: Context):
    g = getattr(context, "model_graph", None)
    assert g is not None, "'model_graph' attribute not found in context"

    context.execution_model = ExecutionModel(graph=g)
    context.us_loader = UserStoryLoader(graph=g)

    from omni.isaac.kit import SimulationApp

    context.simulation_app = SimulationApp({"headless": False})

    from omni.isaac.core import World

    context.world = World(stage_units_in_meters=1.0)
    context.task = None


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

    assert hasattr(context, "task"), "no 'task' in context"
    if context.task is None:
        # TODO(minhnh): handles different task? Isaacsim expects a single task
        from bdd_isaacsim_exec.tasks import load_isaacsim_task

        task = load_isaacsim_task(
            graph=model_graph, scene=scenario_var_model.scene, task_id=scenario_var_model.task_id
        )
        context.task = task
        context.world.add_task(task)

    context.world.reset()
    # context.behaviour.reset()


@given("a set of objects")
def given_objects_isaacsim(context: Context):
    object_set = set()
    assert context.table is not None, "no table added to context, expected a list of objects"
    for row in context.table:
        object_set.add(row["name"])

    context.objects = object_set


@given("a set of workspaces")
def given_workspaces_isaacsim(context: Context):
    ws_set = set()
    assert context.table is not None, "no table added to context, expected a list of workspaces"
    for row in context.table:
        ws_set.add(row["name"])

    context.workspaces = ws_set


@given("a set of agents")
def given_agents_isaacsim(context: Context):
    agent_set = set()
    assert context.table is not None, "no table added to context, expected a list of agents"
    for row in context.table:
        agent_set.add(row["name"])


@given("specified objects, workspaces and agents are available")
def given_scene_isaacsim(context: Context):
    assert getattr(context, "objects", None) is not None
    assert getattr(context, "workspaces", None) is not None
    assert getattr(context, "agents", None) is not None
    # from bdd_isaacsim_exec.tasks import Pickplace
    # from omni.isaac.franka.controllers import PickPlaceController
