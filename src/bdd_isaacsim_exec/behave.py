# SPDX-License-Identifier:  GPL-3.0-or-later
from typing import Any
import time
import numpy as np
from behave import use_fixture
from behave.model import Scenario
from behave.runner import Context
from rdflib import Graph, URIRef
from rdf_utils.uri import try_expand_curie
from rdf_utils.models.python import (
    load_py_module_attr,
)
from bdd_dsl.behave import (
    PARAM_AGN,
    PARAM_EVT,
    PARAM_OBJ,
    PARAM_WS,
    ParamType,
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


DIST_THRESHOLD = 0.1
SPEED_THRESHOLD = 1


def isaacsim_fixture(context: Context, **kwargs: Any):
    from isaacsim import SimulationApp

    unit_length = kwargs.get("unit_length", 1.0)
    headless = context.headless
    time_step_sec = context.time_step_sec

    print(f"*** STARTING ISAAC SIM, headless={headless}, unit_length={unit_length} ****")
    context.simulation_app = SimulationApp({"headless": headless})

    from omni.isaac.core import World

    context.world = World(stage_units_in_meters=unit_length, physics_dt=time_step_sec)

    yield context.simulation_app

    print("*** CLOSING ISAAC SIM ****")
    context.simulation_app.close()


def before_all_isaac(context: Context, headless: bool, time_step_sec: float):
    context.headless = headless
    context.time_step_sec = time_step_sec
    use_fixture(isaacsim_fixture, context, unit_length=1.0)

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
    from bdd_isaacsim_exec.tasks import MeasurementType

    params = load_str_params(param_names=[PARAM_OBJ, PARAM_WS, PARAM_EVT], **kwargs)

    assert context.model_graph is not None, "no 'model_graph' in context"
    ns_manager = context.model_graph.namespace_manager

    assert (
        context.current_scenario is not None
    ), "no 'current_scenario' in context, expected a ScenarioVariantModel"

    _, obj_uris = parse_str_param(param_str=params[PARAM_OBJ], ns_manager=ns_manager)
    assert (
        len(obj_uris) == 1
    ), f"is_located_at_isaac: expected 1 obj, got '{len(obj_uris)}': {obj_uris}"
    obj_id = obj_uris[0]
    assert isinstance(obj_id, URIRef), f"obj id not a URI: {obj_id}"
    context.task.add_measurement(elem_id=obj_id, meas_type=MeasurementType.OBJ_POSE)

    ws_param_type, ws_uris = parse_str_param(param_str=params[PARAM_WS], ns_manager=ns_manager)
    for ws_id in ws_uris:
        assert isinstance(ws_id, URIRef), f"ws id not a URI: {ws_id}"
        context.task.add_measurement(elem_id=ws_id, meas_type=MeasurementType.WS_BOUNDS)

    obs = context.world.get_observations()
    assert obj_id in obs, f"is_located_at_isaac: no measurement for obj '{obj_id.n3(ns_manager)}'"
    obj_position = obs[obj_id]["position"]

    is_located = True
    if ws_param_type == ParamType.EXISTS_SET:
        # Or comparision initialized to false
        is_located = False

    all_bounds = []
    for ws_id in ws_uris:
        assert isinstance(ws_id, URIRef), f"ws id not a URI: {ws_id}"
        assert ws_id in obs, f"is_located_at_isaac: no measurement for ws '{ws_id.n3(ns_manager)}'"
        assert (
            "bounds" in obs[ws_id]
        ), f"bounds not loaded for ws '{ws_id.n3(ns_manager)}, meas: {obs[ws_id]}'"
        ws_bounds = obs[ws_id]["bounds"]
        all_bounds.append(ws_bounds)

        is_above = np.all(np.greater(obj_position, ws_bounds[:3]))
        is_below = np.all(np.less(obj_position, ws_bounds[3:] + DIST_THRESHOLD))
        if ws_param_type == ParamType.EXISTS_SET:
            is_located |= is_above and is_below
        else:
            is_located &= is_above and is_below

    assert is_located, f"obj '{obj_id.n3(ns_manager)}' (pos={obj_position}) not in workspace(s), bounds: {all_bounds}"

    evt_uri = try_expand_curie(curie_str=params[PARAM_EVT], ns_manager=ns_manager, quiet=False)
    assert evt_uri is not None, f"can't parse '{params[PARAM_EVT]}' as URI"


def move_safely_isaac(context: Context, **kwargs):
    assert context.model_graph is not None, "no 'model_graph' in context"
    params = load_str_params(param_names=[PARAM_AGN], **kwargs)

    assert hasattr(context, "agent_max_speed"), "move_safely_isaac: no 'agent_max_speed' in context"
    assert (
        context.agent_max_speed < SPEED_THRESHOLD
    ), f"agent '{params[PARAM_AGN]}' moves EE too fast: {context.agn_max_speed} > {SPEED_THRESHOLD}"


def behaviour_isaac(context: Context, **kwargs):
    from bdd_isaacsim_exec.tasks import MeasurementType

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

    render = not context.headless

    bhv = behaviour_model.behaviour
    assert bhv is not None, f"behaviour not processed for {behaviour_model.id}"
    bhv.reset(context=context)

    # Move safely clause requires assertion over a time horizon
    _, agn_uris = parse_str_param(
        param_str=params[PARAM_AGN], ns_manager=context.model_graph.namespace_manager
    )
    assert len(agn_uris) == 1 and isinstance(
        agn_uris[0], URIRef
    ), f"unexpected agn params: {agn_uris}"
    context.task.add_measurement(elem_id=agn_uris[0], meas_type=MeasurementType.AGN_EE_LINEAR_VEL)
    agn_speeds = []
    agn_max_speed = -1

    time_step_sec = context.time_step_sec
    now = time.process_time()
    loop_end = now
    exec_times = []
    while context.simulation_app.is_running():
        if bhv.is_finished(context=context):
            break
        context.world.step(render=render)
        # observations
        obs = context.world.get_observations()
        agn_speed = np.linalg.norm(obs[agn_uris[0]]["ee_linear_velocities"])
        if agn_speed > agn_max_speed:
            agn_max_speed = agn_speed
        agn_speeds.append(agn_speed)
        context.observations = obs
        # behaviour step
        bhv.step(context=context)
        # real time check
        exec_times.append(time.process_time() - now)
        loop_end += time_step_sec
        while now < loop_end:
            now = time.process_time()

    context.agent_max_speed = agn_max_speed

    print(
        "\n\n*** Agent speed statistics: "
        + f" mean={np.mean(agn_speeds):.5f}, std={np.std(agn_speeds):.5f},"
        + f" min={min(agn_speeds):.5f}, max={min(agn_speeds):.5f}\n\n"
    )
    print(
        f"\n\n*** Execution time statistics (secs) for '{len(exec_times)}' loops:"
        + f" mean={np.mean(exec_times):.5f}, std={np.std(exec_times):.5f},"
        + f" min={min(exec_times):.5f}, max={min(exec_times):.5f}\n\n"
    )
