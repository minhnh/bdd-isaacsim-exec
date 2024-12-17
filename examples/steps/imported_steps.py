from behave import given, then, when
from bdd_dsl.behave import (
    CLAUSE_BG_AGENTS,
    CLAUSE_BG_OBJECTS,
    CLAUSE_BG_WORKSPACES,
    CLAUSE_BG_SCENE,
    CLAUSE_BHV_PICKPLACE,
    CLAUSE_FL_LOCATED_AT,
    CLAUSE_FL_MOVE_SAFE,
    CLAUSE_TC_AFTER_EVT,
    CLAUSE_TC_BEFORE_EVT,
    CLAUSE_TC_DURING,
)
from bdd_isaacsim_exec.behave import (
    given_objects_isaac,
    given_workspaces_isaac,
    given_agents_isaac,
    is_located_at_isaac,
    move_safely_isaac,
    behaviour_isaac,
    setup_scene_isaac,
)

given(CLAUSE_BG_OBJECTS)(given_objects_isaac)
given(CLAUSE_BG_WORKSPACES)(given_workspaces_isaac)
given(CLAUSE_BG_AGENTS)(given_agents_isaac)
given(CLAUSE_BG_SCENE)(setup_scene_isaac)

given(f"{CLAUSE_FL_LOCATED_AT} {CLAUSE_TC_BEFORE_EVT}")(is_located_at_isaac)
given(f"{CLAUSE_FL_LOCATED_AT} {CLAUSE_TC_AFTER_EVT}")(is_located_at_isaac)
then(f"{CLAUSE_FL_LOCATED_AT} {CLAUSE_TC_BEFORE_EVT}")(is_located_at_isaac)
then(f"{CLAUSE_FL_LOCATED_AT} {CLAUSE_TC_AFTER_EVT}")(is_located_at_isaac)
then(f"{CLAUSE_FL_MOVE_SAFE} {CLAUSE_TC_DURING}")(move_safely_isaac)

when(CLAUSE_BHV_PICKPLACE)(behaviour_isaac)
