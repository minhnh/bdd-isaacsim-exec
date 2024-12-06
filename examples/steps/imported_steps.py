from behave import given, then, when
from bdd_dsl.behave import (
    CLAUSE_BG_AGENTS,
    CLAUSE_BG_OBJECTS,
    CLAUSE_BG_WORKSPACES,
    CLAUSE_BHV_PICKPLACE,
    CLAUSE_FL_LOCATED_AT,
    CLAUSE_TC_AFTER_EVT,
    CLAUSE_TC_BEFORE_EVT,
    given_ws_models,
)
from bdd_isaacsim_exec.behave import (
    given_objects_isaac,
    given_agents_isaac,
    is_located_at_isaac,
    behaviour_isaac,
)

given(CLAUSE_BG_OBJECTS)(given_objects_isaac)
given(CLAUSE_BG_WORKSPACES)(given_ws_models)
given(CLAUSE_BG_AGENTS)(given_agents_isaac)

given(f"{CLAUSE_FL_LOCATED_AT} {CLAUSE_TC_BEFORE_EVT}")(is_located_at_isaac)
given(f"{CLAUSE_FL_LOCATED_AT} {CLAUSE_TC_AFTER_EVT}")(is_located_at_isaac)
then(f"{CLAUSE_FL_LOCATED_AT} {CLAUSE_TC_BEFORE_EVT}")(is_located_at_isaac)
then(f"{CLAUSE_FL_LOCATED_AT} {CLAUSE_TC_AFTER_EVT}")(is_located_at_isaac)

when(CLAUSE_BHV_PICKPLACE)(behaviour_isaac)
