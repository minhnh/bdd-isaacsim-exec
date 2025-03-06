import sys
from json import JSONDecodeError
from rdflib import ConjunctiveGraph
from behave.runner import Context
from rdf_utils.uri import URL_SECORO_M
from rdf_utils.resolver import install_resolver
from bdd_isaacsim_exec.behave import before_all_isaac, before_scenario, after_scenario


MODELS = {
    f"{URL_SECORO_M}/acceptance-criteria/bdd/agents/isaac-sim.agn.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/scenes/isaac-agents.scene.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/environments/secorolab.env.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/environments/secorolab.env.geom.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/environments/avl.env.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/environments/avl.env.geom.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/simulation/secorolab-isaac.sim.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/simulation/secorolab.sim.geom.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/simulation/secorolab.sim.geom.distrib.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/scenes/secorolab-env.scene.json": "json-ld",
    # f"{URL_SECORO_M}/acceptance-criteria/bdd/templates/pickplace.tmpl.json": "json-ld",
    # f"{URL_SECORO_M}/acceptance-criteria/bdd/pickplace-secorolab-isaac.var.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/templates/sorting.tmpl.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/sorting-secorolab-isaac.var.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/pickplace-secorolab-isaac.exec.json": "json-ld",
}


def before_all(context: Context):
    install_resolver()
    g = ConjunctiveGraph()
    for url, fmt in MODELS.items():
        try:
            g.parse(url, format=fmt)
        except JSONDecodeError as e:
            print(f"error parsing '{url}' into graph (format='{fmt}'):\n{e}")
            sys.exit(1)

    context.model_graph = g
    before_all_isaac(context, headless=True)
