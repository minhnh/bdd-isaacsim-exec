# SPDX-License-Identifier:  GPL-3.0-or-later
from rdflib import Namespace


NS_MM_USD = Namespace("https://openusd.org#")
NS_MM_ISAACSIM = Namespace("https://developer.nvidia.com/isaac/sim#")

URI_TYPE_USD_FILE = NS_MM_USD["UsdFile"]
URI_SIM_TYPE_ISAAC_RES = NS_MM_ISAACSIM["IsaacResource"]
