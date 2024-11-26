# SPDX-License-Identifier:  GPL-3.0-or-later
from rdflib import Namespace


NS_MM_USD = Namespace("https://openusd.org#")
NS_MM_ISAACSIM = Namespace("https://developer.nvidia.com/isaac/sim#")
NS_FRANKA = Namespace("https://www.franka.de/")
NS_UR = Namespace("https://www.universal-robots.com/products/")

URI_TYPE_USD_FILE = NS_MM_USD["UsdFile"]
URI_SIM_TYPE_ISAAC_RES = NS_MM_ISAACSIM["IsaacResource"]
URI_FRANKA_PANDA = NS_FRANKA["emika-panda"]
URI_UR_UR10 = NS_UR["ur10-robot"]
