# bdd-isaacsim-exec

## Dependencies

- [minhnh/rdf-utils](https://github.com/minhnh/rdf-utils)
- [minhnh/bdd-dsl](https://github.com/minhnh/bdd-dsl)

## Environment setup

Sample shell setup for running `pip` and `behave` from Nvidia Isaac Sim environment:

```sh
ISAAC_VER='4.2.0'
ISAAC_SIM_DIR="/path/to/omniverse/pkg/isaac-sim-${ISAAC_VER}"
ISAAC_PYTHON_SH="${ISAAC_SIM_DIR}/python.sh"
alias isaacsim-python-sh="${ISAAC_PYTHON_SH}"
alias isaacsim-pip="${ISAAC_PYTHON_SH} -m pip"
alias isaacsim-ipython="${ISAAC_PYTHON_SH} ${ISAAC_SIM_DIR}/kit/python/bin/ipython"
alias isaacsim-behave="${ISAAC_PYTHON_SH} ${ISAAC_SIM_DIR}/kit/python/bin/behave"
```

This allows running `isaacsim-pip` to install to the Isaac Python package
location and `isaacsim-behave` to load packages from this location.
