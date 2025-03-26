# bdd-isaacsim-exec

Repository for executing acceptance tests of sample pick & place application in [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim),
using the [bdd-dsl](https://github.com/minhnh/bdd-dsl) framework.

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

## Usage

1. [Download & install Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html)
2. Configure the environment using instructions above
3. Install [rdf-utils](https://github.com/minhnh/rdf-utils) & [bdd-dsl](https://github.com/minhnh/bdd-dsl) with `isaacsim-pip` alias
4. Generate the Gherkin feature files per instruction on the [bdd-dsl](https://github.com/minhnh/bdd-dsl) repository.
5. Copy/link the generated Gherkin into the [`examples`](./examples) directory
6. Run `isaacsim-behave` alias under the `examples` folder
