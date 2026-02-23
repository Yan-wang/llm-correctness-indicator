# OpenEvolve Integration

This directory contains code derived from or inspired by the [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) project.

## Licensing and Attribution

The original code is licensed under the **Apache License, Version 2.0** (the "License"). You may obtain a copy of the License at:

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

* **Original Project:** [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)
* **Copyright:** Copyright (c) 2024 Algorithmics Superintelligence

## Notice of Modification

In compliance with Section 4(b) of the Apache License 2.0, please note that the original OpenEvolve framework has been integrated into this project with the following modifications:

### Core Framework
* **No modifications to core library code**: The OpenEvolve core modules (controller, database, evaluator, LLM ensemble, etc.) remain unchanged from the original repository.
* **Purpose**: Used as-is to provide evolutionary optimization capabilities for evolving confidence scoring algorithms.


### Integration Context
The OpenEvolve framework serves as the evolutionary engine for discovering optimal confidence indicators that distinguish correct from incorrect LLM reasoning traces on AIME/HMMT mathematics problems. The framework's MAP-Elites algorithm and LLM ensemble capabilities enable automated discovery of mathematical primitives and their optimal weightings.

## Citation

If you use this component in your research, please cite the original authors as follows:

```bibtex
@software{openevolve,
  title = {OpenEvolve: an open-source evolutionary coding agent},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/algorithmicsuperintelligence/openevolve}
}
}