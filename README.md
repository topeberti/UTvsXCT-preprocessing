# preprocess_tools

This module provides utilities for preprocessing and analyzing 3D XCT and UT data, including volume alignment, pore detection, and material segmentation for the UT vs XCT project.

## Installation and Updating

### Local installation

To install the package locally, run:

```bash
pip install .
```

To update the package after making changes locally, run:

```bash
pip install --upgrade .
```

### Remote installation (from GitHub)

To install directly from the repository, run:

```bash
pip install git+https://github.com/topeberti/UTvsXCT-preprocessing.git
```

To update to the latest version from the repository, run:

```bash
pip install --upgrade git+https://github.com/topeberti/UTvsXCT-preprocessing.git
```

## Importing the module

After installation, you can import the module in your Python scripts as follows:

```python
import preprocess_tools
```

Or import specific functions or classes:

```python
from preprocess_tools import your_function_or_class
```

## Documentation

You can generate and view the documentation for this package using [pdoc](https://pdoc.dev/):

### Generate and view documentation locally

1. Install the requirements (if not already done):

```bash
pip install -r requirements.txt
```

2. Generate and serve the documentation for the `preprocess_tools` module:

```bash
pdoc .\preprocess_tools
```

3. This will start a local web server and open your browser to view the documentation. If it does not open automatically, follow the link shown in the terminal (usually http://localhost:8080).

4. To generate static HTML files instead, run:

```bash
pdoc -o docs_html .\preprocess_tools
```

This will create HTML documentation in the `docs_html` folder.


