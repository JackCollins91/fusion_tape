"""Load the repository folder as the package ``fusion_tape``, whatever the folder is called."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if 'fusion_tape' not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        'fusion_tape', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['fusion_tape'] = module
    spec.loader.exec_module(module)
