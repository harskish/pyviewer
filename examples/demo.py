from pathlib import Path
import pyviewer # python setup.py develop

# Don't accidentally test different version
assert Path(pyviewer.__file__).parents[1] == Path(__file__).parents[1], \
    'Not running local editable install, please run "pip uninstall -y pyviewer && python setup.py develop"'

# Run built in demo
pyviewer.toolbar_viewer.main()