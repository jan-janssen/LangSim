from llm_compmat.shell import dialog as dialog_python
from llm_compmat.widget import dialog as dialog_widget


def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    from .magics import CompMatMagics
    ipython.register_magics(CompMatMagics)