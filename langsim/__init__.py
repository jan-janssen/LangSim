from langsim.shell import dialog as dialog_python


def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    from .magics import CompMatMagics
    ipython.register_magics(CompMatMagics)