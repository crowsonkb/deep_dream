"""Miscellaneous utilities for the CLI interfaces."""

import sys

import click


def setup_traceback(mode='Plain', color_scheme='Neutral', require=False, **kwargs):
    """If IPython is available, sets up IPython-style verbose exception tracebacks. There are three
    modes: in order of increasing verbosity, Plain, Context, and Verbose. As of IPython 5, there
    are four available color schemes: NoColor, Neutral, Linux, and LightBG."""
    try:
        from IPython.core import ultratb
        sys.excepthook = ultratb.FormattedTB(mode, color_scheme, **kwargs)
    except ImportError as err:
        if require:
            raise err


class IntList(click.ParamType):
    """A Click parameter type that converts a comma-separated list of ints to a list."""
    name = 'int_list'
    sep = ','

    def convert(self, value, param, ctx):
        lst = []
        if value:
            try:
                items = value.split(self.sep)
                for item in items:
                    lst.append(int(item.strip()))
            except ValueError:
                self.fail('%s is not a valid comma separated list of integers' % value, param, ctx)
        return lst
