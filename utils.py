"""Miscellaneous utilities for the CLI interfaces."""

import logging
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


class List(click.ParamType):
    """A Click parameter type: a comma-separated list."""
    def __init__(self, converter, type_name=None, sep=','):
        self.converter = converter
        self.name = 'list'
        self.sep = sep
        if type_name:
            self.name = type_name + '_list'

    def convert(self, value, param, ctx):
        lst = []
        if value:
            try:
                items = value.split(self.sep)
                for item in items:
                    lst.append(self.converter(item))
            except (TypeError, ValueError):
                self.fail('%s is not a valid comma separated %s' % (value, self.name), param, ctx)
        return lst


class ColorFormatter(logging.Formatter):
    colors = {
        'critical': dict(fg='red', bold=True),
        'error': dict(fg='red'),
        'exception': dict(fg='red'),
        'warning': dict(fg='yellow'),
        'debug': dict(fg='blue'),
    }

    def format(self, record):
        if not record.exc_info:
            level = record.levelname.lower()
            if level in self.colors:
                prefix = click.style('{}: '.format(level),
                                     **self.colors[level])
                record.msg = '\n'.join(prefix + x
                                       for x in str(record.msg).splitlines())

        return logging.Formatter.format(self, record)
