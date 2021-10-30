from __future__ import absolute_import, division, print_function

from absl import logging


class WriteOnceObject(object):
  """A dictionary where all attribute can only be written once."""

  def __setattr__(self, key, val):
    # TODO(yi.sun): Make sure key is valid identifier.
    if key in self.__dict__:
      logging.fatal("Cannot override attribute %s", key)
    self.__dict__[key] = val

  def flatten(self):
    out = []
    WriteOnceObject._flatten_recur(self, out)
    return out

  @staticmethod
  def _flatten_recur(obj, out):
    for k in obj.__dict__:
      v = obj.__dict__[k]
      if isinstance(v, WriteOnceObject):
        WriteOnceObject._flatten_recur(v, out)
        continue
      out.append(v)
