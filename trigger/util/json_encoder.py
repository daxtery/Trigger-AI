import copy
import dataclasses
from enum import Enum
import json
from trigger.transformers.transformer_pipeline import Instance
from trigger.scoring import ScoringCalculator

import numpy

# FIXME: Huge hack... From https://github.com/python/cpython/blob/6b1ac809b9718a369aea67b99077cdd682be2238/Lib/dataclasses.py#L1095

def _is_dataclass_instance(obj):
    """Returns True if obj is an instance of a dataclass."""
    return hasattr(type(obj), '__dataclass_fields__')


def asdict(obj, *, dict_factory=dict):
    """Return the fields of a dataclass instance as a new dictionary mapping
    field names to field values.
    Example usage:
      @dataclass
      class C:
          x: int
          y: int
      c = C(1, 2)
      assert asdict(c) == {'x': 1, 'y': 2}
    If given, 'dict_factory' will be used instead of built-in dict.
    The function applies recursively to field values that are
    dataclass instances. This will also look into built-in containers:
    tuples, lists, and dicts.
    """
    if not _is_dataclass_instance(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, dict_factory)


def _asdict_inner(obj, dict_factory):
    if _is_dataclass_instance(obj):
        result = []
        for f in dataclasses.fields(obj):
            if f.repr:
                value = _asdict_inner(getattr(obj, f.name), dict_factory)
                result.append((f.name, value))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict()
        # method, because:
        # - it does not recurse in to the namedtuple fields and
        #   convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The main
        #   use case here is json.dumps, and it handles converting
        #   namedtuples to lists.  Admittedly we're losing some
        #   information here when we produce a json list instead of a
        #   dict.  Note that if we returned dicts here instead of
        #   namedtuples, we could no longer call asdict() on a data
        #   structure where a namedtuple was used as a dict key.

        return type(obj)(*[_asdict_inner(v, dict_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_asdict_inner(v, dict_factory) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_asdict_inner(k, dict_factory),
                          _asdict_inner(v, dict_factory))
                         for k, v in obj.items())
    else:
        return copy.deepcopy(obj)

# FIXME: Hack over...

class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return asdict(o)
            if isinstance(o, Enum):
                return o.name
            if type(o).__module__ == numpy.__name__:
                if isinstance(o, numpy.ndarray):
                    return o.tolist()
                else:
                    return o.item()
            if isinstance(o, ScoringCalculator):
                return o.describe()
            return super().default(o)