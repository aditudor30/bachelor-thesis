"""Simple console formatting helpers."""

from typing import Any, Dict


def print_dict(data: Dict[str, Any], indent: int = 0) -> None:
    """Print nested dictionaries in a readable deterministic format."""
    prefix = " " * indent
    for key in sorted(data.keys()):
        value = data[key]
        if isinstance(value, dict):
            print("%s%s:" % (prefix, key))
            print_dict(value, indent + 2)
        else:
            print("%s%s: %s" % (prefix, key, value))

