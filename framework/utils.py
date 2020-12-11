import argparse

class YamlNamespace(argparse.Namespace):
    """Namespace from a nested dict returned by yaml.load()"""
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [YamlNamespace(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, YamlNamespace(b) if isinstance(b, dict) else b)
