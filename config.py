import yaml
import sys

cfg = []

""" For loader tuple"""
class YAMLPatch(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

YAMLPatch.add_constructor(u'tag:yaml.org,2002:python/tuple', YAMLPatch.construct_python_tuple)

def load_config(f_name = "config/default.yaml"):
    global cfg
    print("I: Load {}".format(f_name))
    """ Load File"""
    with open(f_name, 'r') as stream:
        try:
            cfg = yaml.load(stream, Loader=YAMLPatch)
            # print(yaml.dump(cfg))
        except yaml.YAMLError as exc:
            print(exc)


if len(sys.argv) >= 2 and sys.argv[1].endswith('.yaml'):
    load_config(sys.argv[1])
else:
    load_config()