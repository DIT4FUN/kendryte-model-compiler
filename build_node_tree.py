leaf_table = ('MaxPool2D', 'Conv', 'Relu6', 'Relu', 'LeakyRelu', 'BatchNorm', 'SeparableConv2d')


def tree_append(tree, paths_rev, item):
    if not isinstance(tree, dict):
        return tree_append({}, paths_rev, item)
    else:
        cur = paths_rev.pop()
        if len(paths_rev):
            old_tree = tree.get(cur, {})
            tree[cur] = tree_append(old_tree, paths_rev, item)
        else:
            tree[cur] = item
        return tree


def build_node_tree(nodes_by_name):
    ret = {}
    for name, item in nodes_by_name.items():
        path_rev = name.split('/')
        path_rev.reverse()
        tree_append(ret, path_rev, item)

    return ret


class LeafNode:
    def __init__(self, tree, ty):
        self.tree = tree
        self.ty = ty
        self.input = None
        self.output = None

        # {
        #     'MaxPool2D': self.io_MaxPool2D,
        #     'Conv': self.io_Conv,
        #     'Relu': self.io_Relu,
        #     'Relu6': self.io_Relu6,
        #     'LeakyRelu': self.io_LeakyRelu,
        #     'BatchNorm': self.io_BatchNorm,
        #     'SeparableConv2d': self.io_SeparableConv2d,
        # }[self.ty]()

    def io_MaxPool2D(self):
        self.input = self.tree['MaxPool'].inputs[0]
        self.output = self.tree['MaxPool']

    def io_Conv(self):
        self.input = self.tree['Conv2D'].inputs[0]
        self.output = self.tree['Relu']

    def io_Relu(self):
        pass

    def io_Relu6(self):
        pass

    def io_LeakyRelu(self):
        pass

    def io_BatchNorm(self):
        pass

    def io_SeparableConv2d(self):
        pass


        # ('MaxPool2D', 'Conv', 'Relu6', 'BatchNorm', 'SeparableConv2d')


def tree_flatten(ret, tree, prefix=None):
    prefix = prefix or []
    for name,item in tree.items():
        name_is_leaf = False
        leaf_ty = None
        for ty in leaf_table:
            if name.startswith(ty):
                name_is_leaf = True
                leaf_ty = ty
                break

        if name_is_leaf:
            ret['/'.join([*prefix,name])] = LeafNode(item, leaf_ty)
        elif isinstance(item, dict):
            tree_flatten(ret, item, [*prefix, name])
        else:
            ret[item.name] = LeafNode(item, item.type)
