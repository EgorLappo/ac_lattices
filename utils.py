# author: egor lappo
# egor (at) ccrma.stanford.edu

from lark import Lark, Token
from math import floor, factorial
import numpy as np
from numpy.linalg import svd
from itertools import product, repeat, groupby
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher as GM
from networkx.algorithms.isomorphism import DiGraphMatcher as DGM
import re
from functools import *

# this is the description of the 
# Newick "grammar", to be consumed by lark
newick = """
start: subtree
?subtree: leaf | internal
?leaf: /\w+/
?internal: "(" branch_set ")" /\w+/
?branch_set: subtree | subtree "," subtree
"""

# an auxilliary function that takes lark-parser's output tree 
# and converts it into a binary tree object defined above
def tree_transform(tree):
    """An auxilliary function that takes lark-parser's output tree
       and converts it into a simplified binary tree object that 
       we use throughout the project.

    Args:
        tree (lark.tree.Tree): output of lark parser.

    Returns:
        Node: transformed binary tree.
    """
    if isinstance(tree, Token):
        return Node(tree.value)
    elif tree.data == "start":
        return tree_transform(tree.children[0])
    elif tree.data == "internal":
        name = tree.children[1].value
        node = Node(name)
        node.left = tree_transform(tree.children[0].children[0])
        node.right = tree_transform(tree.children[0].children[1])
        return node

def parse_tree(Ts):
    """Given a newick string with __named__ internal nodes, 
       output a corresponding binary tree object.

    Args:
        Ts (str): Newick tree with named internal nodes.

    Returns:
        Node: binary tree that was encoded by the Newick string.
    """
    parser = Lark(newick, parser='lalr')
    lark_parse = parser.parse
    T = tree_transform(lark_parse(Ts))
    return T

def name_internal_newick(newick_tree, label= "d"):
    """Names internal 'nodes' in a Newick string.

    Args:
        newick_tree (str): Newick string with __unnamed__ internal nodes.
        label (str, optional): prefix for internal node names. Defaults to "d".

    Returns:
        str: Newick string with named internal nodes.
    """
    i = 1
    new_tree = []
    for ch in newick_tree:
        new_tree.append(ch)
        if ch == ")":
            new_tree.append(label+str(i))
            i += 1
    return ''.join(new_tree)

def process_tree(Ts):
    """Function to parse an __unnamed__ Newick string into a binary tree object.

    Args:
        Ts (str): Newick string.

    Returns:
        Node: binary tree that was encoded by the Newick string.
    """
    return parse_tree(name_internal_newick(Ts))

def iter_flatten(iterable):
    """Recursively flatten lists of list of...

    Args:
        iterable (list or tuple): possibly nested iterable.

    Yields:
        list or tuple: flattened iterable.
    """
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in iter_flatten(e):
                yield f
        else:
            yield e

# ----------------------------------------------------------------------------#
# AC enumeration algorithms and functions                                     #
# ----------------------------------------------------------------------------#

@lru_cache(maxsize=128)
def get_root_acs(T):
    """Enumerate all ancestral configurations at the root 
       for matching gene and species tree.

    Args:
        T (Node): a binary tree as outputted by `process_tree` above.

    Returns:
        list: list of root configurations, each a list of lineages of the gene tree.
    """
    if len(T) == 1:
        return []
    else:
        l = [(T.left.value,T.right.value)] \
          +list(product([T.left.value], get_root_acs(T.right))) \
          + list(product(get_root_acs(T.left), [T.right.value])) \
          + list(product(get_root_acs(T.left), get_root_acs(T.right)))
        return [tuple([i for i in iter_flatten(ac)]) for ac in l]

@lru_cache(maxsize=128)
def count_matching_acs(T):
    """Count the __total__ (over all nodes) number of ancestral configurations for a 
       pair of gene and species trees.

    Args:
        T (Node): a binary tree as outputted by `process_tree` above.

    Returns:
        int: total number of ancestral configurations for a tree.
    """
    if len(T) == 1:
        return 0
    else:
        return len(get_root_acs(T)) + count_matching_acs(T.left) + count_matching_acs(T.right)

def get_kids(node_name, T):
    """Find the labels of two direct descendent nodes given the name of an ancestor.

    Args:
        node_name (str): ancestor node label.
        T (Node): binary tree.

    Returns:
        list: names of two direct descendants.
    """
    v = T.values
    loc = v.index(node_name)
    return v[2*loc+1], v[2*loc+2]

def is_leaf(node_name, T):
    """Check if the node is a leaf, given the node label.

    Args:
        node_name (str): ancestor node label.
        T (Node): binary tree.

    Returns:
        bool: is the node a leaf.
    """
    v = T.values
    loc = v.index(node_name)
    return (2*loc + 1 >= len(v)) or (v[2*loc+1] == None and v[2*loc+2] == None)


def is_leaf_array(i, values):
    """Version of `is_leaf` that directly uses the array representation.
       https://en.wikipedia.org/wiki/Binary_tree#Arrays

    Args:
        i (int): index of a node in the array.
        values (list): array representation of a binary tree.

    Returns:
        bool: is the node a leaf.
    """
    return (2*i + 1 >= len(values)) or (values[2*i+1] == None and values[2*i+2] == None)


def name_leaves(newick_tree):
    """Suppose you are lazy and write down an unlabeled tree,
       like ((*,*),(*,(*,*)))... Then this function names the nodes.

    Args:
        newick_tree (str): unlabeled topology encoded as a Newick string.

    Returns:
        str: labeled Newick string.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyzs'
    i = 0
    new_tree = []
    for ch in newick_tree:
        if ch != " ":
            if ch == "*":
                new_tree.append(alphabet[i])
                i += 1
            else:
                new_tree.append(ch)
    return ''.join(new_tree)

def str_format(s):
    """
    Helper to format the strings for display.
    """
    return "".join([c for c in s if c != "'"])

def ac_lattice(T, di = True):
    """Given a tree, get a lattice diagram of root configurations
       as a digraph or graph.

    Args:
        T (Node): binary tree
        di (bool, optional): Whether to return a directed or undirected graph object. Defaults to True.

    Returns:
        networkx.Graph or networkx.DiGraph: lattice diagram (digraph) of root configurations.
    """
    acs = sorted(get_root_acs(T),key = len)
    if di:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for i in range(len(acs)):
        ac = acs[i]
        for node in [l for l in acs[i] if not is_leaf(l, T)]:
            kids = get_kids(node,T)
            new_ac = list(ac)
            new_ac[ac.index(node)] = kids
            new_ac = tuple([a for a in iter_flatten(new_ac)])
            for larger_ac in acs[i+1:]:
                if larger_ac == new_ac:
                    G.add_edge(str_format(str(new_ac)),str_format(str(ac)))
            for larger_ac in acs[:i]:
                if larger_ac == new_ac:
                    G.add_edge(str_format(str(new_ac)),str_format(str(ac)))

    return G

def ac_lattice_rev(T, di = True):
    """Given a tree, get a lattice diagram of root configurations
       as a digraph or graph, with edge directions reversed. Necessary for visualisation.

    Args:
        T (Node): binary tree
        di (bool, optional): Whether to return a directed or undirected graph object. Defaults to True.

    Returns:
        networkx.Graph or networkx.DiGraph: lattice diagram (digraph) of root configurations.
    """
    acs = sorted(get_root_acs(T),key = len)
    if di:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for i in range(len(acs)):
        ac = acs[i]
        for node in [l for l in acs[i] if not is_leaf(l, T)]:
            kids = get_kids(node,T)
            new_ac = list(ac)
            new_ac[ac.index(node)] = kids
            new_ac = tuple([a for a in iter_flatten(new_ac)])
            for larger_ac in acs[i+1:]:
                if larger_ac == new_ac:
                    G.add_edge(str_format(str(ac)),str_format(str(new_ac)))
            for larger_ac in acs[:i]:
                if larger_ac == new_ac:
                    G.add_edge(str_format(str(ac)),str_format(str(new_ac)))

    return G

def draw(G):
    """Draw the lattice diagram using graphviz.

    Args:
        G (networkx.DiGraph): lattice diagram for root configurations

    Returns:
        The plot.
    """
    return nx.draw(G,pos = nx.nx_pydot.graphviz_layout(G, prog='dot'),with_labels=True)

def draw_custom_lab(T, ax):
    """Custom drawing funcion for the paper, with labels.

    Args:
        T (Node): tree.
        ax (matplotlib.Axes): axes to plot.

    Returns:
        The plot.
    """
    return nx.draw(ac_lattice(T),pos = nx.nx_pydot.graphviz_layout(ac_lattice_rev(T), prog='dot'),with_labels=True, node_color = '#d3d3d3', font_size = 17, arrowsize = 15, ax = ax)

def draw_custom_unlab(T, ax, labels=False):
    """Custom drawing funcion for the paper, in purple without labels.

    Args:
        T (Node): tree.
        ax (matplotlib.Axes): axes to plot.

    Returns:
        The plot.
    """
    return nx.draw(ac_lattice(T),pos = nx.nx_pydot.graphviz_layout(ac_lattice_rev(T), prog='dot'),with_labels=labels,\
                   ax = ax, node_color = '#9100ff', font_size = 17, arrowsize = 15, node_size = 250, width = 3)

# NOW WORK WITH NON-MATCHING TREES

def get_parents(node_location):
    """Math of array representations: given the location of the node,
       get locations of all of its ancestor nodes.

    Args:
        node_location (int): location of a node in the array representation.

    Returns:
        list(int): locations of all of the parent nodes.
    """
    if node_location <= 0:
        return []
    else:
        return [floor((node_location-1)/2)]+get_parents(floor((node_location-1)/2))
        

def get_mrca(descendent_name_list, T):
    """Given a list of nodes of a tree T,
       give the label of their most recent common ancestor (MRCA).

    Args:
        descendent_name_list (list): list of node labels.
        T (Node): binary trees.

    Returns:
        str: label of the MRCA of the list.
    """
    tree_values = T.values
    ancs = [set(get_parents(tree_values.index(d))) for d in descendent_name_list]
    return tree_values[max(set.intersection(*ancs))]

def get_descendant_leaves(node_name, T):
    """Get labels of all the descendent leaves of a given node.

    Args:
        node_name (str): node label.
        T (Node): binary tree.

    Returns:
        list: leaf labels of all descendants of a node.
    """
    values = T.values
    j = values.index(node_name)
    def get_desc(i):
        if is_leaf_array(i, values):
            return []
        elif is_leaf_array(2*i + 1, values) and is_leaf_array(2*i + 2, values):
            return [values[2*i+1]] + [values[2*i+2]]
        elif (not is_leaf_array(2*i + 1, values)) and is_leaf_array(2*i + 2, values):
            return [values[2*i+2]] + get_desc(2*i+1)
        elif (not is_leaf_array(2*i + 2, values)) and is_leaf_array(2*i + 1, values):
            return [values[2*i+1]] + get_desc(2*i+2)
        else:
            return get_desc(2*i+1) + get_desc(2*i+2)
    return sorted(get_desc(j))

def get_subtree(ancestor_node, T):
    """Get the subtree induced by a given node as a root.

    Args:
        node_name (str): node label.
        T (Node): binary tree.

    Returns:
        Node: induced tree.
    """
    values = T.values
    anc_ind = values.index(ancestor_node)
    def build_tree(i):
        if is_leaf_array(i,values):
            return Node(values[i])
        elif is_leaf_array(2*i + 1, values) and is_leaf_array(2*i + 2, values):
            return Node(values[i],left=Node(values[2*i+1]),right=Node(values[2*i+2]))
        elif (not is_leaf_array(2*i + 1, values)) and is_leaf_array(2*i + 2, values):
            return Node(values[i],left=build_tree(2*i+1),right=Node(values[2*i+2]))
        elif (not is_leaf_array(2*i + 2, values)) and is_leaf_array(2*i + 1, values):
            return Node(values[i],left=Node(values[2*i+1]),right=build_tree(2*i+2))
        else:
            return Node(values[i],left=build_tree(2*i+1),right=build_tree(2*i+2))
    return build_tree(anc_ind)

@lru_cache(maxsize=128)
def get_node_acs(s_node, G, S):
    """Get ancestral cnfigurations at the node of a species tree

    Args:
        s_node (str): node of a species tree.
        G (Node): gene tree.
        S (Node): species tree.

    Returns:
        list: list of ancestral configurations at a node. 
    """
    s_node_desc_leaves = get_descendant_leaves(s_node.value, S)
    G_mrca = get_mrca(s_node_desc_leaves, G)
    induced_G_subtree = get_subtree(G_mrca, G)
    G_acs = get_root_acs(induced_G_subtree)

    G_acs_without_extra_lineages = [] 
    for ac in G_acs:
        new_ac = []
        for lineage in ac:
            if not (is_leaf(lineage, G) and lineage not in s_node_desc_leaves):
                new_ac.append(lineage)
        G_acs_without_extra_lineages.append(new_ac)

    valid_acs = []
    for ac in G_acs_without_extra_lineages:
        valid = True
        for lineage in ac:
            if not is_leaf(lineage, G):
                lineage_leaves = get_descendant_leaves(lineage, G)
                S_mrca = get_mrca(lineage_leaves, S)
                if S_mrca not in s_node.values[1:]:
                    valid = False 
        if valid:
            valid_acs.append(ac)

    return valid_acs

@lru_cache(maxsize=128)
def get_acs(G,S):
    """Get ancestral configurations for every node of the species tree.

    Args:
        G (Node): gene tree.
        S (Node): species tree.
    
    Returns (dict): dictionary {node:AC's for the node}.
    """
    def get_acs_list_rec(s):
        if is_leaf(s.value, S):
            return []
        elif is_leaf(s.left.value, S) and is_leaf(s.right.value, S):
            return [(s.value, get_node_acs(s, G, S))]
        elif (not is_leaf(s.left.value, S)) and is_leaf(s.right.value, S):
            return [(s.value,get_node_acs(s, G, S))]  + get_acs_list_rec(s.left)
        elif is_leaf(s.left.value, S) and (not is_leaf(s.right.value, S)):
            return [(s.value,get_node_acs(s, G, S))]  + get_acs_list_rec(s.right)
        elif (not is_leaf(s.left.value, S)) and (not is_leaf(s.right.value, S)):
            return [(s.value, get_node_acs(s, G, S))]  + get_acs_list_rec(s.left) + get_acs_list_rec(s.right)
    acs_list = get_acs_list_rec(S)
    return {node:acs for node, acs in acs_list}

@lru_cache(maxsize=128)
def count_acs(G,S):
    """Conut total ancestral configurations for a pair of trees.

    Args:
        G (Node): gene tree.
        S (Node): species tree.

    Returns:
        int: total number of ancesral configurations
    """
    return sum([len(x) for x in get_acs(G,S).values()])

def count_tree(G, S):
    """Get a binary tree object, having the same topology as S, where at each node
       we record the number of ancestral configurations.

    Args:
        G (Node): gene tree.
        S (Node): species tree.

    Returns:
        Node: binary tree with counts.
    """
    def get_counts_tree(s):
        if is_leaf(s.value, S):
            return Node(0)
        elif is_leaf(s.left.value, S) and is_leaf(s.right.value, S):
            return Node(len(get_node_acs(s, G, S)),left=Node(s.left.value),right=Node(s.right.value))
        elif (not is_leaf(s.left.value, S)) and is_leaf(s.right.value, S):
            return Node(len(get_node_acs(s, G, S)),left=get_counts_tree(s.left),right=Node(s.right.value))
        elif is_leaf(s.left.value, S) and (not is_leaf(s.right.value, S)):
            return Node(len(get_node_acs(s, G, S)),right=get_counts_tree(s.right),left=Node(s.left.value))
        elif (not is_leaf(s.left.value, S)) and (not is_leaf(s.right.value, S)):
            return Node(len(get_node_acs(s, G, S)),left=get_counts_tree(s.left),right=get_counts_tree(s.right))
    return get_counts_tree(S)


class Node(object):
    # CODE BORROWED WITH MODIFICATION FROM GITHUB 
    # https://github.com/joowani/binarytree
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self):
        return 'Node({})'.format(self.value)

    def __str__(self):
        lines = _build_tree_string(self, 0, False, '-')[0]
        return '\n' + '\n'.join((line.rstrip() for line in lines))

    def __len__(self):
        return len([n for n  in self.values if n != None])

    @property
    def values(self):
        current_nodes = [self]
        has_more_nodes = True
        values = []

        while has_more_nodes:
            has_more_nodes = False
            next_nodes = []
            for node in current_nodes:
                if node is None:
                    values.append(None)
                    next_nodes.extend((None, None))
                    continue

                if node.left is not None or node.right is not None:
                    has_more_nodes = True

                values.append(node.value)
                next_nodes.extend((node.left, node.right))

            current_nodes = next_nodes

        # Get rid of trailing None's
        while values and values[-1] is None:
            values.pop()

        return values

    @property
    def leaves(self):
        current_nodes = [self]
        leaves = []

        while len(current_nodes) > 0:
            next_nodes = []
            for node in current_nodes:
                if node.left is None and node.right is None:
                    leaves.append(node)
                    continue
                if node.left is not None:
                    next_nodes.append(node.left)
                if node.right is not None:
                    next_nodes.append(node.right)
            current_nodes = next_nodes
        return leaves

def _build_tree_string(root, curr_index, index=False, delimiter='-'):
    # CODE BORROWED WITH MODIFICATION FROM GITHUB 
    # https://github.com/joowani/binarytree
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if index:
        node_repr = '{}{}{}'.format(curr_index, delimiter, root.value)
    else:
        node_repr = str(root.value)

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = \
        _build_tree_string(root.left, 2 * curr_index + 1, index, delimiter)
    r_box, r_box_width, r_root_start, r_root_end = \
        _build_tree_string(root.right, 2 * curr_index + 2, index, delimiter)

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(' ' * (l_root + 1))
        line1.append('_' * (l_box_width - l_root))
        line2.append(' ' * l_root + '/')
        line2.append(' ' * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(' ' * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append('_' * r_root)
        line1.append(' ' * (r_box_width - r_root + 1))
        line2.append(' ' * r_root + '\\')
        line2.append(' ' * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = ' ' * gap_size
    new_box = [''.join(line1), ''.join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
        r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end



# --------------------------------------------------------------------------- #
# a chunk dedicated to generating all labeled tree topologies *NONESSENTIAL*  #   
# NOTE: slow, look in other folders for the ones using generators             #
# ----------------------------------------------------------------------------#

class GNode(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return '(%s,%s)' % (self.left, self.right)

def add_leaf(tree, label):
    yield GNode(label, tree)
    if isinstance(tree, GNode):
        for left in add_leaf(tree.left, label):
            yield GNode(left, tree.right)
        for right in add_leaf(tree.right, label):
            yield GNode(tree.left, right)

def enum_unordered(labels):
    if len(labels) == 1:
        yield labels[0]
    else:
        for tree in enum_unordered(labels[1:]):
            for new_tree in add_leaf(tree, labels[0]):
                yield new_tree

def proper_order(tree, labelset):
    alph = ''.join(re.findall('[a-z]+', tree))
    return True if alph == labelset else False        

@lru_cache(maxsize=128)
def get_topologies(n, unordered = True):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    labelset = alphabet[:n]
    if unordered:
        return list(map(str,enum_unordered(labelset)))
    else:
        return [t for t in map(str,enum_unordered(labelset)) if proper_order(t, labelset)]

def get_unlabeled_topologies(n):
    if n == 1:
        yield "*"
    elif n % 2 == 0:
        for k in range(1, (n-2)//2 + 1):
            gen_left = get_unlabeled_topologies(n-k)
            gen_right = get_unlabeled_topologies(k)
            for left, right in product(gen_left, gen_right):
                yield (left,right)
        half_n_trees = list(get_unlabeled_topologies(n//2))
        l = len(half_n_trees)
        for i in range(l):
            for j in range(i,l):
                yield (half_n_trees[i], half_n_trees[j])
        
    elif n % 2 == 1:
        for k in range(1, (n-1)//2 + 1):
            gen_left = get_unlabeled_topologies(n-k)
            gen_right = get_unlabeled_topologies(k)
            for left, right in product(gen_left, gen_right):
                #yield Node(left=left, right=right)
                yield (left,right)
# ----------------------------------------------------------------------------#