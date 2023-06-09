from pastml.tree import read_tree, name_tree, read_forest
from pastml.acr import _validate_input, acr
from pastml.annotation import preannotate_forest
from pastml import col_name2cat
from collections import defaultdict, Counter

import os, math
import numpy as np
import pandas as pd

def _validate_input(tree_nwk, data=None, data_sep='\t', id_index=0, file=False):

    df      = pd.read_csv(data, sep=data_sep, index_col=id_index, header=0, dtype=str)
    columns = df.columns[0]
    if file == True:
        roots = read_forest(tree_nwk, columns=[columns] if data is None else None)
    else:
        roots = [read_tree(tree_nwk + ';', [columns])]

    num_neg = 0
    for root in roots:
        for _ in root.traverse():
            if _.dist < 0:
                num_neg += 1
                _.dist = 0

    column2annotated = Counter()
    column2states    = defaultdict(set)
    
    if data:
        df.index = df.index.map(str)
        if columns:
            df = df[[columns]]
        df.columns  = [col_name2cat(column) for column in df.columns]
        df_columns = df.columns

        node_names     = set.union(*[{n.name for n in root.traverse() if n.name} for root in roots])
        df_index_names = set(df.index)
        common_ids     = node_names & df_index_names

        # strip quotes if needed
        if not common_ids:
            node_names = {_.strip("'").strip('"') for _ in node_names}
            common_ids = node_names & df_index_names
            if common_ids:
                for root in roots:
                    for n in root.traverse():
                        n.name = n.name.strip("'").strip('"')

        preannotate_forest(roots, df=df)
        for c in df.columns:
            column2states[c] |= {_ for _ in df[c].unique() if pd.notnull(_) and _ != ''}

    num_tips = 0

    column2annotated_states = defaultdict(set)
    for root in roots:
        for n in root.traverse():
            for c in df_columns:
                vs = getattr(n, c, set())
                column2states[c] |= vs
                column2annotated_states[c] |= vs
                if vs:
                    column2annotated[c] += 1
            if n.is_leaf():
                num_tips += 1

    column2states = {columns: np.array(sorted(df[columns].unique()))}

    for i, tree in enumerate(roots):
        name_tree(tree, suffix='' if len(roots) == 1 else '_{}'.format(i))

    return roots, df_columns, column2states


def marginal(tree, data=None, prediction_method='MPPA', model='F81',
            forced_joint=False, threads=0):

    if threads < 1:
        threads = max(os.cpu_count(), 1)

    roots, columns, column2states = \
        _validate_input(tree, data, data_sep=',')

    acr_results = acr(forest=roots, columns=columns, column2states=column2states, prediction_method=prediction_method, model=model, 
                    force_joint=forced_joint, threads=threads)
    
    leaf_names = read_tree(tree).get_leaf_names()
    marginal   = np.asarray( acr_results[0]['marginal_probabilities'].drop(leaf_names) )

    return marginal