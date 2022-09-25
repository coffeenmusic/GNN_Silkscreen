from itertools import permutations
import pandas as pd
import torch
import random

# def get_cmp_edge_idx(df, remove_sym=False):
    # if remove_sym:
        # # Remove symmetric edges. Example (0, 1) & (1, 0)
        # edge_sets = set()
        # for l in permutations(set(df.idx), 2):
            # lset = frozenset(l)
            # if lset not in edge_sets:
                # edge_sets.add(lset)

        # edge_index = torch.tensor([tuple(s) for s in list(edge_sets)], dtype=torch.long).T
    # else:
        # edge_tuple_list = [l for l in permutations(set(df.idx), 2)] # Create edges from every cmp to every other cmp
        # edge_index = torch.tensor(edge_tuple_list, dtype=torch.long).T
    
    # return edge_index
    
def get_cmp_edge_idx(c, n=5):
    """
    n: get n closest components
    """
    srcs, dsts = [], []
    for i, r in c.iterrows():
        x, y = r['x'], r['y']
        srcs += [r['idx']]*n
        dsts += c.iloc[((c.x - x)**2 + (c.y - y)**2).argsort()].idx[:n].tolist()

    edge_df = pd.DataFrame({'src':srcs, 'dst':dsts})
    edge_idx = torch.tensor(edge_df[['src', 'dst']].values, dtype=torch.long).T
    
    return edge_idx

def get_cmp_slk_edge_idx(c, s, edge_key):
    # s = Silkscreen dataframe
    # c = Component dataframe
    # edge_key = pyGeometric hetero edge_dict key

    cols = ['Designator','idx']
    edge_df = pd.merge(s.loc[:,cols], c.loc[:,cols], how='inner', on=['Designator'], suffixes=('_s', '_c'))
    
    # Ensure edge index is returned in the correct order (cmp, slk) or (slk, cmp)
    if edge_key[0] == 'slk':
        edge_idx = torch.tensor(edge_df[['idx_s', 'idx_c']].values, dtype=torch.long).T
    elif edge_key[0] == 'cmp':
        edge_idx = torch.tensor(edge_df[['idx_c', 'idx_s']].values, dtype=torch.long).T
    else:
        print('Warning! Incorrect edge key.')
        
    return edge_idx
        
def get_split_mask(df, val_pct=0.02, test_pct=0.02):
    slk_idx = df['idx'].astype(int).values
    random.shuffle(slk_idx)

    val_len = int(len(slk_idx)*val_pct)
    test_len = int(len(slk_idx)*test_pct)

    val_idx = slk_idx[:val_len]
    test_idx = slk_idx[val_len:val_len+test_len]
    train_idx = slk_idx[val_len+test_len:]

    val_mask = torch.tensor(df.idx.isin(val_idx).values, dtype=torch.bool)
    test_mask = torch.tensor(df.idx.isin(test_idx).values, dtype=torch.bool)
    train_mask = torch.tensor(df.idx.isin(train_idx).values, dtype=torch.bool)
    
    return train_mask, val_mask, test_mask

def get_trk_edge_idx(t):
    # Get any tracks that are overlapping w/ same x/y or x2/y2
    # t = Track dataframe
    t = t.copy()
    t2 = t.copy()
    
    t2.x1 = t.x2
    t2.y1 = t.y2
    t.drop(columns=['x','y','x2','y2'], inplace=True)
    t2.drop(columns=['x','y','x2','y2'], inplace=True)
    t = pd.concat((t, t2), axis=0)
    t2 = t.copy()

    cols = ['idx','x1','y1']
    edge_df = pd.merge(t.loc[:,cols], t2.loc[:,cols], how='inner', on=['x1','y1'], suffixes=('_t', '_t2'))
    
    # Remove any matches that have the same index because they are the same object
    edge_df = edge_df.loc[edge_df.idx_t != edge_df.idx_t2] 
    
    edge_idx = torch.tensor(edge_df[['idx_t','idx_t2']].values, dtype=torch.long).T
    
    return edge_idx

def get_cmp_trk_edge_idx(c, t, edge_key):
    # t = Track dataframe
    # c = Component dataframe
    # edge_key = pyGeometric hetero edge_dict key

    # Connect InComponent Tracks to their cmp designator
    cols = ['Designator','idx']
    edge_df = pd.merge(t.loc[t.X_InCmp==1, cols], c.loc[:,cols], how='inner', on=['Designator'], suffixes=('_t', '_c'))
    
    edge_df = edge_df[['idx_t','idx_c']]
    
    # Connect Out of Component Tracks to the cmp closest in the x/y coordinates
    cidxs, tidxs = [], []
    for i, r in t.loc[t.X_InCmp == 0].iterrows():
        tidxs += [r.idx]
        cidxs += [c.iloc[(abs(c.x - r.x1) + abs(c.y - r.y1)).argsort()].idx.values[0]]
    
    # Append Out of Component Tracks to edge_df
    edge_df = pd.concat((edge_df, pd.DataFrame({'idx_t':tidxs, 'idx_c':cidxs})), axis=0)
    
    # Ensure edge index is returned in the correct order (cmp, slk) or (slk, cmp)
    if edge_key[0] in ['trk','arc']:
        edge_idx = torch.tensor(edge_df[['idx_t', 'idx_c']].values, dtype=torch.long).T
    elif edge_key[0] == 'cmp':
        edge_idx = torch.tensor(edge_df[['idx_c', 'idx_t']].values, dtype=torch.long).T
    else:
        print('Warning! Incorrect edge key.')
        
    return edge_idx