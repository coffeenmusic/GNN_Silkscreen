import sys

import torch
from torch_geometric.data import Data

import pandas as pd
import os


def export_predictions(loader, model, pr, device, class_to_rotation, export_filename='predictions.csv', export_subdir=''):
    """
    pr: is a copy of the slk-des type dataframe used to make predictions on
    """
    model.eval()
    
    export_path = os.path.join(export_subdir, export_filename)

    rotations, xs, ys = [],[],[]
    for d in loader:
        d.to(device)

        _, rot_out, dx_out, dy_out = model(d.x_dict, d.edge_index_dict)
        
        rotations += [class_to_rotation[int(i)] for i in rot_out.argmax(dim=1)]
        xs += dx_out.reshape((-1,)).tolist()
        ys += dy_out.reshape((-1,)).tolist()

    pr['pred_x'] = xs
    pr['pred_y'] = ys
    pr['pred_rot'] = rotations
    pr['pred_lay'] = pr.X_Layer # Not used in import script. Layer is assumed to remain the same.
    
    pr.loc[:,['Designator','pred_x','pred_y','pred_rot']].to_csv(export_path, header=False, index=False)