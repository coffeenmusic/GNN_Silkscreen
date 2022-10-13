import torch
from torch.nn import Linear, ReLU, Dropout, Sequential, LeakyReLU, ModuleList, BatchNorm1d
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import HeteroConv
import pickle

pp = pickle.load(open('preprocessing.pkl', 'rb'))
categorical_dict = pp['categorical_dict']
cmp_features = pp['cmp_features']
slk_features = pp['slk_features']
trk_features = pp['trk_features']
arc_features = pp['arc_features']
slk_lbls = pp['slk_lbls']
categorical_cmp_features = pp['categorical_cmp_features']
categorical_slk_features = pp['categorical_slk_features']
categorical_trk_features = pp['categorical_trk_features']
categorical_arc_features = pp['categorical_arc_features']

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=64, gnn_channels=512, act=ReLU, do=0.4, gnn=SAGEConv, gnn_layers=1, h_layers=1, normalize=False, bias=False):
        """
        cd : categorical dictionary {feature_name: num_categories, ...}
        """
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        
        cd = categorical_dict
        
        self.do = do
        
        def emb_dim(class_cnt):
            return max(int((class_cnt**.5)*1.6)+1, 2)
        
        self.des_emb = torch.nn.Embedding(cd['X_DES'], emb_dim(cd['X_DES']))
        self.lay_emb = torch.nn.Embedding(cd['X_Layer'], emb_dim(cd['X_Layer']))
        self.tool_emb = torch.nn.Embedding(cd['X_Tool'], emb_dim(cd['X_Tool']))
        self.rot_emb = torch.nn.Embedding(cd['X_ROT'], emb_dim(cd['X_ROT']))
        
        self.convs = ModuleList()
        for _ in range(gnn_layers):
            conv = HeteroConv({
                ('cmp','cmp-slk','slk'): gnn((-1, -1), gnn_channels, normalize=normalize, bias=bias),
                ('cmp','cmp-cmp','cmp'): gnn((-1, -1), gnn_channels, normalize=normalize, bias=bias),
                ('cmp','cmp-trk','trk'): gnn((-1, -1), gnn_channels, normalize=normalize, bias=bias),
                ('trk','trk-trk','trk'): gnn((-1, -1), gnn_channels, normalize=normalize, bias=bias),
            }, aggr='sum')
            self.convs.append(conv)
        
        # Define network branches as separate module lists
        self.mlp1, self.mlp2, self.mlp3 = ModuleList(), ModuleList(), ModuleList()
        
        layer_dict = {'in': Linear(gnn_channels, hidden_channels), 'lin': Linear(hidden_channels, hidden_channels), 'act': act(), 'do': Dropout(do)}   
        if normalize:
            layer_dict['bn'] = BatchNorm1d(hidden_channels)
            
        for mlp in [self.mlp1, self.mlp2, self.mlp3]:
            mlp.extend([layer_dict[k] for k in ['in','bn','act','do'] if k in layer_dict.keys()])
            for _ in range(h_layers):
                mlp.extend([layer_dict[k] for k in ['lin','bn','act','do'] if k in layer_dict.keys()])
        
        self.x_out = Linear(hidden_channels, 1)
        self.y_out = Linear(hidden_channels, 1)
        
        self.rot_out = Linear(hidden_channels, cd['X_ROT'])

    def forward(self, x_dict, edge_index_dict): 
        # Component Type
        xc = x_dict['cmp']
        
        # CMP Categorical Features: 'X_DES', 'X_Layer'
        tool = xc[:, cmp_features.index('X_Tool')].to(dtype=torch.long)
        tool_emb = self.tool_emb(tool)
        des = xc[:, cmp_features.index('X_DES')].to(dtype=torch.long)
        des_emb = self.des_emb(des)
        lay = xc[:, cmp_features.index('X_Layer')].to(dtype=torch.long)
        lay_emb = self.lay_emb(lay)       
        rot = xc[:, cmp_features.index('X_ROT')].to(dtype=torch.long)
        rot_emb = self.rot_emb(rot)         

        x_dict['cmp'] = torch.cat([tool_emb, des_emb, lay_emb, rot_emb, xc[:, len(categorical_cmp_features):]], dim=-1)
        
        # Silkscreen Type
        xs = x_dict['slk']
        
        # SLK Categorical Features: 'X_DES'
        tool = xs[:, slk_features.index('X_Tool')].to(dtype=torch.long)
        tool_emb = self.tool_emb(tool)
        des = xs[:, slk_features.index('X_DES')].to(dtype=torch.long)
        des_emb = self.des_emb(des)
        
        x_dict['slk'] = torch.cat([tool_emb, des_emb, xs[:, len(categorical_slk_features):]], dim=-1)
        
        # Track Type
        xt = x_dict['trk']
        
        # TRK Categorical Features: 'X_DES', 'X_Layer'
        tool = xt[:, trk_features.index('X_Tool')].to(dtype=torch.long)
        tool_emb = self.tool_emb(tool)
        des = xt[:, trk_features.index('X_DES')].to(dtype=torch.long)
        des_emb = self.des_emb(des)
        lay = xt[:, trk_features.index('X_Layer')].to(dtype=torch.long)
        lay_emb = self.lay_emb(lay)
        
        x_dict['trk'] = torch.cat([tool_emb, des_emb, lay_emb, xt[:, len(categorical_trk_features):]], dim=-1)
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: Dropout(self.do)(x.relu()) for key, x in x_dict.items()}
        
        def forward_modules(module_list, x):
            for h in module_list:
                x = h(x)
            return x
            
        h1 = forward_modules(self.mlp1, x_dict['slk']) 
        rot_out = self.rot_out(h1)
        
        h2 = forward_modules(self.mlp2, x_dict['slk'])
        x_out = self.x_out(h2)
        
        h3 = forward_modules(self.mlp3, x_dict['slk'])
        y_out = self.y_out(h3)
        
        return [h1, h2, h3], rot_out, x_out, y_out
