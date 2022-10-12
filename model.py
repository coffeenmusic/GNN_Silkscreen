import torch
from torch.nn import Linear, ReLU, Dropout, Sequential, LeakyReLU
from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm
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
rotation_to_class = pp['rotation_to_class']

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=64, gnn_channels=512, type_dim=8, emb_dim=16, act=ReLU, do=0.4, gnn=SAGEConv, num_layers=1):
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
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('cmp','cmp-slk','slk'): SAGEConv((-1, -1), gnn_channels, normalize=True, bias=False),
                ('cmp','cmp-cmp','cmp'): SAGEConv((-1, -1), gnn_channels, normalize=True, bias=False),
                ('cmp','cmp-trk','trk'): SAGEConv((-1, -1), gnn_channels, normalize=True, bias=False),
            }, aggr='sum')
            self.convs.append(conv)
        
        self.mlp1 = Sequential(
            Linear(gnn_channels, hidden_channels),
            ReLU(), BatchNorm(hidden_channels), Dropout(do),
            Linear(hidden_channels, hidden_channels), 
            ReLU(), Dropout(do),
            Linear(hidden_channels, hidden_channels), 
            ReLU(), Dropout(do),
            Linear(hidden_channels, hidden_channels), 
            ReLU(), Dropout(do))
        
        self.mlp2 = Sequential(
            Linear(gnn_channels, hidden_channels),
            ReLU(), BatchNorm(hidden_channels), Dropout(do),
            Linear(hidden_channels, hidden_channels), 
            ReLU(), Dropout(do),
            Linear(hidden_channels, hidden_channels), 
            ReLU(), Dropout(do),
            Linear(hidden_channels, hidden_channels), 
            ReLU(), Dropout(do))
        
        self.mlp3 = Sequential(
            Linear(gnn_channels, hidden_channels),
            ReLU(), BatchNorm(hidden_channels), Dropout(do),
            Linear(hidden_channels, hidden_channels), 
            ReLU(), Dropout(do),
            Linear(hidden_channels, hidden_channels), 
            ReLU(), Dropout(do),
            Linear(hidden_channels, hidden_channels), 
            ReLU(), Dropout(do))
        
        self.x_out = Linear(hidden_channels, 1)
        self.y_out = Linear(hidden_channels, 1)
        
        self.rot_out = Linear(hidden_channels, len(rotation_to_class))

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

        x_dict['cmp'] = torch.cat([tool_emb, des_emb, lay_emb, xc[:, len(categorical_cmp_features):]], dim=-1)
        
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
            
        x = x_dict['slk']
        
        h1 = self.mlp1(x)
        rot_out = self.rot_out(h1)
        
        h2 = self.mlp2(x)
        x_out = self.x_out(h2)
        
        h3 = self.mlp3(x)
        y_out = self.y_out(h3)
        
        return [h1, h2, h3], rot_out, x_out, y_out
    
gnn_type = SAGEConv

gnn_channels = 2056
hidden_channels = 2056

actFunc = LeakyReLU

dropout = 0.41

model = GCN( hidden_channels=hidden_channels,
             gnn_channels=gnn_channels,
             gnn=gnn_type, 
             num_layers=1,
             act=actFunc, 
             do=dropout)
