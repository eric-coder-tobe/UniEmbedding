import torch
import torch.nn as nn
import torch.nn.functional as F

class UniEmbedding(nn.Module):
    def __init__(self, config, dataset):
        self.config = config
        # dataset class include attributes text_emb_dim
        self.dataset = dataset
        self.n_items = dataset.n_items
        self.text_emb_dim = dataset.text_emb_dim
        self.text_embedding = dataset.text_emb
        self.hidden_size = config['hidden_size']
        self.encoder_layer_num = config['encoder_layer_num']
        self.d_model = config['d_model']
        self.trm_head_num = config['transformer_head_num']
        self.dim_feedforward = config['dim_feedforward']
        self.expert_num = config['expert_num']
        self.user_matrix_head = config['user_matrix_head']
        self.temperature = config['temperature']
        
        self.user_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.head_num, dim_feedforward=self.dim_feedforward, batch_first=True)
        # to encode interactive sequence as user_view
        # when FP, pass the src_key_padding_mask to mark the position of pads(pad position is set True)
        self.user_encoder = nn.TransformerEncoder(encoder_layer=self.user_encoder_layer, num_layers=self.encoder_layer_num)
        # use item_embedding to encode user history sequence
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        # text_embedding of different items
        self.text_embedding = nn.Embedding.from_pretrained(self.text_embedding)
        # transform intermediate output from dim_feedforward to hidden_size*hidden_size, when PF reshape intermediate to user_projection_matrix
        self.user_transform_layer = nn.Linear(self.dim_feedforward, self.hidden_size * self.hidden_size * self.user_matrix_head)
        # transform plm text_embedding to final embedding to contrast learning
        self.item_adapter = MOE_adaptor(expert_num=self.expert_num, input_dim=self.text_emb_dim, out_dim=self.hidden_size)
        
        for name, param in self.named_parameters:
            if 'text_embedding' in name:
                param.requires_grad = False
                
        
    def forward(self, batch_data):
        # 注意dataset喂进来的数据要有对应的属性
        # 此时的数据排除长度小于2的，每个序列都是有效序列
        history_seq = batch_data['history_seq']     # B, S
        seq_len = batch_data['seq_len']     # B     # 似乎不需要seq_len，处理好的数据pad_id为0
        domain_id = batch_data['domain_id']     # B, S
        # 实际上应该针对不同domain的item使用不同的MOE，先实现一版，后面再考虑训练效率
        padding_mask = history_seq == 0     # B, S
        item_emb = self.item_embedding(history_seq)     # B, S, hidden_d
        trm_out = self.user_encoder(item_emb, src_key_padding_mask=padding_mask)        # B, S, hidden_d
        trm_out = trm_out * padding_mask.unsqueeze(2)
        # using the average of transformer out as user_represention
        user_emb = trm_out.sum(dim=1)       # B, hidden_d
        user_emb /= seq_len
        user_emb = F.normalize(user_emb, p=2, dim=1)
        # use multi-head Linear layer to transform user_emb to user_matrix
        user_projection = self.user_transform_layer(user_emb).reshape(-1, self.hidden_size * self.hidden_size, self.user_matrix_head)
        user_projection = user_projection.mean(dim=2).reshape(-1, self.hidden_size, self.hidden_size)       # B, d_hidden, d_hidden
        text_emb = self.text_embedding(history_seq)     # B, S, d_mm
        text_mediate = self.item_adapter(text_emb)      # B, S, d_hidden
        tensor_shape = text_mediate.shape
        user_projection = user_projection.unsqueeze(1).expand(tensor_shape[0], tensor_shape[1], tensor_shape[2], tensor_shape[2]).reshape(-1, tensor_shape[2], tensor_shape[2])
        text_mediate_reshape = text_mediate.unsqueeze(3).reshape(-1, tensor_shape[2], 1)
        user_viewed_emb = torch.bmm(user_projection, text_mediate_reshape).squeeze(2).reshape(tensor_shape)
        
        # 中间层其实还需要MMD进行分布对齐
        return text_mediate, user_viewed_emb
        
        
    def pos_neg_sample(self, batch_data):
        # 用于生成正采样负采样的样本对
        # 对于每一个样本，对应都要有同一序列的正样本和不同序列的负样本，难点在于不同序列的负样本可能会和正样本有重叠
        # 最好写成高并行性版本
        pass
        
        
        
    def pretrain(self, batch_data):
        text_mediate, user_viewed_emb = self.forward(batch_data)
        
        
        
        
    
        
        
        
        
        
class MOE_adaptor(nn.Module):
    def __init__(self, expert_num, input_dim, out_dim, gate_hidden_units=[128, 64], expert_hidden_units=[512, 256, 128], dropout=0.2):
        super().__init__()
        self.expert_num = expert_num
        self.input_dim = input_dim
        self.out_dim = out_dim
        gate_hidden_units.append(expert_num)
        gate_hidden_units = [self.input_dim] + gate_hidden_units
        self.gate_hidden_units = gate_hidden_units
        self.expert_hidden_units = [self.input_dim] + expert_hidden_units + [self.out_dim]
        self.gate = MLP(self.gate_hidden_units, dropout=dropout)
        self.gate_softmax = nn.Softmax(dim=1)
        self.experts = nn.ModuleList([MLP(self.expert_hidden_units, dropout=dropout) for _ in range(self.expert_num)])
        
    def forward(self, x):
        # x.shape: [batchsize, input_dim]
        # out.shape: [batchsize, out_dim]
        gate_out = self.gate(x)
        gate_out = self.gate_softmax(gate_out)
        # gate_out: [batchsize, expert_num, 1]
        gate_out = gate_out.unsqueeze(2)
        # expert_out: [batchsize, out_dim, expert_num]
        experts_out = torch.cat([self.experts[i](x).unsqueeze(1) for i in range(self.expert_num)], dim=1).permute(0,2,1)
        out = torch.bmm(experts_out, gate_out).squeeze(2)
        return out
        
        
        
        
class MLP(nn.Module):
    '''
    MLP modules, with batch norm layer and dropout layer, take relu as its activation function.
    The dim of its layers follows dims in hidden_units.
    '''
    def __init__(self, hidden_units, dropout):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
        layers = list()
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
            layers.append(nn.BatchNorm1d(hidden_units[i+1]))
            layers.append(self.dropout)
            layers.append(self.activation)
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        # x.shape: [batchsize, hidden_units[0]]
        # out.shape: [batchsize, hidden_units[-1]]
        out = self.mlp(x)
        return out
            
        
        
    