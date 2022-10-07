import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch import nn
import math

TYPE_REMOVAL = 'N2S'
#TYPE_REMOVAL = 'random'
#TYPE_REMOVAL = 'greedy'

TYPE_REINSERTION = 'N2S'
#TYPE_REINSERTION = 'random'
#TYPE_REINSERTION = 'greedy'


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q):
        
        h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim) #################   reshape
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)   
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        attn = F.softmax(compatibility, dim=-1)   
       
        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out

class MultiHeadAttentionNew(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttentionNew, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
         
        self.score_aggr = nn.Sequential(
                        nn.Linear(8, 8),
                        nn.ReLU(inplace=True),
                        nn.Linear(8, 4))


        self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()
        
    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h, out_source_attn):
        
        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()

        hflat = h.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (4, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(hflat, self.W_query).view(shp)
        K = torch.matmul(hflat, self.W_key).view(shp)   
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, bastch_size, n_query, graph_size)
        compatibility = torch.cat((torch.matmul(Q, K.transpose(2, 3)), out_source_attn), 0)
        
        attn_raw = compatibility.permute(1,2,3,0)
        attn = self.score_aggr(attn_raw).permute(3,0,1,2)
        heads = torch.matmul(F.softmax(attn, dim=-1), V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, graph_size, self.embed_dim)

        return out, out_source_attn

class MultiHeadPosCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadPosCompat, self).__init__()
    
        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, pos):
        
        batch_size, graph_size, input_dim = pos.size()
        posflat = pos.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(posflat, self.W_query).view(shp)  
        K = torch.matmul(posflat, self.W_key).view(shp)   

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        return torch.matmul(Q, K.transpose(2, 3))

class MultiHeadCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadCompat, self).__init__()
    
        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h = None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim) #################   reshape
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)  
        K = torch.matmul(hflat, self.W_key).view(shp)   

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility_s2n = torch.matmul(Q, K.transpose(2, 3))
        
        return  compatibility_s2n

class CompatNeighbour(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(CompatNeighbour, self).__init__()
    
        n_heads = 4
        
        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_Q = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_K = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        
        self.agg = MLP(12, 32, 32, 1, 0)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h, rec, visited_order_map, selection_sig):
        
        pre = rec.argsort()
        post = rec.gather(1, rec)
        batch_size, graph_size, input_dim = h.size()

        flat = h.contiguous().view(-1, input_dim) #################   reshape

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        hidden_Q = torch.matmul(flat, self.W_Q).view(shp)
        hidden_K = torch.matmul(flat, self.W_K).view(shp)
        
        Q_pre = hidden_Q.gather(2, pre.view(1, batch_size, graph_size, 1).expand_as(hidden_Q))
        K_post = hidden_K.gather(2, post.view(1, batch_size, graph_size, 1).expand_as(hidden_Q))
    
        compatibility = ((Q_pre * hidden_K).sum(-1) + (hidden_Q * K_post).sum(-1) - (Q_pre * K_post).sum(-1))[:,:,1:]
        
        compatibility_pairing = torch.cat((compatibility[:,:,:graph_size // 2], compatibility[:,:,graph_size // 2:]), 0)

        compatibility_pairing = self.agg(torch.cat((compatibility_pairing.permute(1,2,0), 
                                                    selection_sig.permute(0,2,1)),-1)).squeeze()
        
        return  compatibility_pairing

class Reinsertion(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(Reinsertion, self).__init__()
    
        n_heads = 4
        
        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        
        self.norm_factor = 1 / math.sqrt(2 * embed_dim)  # See Attention is all you need


        self.compater_insert1 = MultiHeadCompat(n_heads,
                                        embed_dim,
                                        embed_dim,
                                        embed_dim,
                                        key_dim)
        
        self.compater_insert2 = MultiHeadCompat(n_heads,
                                        embed_dim,
                                        embed_dim,
                                        embed_dim,
                                        key_dim)
        
        self.agg = MLP(16, 32, 32, 1, 0)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, h, pos_pickup, pos_delivery, rec, mask=None):
        
        batch_size, graph_size, input_dim = h.size()
        shp = (batch_size, graph_size, graph_size, self.n_heads)
        shp_p = (batch_size, -1, 1, self.n_heads)
        shp_d = (batch_size, 1, -1, self.n_heads)
        
        arange = torch.arange(batch_size, device = h.device)
        h_pickup = h[arange,pos_pickup].unsqueeze(1)
        h_delivery = h[arange,pos_delivery].unsqueeze(1)
        h_K_neibour = h.gather(1, rec.view(batch_size, graph_size, 1).expand_as(h))
        
        compatibility_pickup_pre = self.compater_insert1(h_pickup, h).permute(1,2,3,0).view(shp_p).expand(shp)
        compatibility_pickup_post = self.compater_insert2(h_pickup, h_K_neibour).permute(1,2,3,0).view(shp_p).expand(shp)
        compatibility_delivery_pre = self.compater_insert1(h_delivery, h).permute(1,2,3,0).view(shp_d).expand(shp)
        compatibility_delivery_post = self.compater_insert2(h_delivery, h_K_neibour).permute(1,2,3,0).view(shp_d).expand(shp)
        
        compatibility = self.agg(torch.cat((compatibility_pickup_pre, 
                                            compatibility_pickup_post, 
                                            compatibility_delivery_pre, 
                                            compatibility_delivery_post),-1)).squeeze()
        return compatibility

class MLP(torch.nn.Module):
    def __init__(self,
                input_dim = 128,
                feed_forward_dim = 64,
                embedding_dim = 64,
                output_dim = 1,
                p_dropout = 0.01
    ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, feed_forward_dim)
        self.fc2 = torch.nn.Linear(feed_forward_dim, embedding_dim)
        self.fc3 = torch.nn.Linear(embedding_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.ReLU = nn.ReLU(inplace = True)
        
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, in_):
        result = self.ReLU(self.fc1(in_))
        result = self.dropout(result)
        result = self.ReLU(self.fc2(result))
        result = self.fc3(result).squeeze(-1)
        return result

class ValueDecoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            input_dim,
    ):
        super(ValueDecoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.project_graph = nn.Linear(self.input_dim, self.embed_dim // 2)
        
        self.project_node = nn.Linear(self.input_dim, self.embed_dim // 2) 
        
        self.MLP = MLP(input_dim + 1, embed_dim)


    def forward(self, h_em, cost): 
                
        # get embed feature
#        max_pooling = h_em.max(1)[0]   # max Pooling
        mean_pooling = h_em.mean(1)     # mean Pooling
        graph_feature = self.project_graph(mean_pooling)[:, None, :]
        node_feature = self.project_node(h_em)
        
        #pass through value_head, get estimated value
        fusion = node_feature + graph_feature.expand_as(node_feature) # torch.Size([2, 50, 128])

        fusion_feature = torch.cat((fusion.mean(1),
                                    fusion.max(1)[0],
                                    cost.to(h_em.device),
                                    ), -1)
        
        value = self.MLP(fusion_feature)
      
        return value

class MultiHeadDecoder(nn.Module):
    def __init__(
            self,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            v_range = 6,
    ):
        super(MultiHeadDecoder, self).__init__()
        self.n_heads = n_heads = 1
        self.embed_dim = embed_dim
        self.input_dim = input_dim        
        self.range = v_range
        
        if TYPE_REMOVAL == 'N2S':
            self.compater_removal = CompatNeighbour(n_heads,
                                            embed_dim,
                                            embed_dim,
                                            embed_dim,
                                            key_dim)
        if TYPE_REINSERTION == 'N2S':
            self.compater_reinsertion = Reinsertion(n_heads,
                                            embed_dim,
                                            embed_dim,
                                            embed_dim,
                                            key_dim)
            
        self.project_graph = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
        
        
    def forward(self, problem, h_em, rec, x_in, top2, visited_order_map, pre_action, selection_sig, fixed_action = None, require_entropy = False):        
    
        bs, gs, dim = h_em.size()
        half_pos =  (gs - 1) // 2
        
        arange = torch.arange(bs)
    
        h = self.project_node(h_em) + self.project_graph(h_em.max(1)[0])[:, None, :].expand(bs, gs, dim)
        
        ############# action1 removal
        if TYPE_REMOVAL == 'N2S':
            action_removal_table = torch.tanh(self.compater_removal(h, rec, visited_order_map, selection_sig).squeeze()) * self.range
            if pre_action is not None and pre_action[0,0] > 0:
                action_removal_table[arange, pre_action[:,0]] = -1e20
            log_ll_removal = F.log_softmax(action_removal_table, dim = -1) if self.training and TYPE_REMOVAL == 'N2S' else None
            probs_removal = F.softmax(action_removal_table, dim = -1)
        elif TYPE_REMOVAL == 'random':
            probs_removal = torch.rand(bs, gs//2).to(h_em.device)
        else:
            # epi-greedy
            first_row = torch.arange(gs, device = rec.device).long().unsqueeze(0).expand(bs, gs)
            d_i =  x_in.gather(1, first_row.unsqueeze(-1).expand(bs, gs, 2))
            d_i_next = x_in.gather(1, rec.long().unsqueeze(-1).expand(bs, gs, 2))
            d_i_pre = x_in.gather(1, rec.argsort().long().unsqueeze(-1).expand(bs, gs, 2))
            cost_ = ((d_i_pre  - d_i).norm(p=2, dim=2) + (d_i  - d_i_next).norm(p=2, dim=2) - (d_i_pre  - d_i_next).norm(p=2, dim=2))[:,1:]
            probs_removal = (cost_[:,:gs//2] + cost_[:,gs//2:])
            probs_removal_random = torch.rand(bs, gs//2).to(h_em.device)
            
        if fixed_action is not None:
            action_removal = fixed_action[:,:1]
        else:
            if TYPE_REMOVAL == 'greedy':
                action_removal_random = probs_removal_random.multinomial(1)
                action_removal_greedy = probs_removal.max(-1)[1].unsqueeze(1)
                action_removal = torch.where(torch.rand(bs,1).to(h_em.device) < 0.1, action_removal_random, action_removal_greedy)
            else:     
                action_removal = probs_removal.multinomial(1)
        selected_log_ll_action1 = log_ll_removal.gather(1, action_removal) if self.training and TYPE_REMOVAL == 'N2S' else torch.tensor(0).to(h.device)
        
        ############# action2
        pos_pickup = (1 + action_removal).view(-1)
        pos_delivery = pos_pickup + half_pos
        mask_table = problem.get_swap_mask(action_removal + 1, visited_order_map, top2).expand(bs, gs, gs).cpu()
        if TYPE_REINSERTION == 'N2S':
            action_reinsertion_table = torch.tanh(self.compater_reinsertion(h, pos_pickup, pos_delivery, rec, mask_table)) * self.range
        elif TYPE_REINSERTION == 'random':
            action_reinsertion_table = torch.ones(bs, gs, gs).to(h_em.device)
        else:
            
            # epi-greedy
            pos_pickup = (1 + action_removal)
            pos_delivery = pos_pickup + half_pos
            rec_new = rec.clone()
            argsort = rec_new.argsort()
            pre_pairfirst = argsort.gather(1, pos_pickup)
            post_pairfirst = rec_new.gather(1, pos_pickup)
            rec_new.scatter_(1,pre_pairfirst, post_pairfirst)
            rec_new.scatter_(1,pos_pickup, pos_pickup)
            argsort = rec_new.argsort()
            pre_pairsecond = argsort.gather(1, pos_delivery)
            post_pairsecond = rec_new.gather(1, pos_delivery)
            rec_new.scatter_(1,pre_pairsecond,post_pairsecond) 
            # perform calc on new rec_new
            first_row = torch.arange(gs, device = rec.device).long().unsqueeze(0).expand(bs, gs)
            d_i =  x_in.gather(1, first_row.unsqueeze(-1).expand(bs, gs, 2))
            d_i_next = x_in.gather(1, rec_new.long().unsqueeze(-1).expand(bs, gs, 2))
            d_pick = x_in.gather(1, pos_pickup.unsqueeze(1).expand(bs, gs, 2))
            d_deli = x_in.gather(1, pos_delivery.unsqueeze(1).expand(bs, gs, 2))
            cost_insert_p = (d_pick  - d_i).norm(p=2, dim=2) + (d_pick  - d_i_next).norm(p=2, dim=2) - (d_i  - d_i_next).norm(p=2, dim=2)
            cost_insert_d = (d_deli  - d_i).norm(p=2, dim=2) + (d_deli  - d_i_next).norm(p=2, dim=2) - (d_i  - d_i_next).norm(p=2, dim=2)
            action_reinsertion_table = - (cost_insert_p.view(bs, gs, 1) + cost_insert_d.view(bs, 1, gs))
            action_reinsertion_table_random = torch.ones(bs, gs, gs).to(h_em.device)
            action_reinsertion_table_random[mask_table] = -1e20
            action_reinsertion_table_random = action_reinsertion_table_random.view(bs, -1)
            probs_reinsertion_random = F.softmax(action_reinsertion_table_random, dim = -1)
             
        action_reinsertion_table[mask_table] = -1e20
        
        del visited_order_map, mask_table
        #reshape action_reinsertion_table
        action_reinsertion_table = action_reinsertion_table.view(bs, -1)
        log_ll_reinsertion = F.log_softmax(action_reinsertion_table, dim = -1) if self.training and TYPE_REINSERTION == 'N2S' else None
        probs_reinsertion = F.softmax(action_reinsertion_table, dim = -1)
        # fixed action
        if fixed_action is not None:
            p_selected = fixed_action[:,1]
            d_selected = fixed_action[:,2]
            pair_index = p_selected * gs + d_selected
            pair_index = pair_index.view(-1,1)
            action = fixed_action
        else:
            if TYPE_REINSERTION == 'greedy':
                action_reinsertion_random = probs_reinsertion_random.multinomial(1)
                action_reinsertion_greedy = probs_reinsertion.max(-1)[1].unsqueeze(1)
                pair_index = torch.where(torch.rand(bs,1).to(h_em.device) < 0.1, action_reinsertion_random, action_reinsertion_greedy)
            else:
                # sample one action
                pair_index = probs_reinsertion.multinomial(1)
            
            p_selected = pair_index // gs 
            d_selected = pair_index % gs     
            action = torch.cat((action_removal.view(bs, -1), p_selected, d_selected),-1)  # pair: no_head bs, 2
        
        selected_log_ll_action2 = log_ll_reinsertion.gather(1, pair_index)  if self.training and TYPE_REINSERTION == 'N2S' else torch.tensor(0).to(h.device)
        
        log_ll = selected_log_ll_action1 + selected_log_ll_action2
        
        if require_entropy and self.training:
            dist = Categorical(probs_reinsertion, validate_args=False)
            entropy = dist.entropy()
        else:
            entropy = None

        return action, log_ll, entropy


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalization = normalization

        if not self.normalization == 'layer':
            self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.normalization == 'layer':
            return (input - input.mean((1,2)).view(-1,1,1)) / torch.sqrt(input.var((1,2)).view(-1,1,1) + 1e-05)

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input

class MultiHeadEncoder(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadEncoder, self).__init__()
        
        self.MHA_sublayer = MultiHeadAttentionsubLayer(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden,
                        normalization=normalization,
                )
        
        self.FFandNorm_sublayer = FFandNormsubLayer(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden,
                        normalization=normalization,
                )
        
    def forward(self, input1, input2):
        out1, out2 = self.MHA_sublayer(input1, input2)
        return self.FFandNorm_sublayer(out1), out2
    
    
class MultiHeadAttentionsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionsubLayer, self).__init__()
        
        self.MHA = MultiHeadAttentionNew(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
        
        self.Norm = Normalization(embed_dim, normalization)
    
    def forward(self, input1, input2):
        # Attention and Residual connection
        out1, out2 = self.MHA(input1, input2)
        
        # Normalization
        return self.Norm(out1 + input1), out2
   
class FFandNormsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(FFandNormsubLayer, self).__init__()
        
        self.FF = nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden, bias = False),
                    nn.ReLU(inplace = True),
                    nn.Linear(feed_forward_hidden, embed_dim, bias = False)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim, bias = False)
        
        self.Norm = Normalization(embed_dim, normalization)
        
    
    def forward(self, input):
    
        # FF and Residual connection
        out = self.FF(input)
        # Normalization
        return self.Norm(out + input)


class EmbeddingNet(nn.Module):
    
    def __init__(
            self,
            node_dim,
            embedding_dim,
            seq_length,
        ):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(node_dim, embedding_dim, bias = False)

        self.pattern = self.cyclic_position_encoding_pattern(seq_length, embedding_dim)
        
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
            
    def basesin(self, x, T, fai = 0):
        return np.sin(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)
    
    def basecos(self, x, T, fai = 0):
        return np.cos(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)
    
    def cyclic_position_encoding_pattern(self, n_position, emb_dim, mean_pooling = True):
        
        Td_set = np.linspace(np.power(n_position, 1 / (emb_dim // 2)), n_position, emb_dim // 2, dtype = 'int')
        x = np.zeros((n_position, emb_dim))
         
        for i in range(emb_dim):
            Td = Td_set[i //3 * 3 + 1] if  (i //3 * 3 + 1) < (emb_dim // 2) else Td_set[-1]
            fai = 0 if i <= (emb_dim // 2) else  2 * np.pi * ((-i + (emb_dim // 2)) / (emb_dim // 2))
            longer_pattern = np.arange(0, np.ceil((n_position) / Td) * Td, 0.01)
            if i % 2 ==1:
                x[:,i] = self.basecos(longer_pattern, Td, fai)[np.linspace(0, len(longer_pattern), n_position, dtype = 'int', endpoint = False)]
            else:
                x[:,i] = self.basesin(longer_pattern, Td, fai)[np.linspace(0, len(longer_pattern), n_position, dtype = 'int', endpoint = False)]
                
        pattern = torch.from_numpy(x).type(torch.FloatTensor)
        pattern_sum = torch.zeros_like(pattern)

        # averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(n_position)
        pooling = [0] if not mean_pooling else[-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + n_position) % n_position
            pattern_sum += pattern.gather(0, index.view(-1,1).expand_as(pattern))
        pattern = 1. / time * pattern_sum - pattern.mean(0)
        #### ---- 
        
        return pattern    

    def position_encoding(self, solutions, embedding_dim, clac_stacks = False):
         batch_size, seq_length = solutions.size()
         half_size = seq_length // 2
         
         # expand for every batch
         position_enc_new = self.pattern.expand(batch_size, seq_length, embedding_dim).clone().to(solutions.device)
         
         # get index according to the solutions
         visited_time = torch.zeros((batch_size,seq_length),device = solutions.device)
         
         pre = torch.zeros((batch_size),device = solutions.device).long()
         
         arange = torch.arange(batch_size)
         if clac_stacks: 
             stacks = torch.zeros(batch_size, half_size + 1, device = solutions.device) - 0.01 # fix bug: topk is not stable sorting
             top2 = torch.zeros(batch_size, seq_length, 2,device = solutions.device).long()
             stacks[arange, pre] = 0  # fix bug: topk is not stable sorting
         
         for i in range(seq_length):
             current_nodes = solutions[arange,pre]
             visited_time[arange,current_nodes] = i+1
             pre = solutions[arange,pre]
             
             if clac_stacks:
                 index1 = (current_nodes <= half_size)& (current_nodes > 0)
                 index2 = (current_nodes > half_size)& (current_nodes > 0)
                 if index1.any():
                     stacks[index1, current_nodes[index1]] = i + 1
                 if (index2).any():
                     stacks[index2, current_nodes[index2] - half_size] = -0.01  # fix bug: topk is not stable sorting
                 top2[arange, current_nodes] = stacks.topk(2)[1]
             
         index = (visited_time % seq_length).long().unsqueeze(-1).expand(batch_size, seq_length, embedding_dim)
         # return 
         return torch.gather(position_enc_new, 1, index), visited_time.long(), top2 if clac_stacks else None

        
    def forward(self, x, solutions, clac_stacks = False):
        pos_enc, visited_time, top2 = self.position_encoding(solutions, self.embedding_dim, clac_stacks)
        x_embedding = self.embedder(x)   
        return  x_embedding, pos_enc, visited_time, top2
    
class MultiHeadAttentionLayerforCritic(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionLayerforCritic, self).__init__(
            SkipConnection(
                    MultiHeadAttention(
                        n_heads,
                        input_dim=embed_dim,
                        embed_dim=embed_dim
                    )                
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                    nn.Sequential(
                        nn.Linear(embed_dim, feed_forward_hidden),
                        nn.ReLU(inplace = True),
                        nn.Linear(feed_forward_hidden, embed_dim,)
                    ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        ) 
