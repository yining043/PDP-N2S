from torch.utils.data import Dataset
import torch
import pickle
import os

class PDTSP(object):

    NAME = 'pdtsp'  #Pickup and Delivery TSP
    
    def __init__(self, p_size, init_val_met = 'p2d', with_assert = False):
        
        self.size = p_size          # the number of nodes in PDTSP 
        self.do_assert = with_assert
        self.init_val_met = init_val_met
        self.state = 'eval'
        print(f'PDTSP with {self.size} nodes.', 
              ' Do assert:', with_assert,)
    
    def input_feature_encoding(self, batch):
        return batch['coordinates']
    
    def get_visited_order_map(self, visited_time):
        bs, gs = visited_time.size()
        visited_time = visited_time % gs
        
        return visited_time.view(bs, gs, 1) > visited_time.view(bs, 1, gs)

        
    def get_real_mask(self, selected_node, visited_order_map):
        
        bs, gs, _ = visited_order_map.size()

        mask = visited_order_map.clone()
        mask[torch.arange(bs), selected_node.view(-1)] = True
        mask[torch.arange(bs), selected_node.view(-1) + gs // 2] = True
        mask[torch.arange(bs),:,selected_node.view(-1)] = True
        mask[torch.arange(bs),:,selected_node.view(-1) + gs // 2] = True
        
        return mask
    
    def get_initial_solutions(self, batch, val_m = 1):
        
        batch_size = batch['coordinates'].size(0)
    
        def get_solution(methods):
            
            half_size = self.size// 2
            
            if methods == 'random':
                candidates = torch.ones(batch_size,self.size + 1).bool()
                candidates[:,half_size + 1:] = 0
                rec = torch.zeros(batch_size, self.size + 1).long()
                selected_node = torch.zeros(batch_size, 1).long()
                candidates.scatter_(1, selected_node, 0)
                
                for i in range(self.size):
                    dists = torch.ones(batch_size, self.size + 1)
                    dists.scatter_(1, selected_node, -1e20)
                    dists[~candidates] = -1e20
                    dists = torch.softmax(dists, -1)
                    next_selected_node = dists.multinomial(1).view(-1,1)
                    
                    add_index = (next_selected_node <= half_size).view(-1)
                    pairing = next_selected_node[next_selected_node <= half_size].view(-1,1) + half_size
                    candidates[add_index] = candidates[add_index].scatter_(1, pairing, 1)
                    
                    rec.scatter_(1,selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node
                    
                return rec
            
            elif methods == 'greedy':
                
                candidates = torch.ones(batch_size,self.size + 1).bool()
                candidates[:,half_size + 1:] = 0
                rec = torch.zeros(batch_size, self.size + 1).long()
                selected_node = torch.zeros(batch_size, 1).long()
                candidates.scatter_(1, selected_node, 0)
                
                for i in range(self.size):
                    
                    d1 = batch['coordinates'].cpu().gather(1, selected_node.unsqueeze(-1).expand(batch_size, self.size + 1, 2))
                    d2 = batch['coordinates'].cpu()
                    
                    dists = (d1 - d2).norm(p=2, dim=2)
                    # dists = batch['dist'].cpu().gather(1,selected_node.view(batch_size,1,1).expand(batch_size, 1, self.size + 1)).squeeze().clone()
                    dists.scatter_(1, selected_node, 1e6)
                    dists[~candidates] = 1e6
                    next_selected_node = dists.min(-1)[1].view(-1,1)
                    
                    add_index = (next_selected_node <= half_size).view(-1)
                    pairing = next_selected_node[next_selected_node <= half_size].view(-1,1) + half_size
                    candidates[add_index] = candidates[add_index].scatter_(1, pairing, 1)
                    
                    rec.scatter_(1,selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node
                    
                return rec
                
            else:
                raise NotImplementedError()

        return get_solution(self.init_val_met).expand(batch_size, self.size + 1).clone()
    
    def step(self, batch, rec, exchange, pre_bsf, action_record):
        
        bs, gs = rec.size()
        pre_bsf = pre_bsf.view(bs,-1)
        
        cur_vec = action_record.pop(0) * 0.
        cur_vec[torch.arange(bs), exchange[:,0]] = 1.
        action_record.append(cur_vec)
        
        selected = exchange[:,0].view(bs,1)
        first = exchange[:,1].view(bs,1)
        second = exchange[:,2].view(bs,1)
        
        next_state = self.insert_star(rec, selected + 1, first, second)
        
        new_obj = self.get_costs(batch, next_state)
        
        now_bsf = torch.min(torch.cat((new_obj[:,None], pre_bsf[:,-1, None]),-1),-1)[0]
        
        reward = pre_bsf[:,-1] - now_bsf
        
        return next_state, reward, torch.cat((new_obj[:,None], now_bsf[:,None]),-1), action_record

    def insert_star(self, solution, pair_index, first, second):
        
        rec = solution.clone()
        bs, gs = rec.size()
        
        # fix connection for pairing node
        argsort = rec.argsort()
        pre_pairfirst = argsort.gather(1, pair_index)
        post_pairfirst = rec.gather(1, pair_index)
        rec.scatter_(1,pre_pairfirst, post_pairfirst)
        rec.scatter_(1,pair_index, pair_index)
        
        argsort = rec.argsort()
        
        pre_pairsecond = argsort.gather(1, pair_index + gs // 2)
        post_pairsecond = rec.gather(1, pair_index + gs // 2)
        
        rec.scatter_(1,pre_pairsecond,post_pairsecond)
        
        # fix connection for pairing node
        post_second = rec.gather(1,second)
        rec.scatter_(1,second, pair_index + gs // 2)
        rec.scatter_(1,pair_index + gs // 2, post_second)
        
        post_first = rec.gather(1,first)
        rec.scatter_(1,first, pair_index)
        rec.scatter_(1,pair_index, post_first)        
        
        return rec
        
    def check_feasibility(self, rec):
        
        p_size = self.size
        
        assert (
            (torch.arange(p_size + 1, out=rec.new())).view(1, -1).expand_as(rec)  == 
            rec.sort(1)[0]
        ).all(), ((
            (torch.arange(p_size + 1, out=rec.new())).view(1, -1).expand_as(rec)  == 
            rec.sort(1)[0]
        ),"not visiting all nodes", rec)
        
        # calculate visited time
        bs = rec.size(0)
        visited_time = torch.zeros((bs,p_size),device = rec.device)
        pre = torch.zeros((bs),device = rec.device).long()
        for i in range(p_size):
            visited_time[torch.arange(bs),rec[torch.arange(bs),pre] - 1] = i + 1
            pre = rec[torch.arange(bs),pre]

        assert (
            visited_time[:, 0: p_size // 2] < 
            visited_time[:, p_size // 2:]
        ).all(), (visited_time[:, 0: p_size // 2] < 
            visited_time[:, p_size // 2:],"deliverying without pick-up")
    
    
    def get_swap_mask(self, selected_node, visited_order_map, top2=None):
        return self.get_real_mask(selected_node, visited_order_map)
        
    
    def get_costs(self, batch, rec):
        
        batch_size, size = rec.size()
        
        # check feasibility
        if self.do_assert:
            self.check_feasibility(rec)
        
        # calculate obj value
        d1 = batch['coordinates'].gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
        d2 = batch['coordinates']
        length =  (d1  - d2).norm(p=2, dim=2).sum(1)
        
        return length
        
    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDPDataset(*args, **kwargs)


class PDPDataset(Dataset):
    def __init__(self, filename=None, size=20, num_samples=10000, offset=0, distribution=None):
        
        super(PDPDataset, self).__init__()
        
        self.data = []
        self.size = size

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', 'file name error'
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [self.make_instance(args) for args in data[offset:offset+num_samples]]

        else:
            self.data = [{
                    'loc': torch.FloatTensor(self.size, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor(2).uniform_(0, 1)} for i in range(num_samples)]
        
        self.N = len(self.data)
        
        # calculate distance matrix
        for i, instance in enumerate(self.data):
            self.data[i]['coordinates'] = torch.cat((instance['depot'].reshape(1, 2), instance['loc']),dim=0)
            del self.data[i]['depot']
            del self.data[i]['loc']
        print(f'{self.N} instances initialized.')
    
    def make_instance(self, args):
        depot, loc, *args = args
        grid_size = 1
        if len(args) > 0:
            depot_types, customer_types, grid_size = args
        return {
            'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
            'depot': torch.tensor(depot, dtype=torch.float) / grid_size}
    
    def calculate_distance(self, data):
        N_data = data.shape[0]
        dists = torch.zeros((N_data, N_data), dtype=torch.float)
        d1 = -2 * torch.mm(data, data.T)
        d2 = torch.sum(torch.pow(data, 2), dim=1)
        d3 = torch.sum(torch.pow(data, 2), dim = 1).reshape(1,-1).T
        dists = d1 + d2 + d3
        dists[dists < 0] = 0
        return torch.sqrt(dists)
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]