import json
from turtle import forward
import torch
torch.manual_seed(12345)
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor
from torchnmf.nmf import NMF
import numpy as np
from model.graph.XSimGCL import XSimGCL_Encoder
from model.graph.SimGCL import SimGCL_Encoder


class PT4Rec_Pop(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(PT4Rec_Pop, self).__init__(conf, training_set, test_set)

        args = OptionConf(self.config['PT4Rec_Pop'])
        self.n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        prompt_size = int(args['-prompt_size'])
        self.user_prompt_num = int(args['-user_prompt_num'])
        self.pretrain_model = args['-pretrain_model']
        
        # New hyperparameters for Popularity Unlearning
        self.gamma = float(args['-gamma']) if args.contain('-gamma') else 0.1
        self.top_k = int(args['-top_k']) if args.contain('-top_k') else 100

        if self.config.contain('num.max.preepoch'):
            self.maxPreEpoch = int(self.config['num.max.preepoch'])
        else:
            self.maxPreEpoch = 0

        self.user_prompt_H = True
        self.user_prompt_M = True
        self.user_prompt_R = True

        if self.pretrain_model == 'XSimGCL':
            self.eps = 0.2
            self.layer_cl = 1
            self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.layer_cl, temp)
        elif self.pretrain_model == 'SimGCL':
            self.model = SimGCL_Encoder(self.data, self.emb_size, eps=0.1, n_layers=3)

        if self.user_prompt_num != 0:         
            self.user_prompt_generator = [Prompts_Generator(self.emb_size, prompt_size).cuda() for _ in range(self.user_prompt_num)]
            self.user_attention = Attention(prompt_size, self.user_prompt_num).cuda()
            
            # Popularity prompt generator
            self.pop_prompt_generator = Prompts_Generator(self.emb_size, prompt_size).cuda()

        self.interaction_mat = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda()
        self.user_matrix, self.Item_matrix = self._adjacency_matrix_factorization()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda()
        
        # Compute top-K popular items
        self.top_k_items = self._compute_top_k_items()

    def _compute_top_k_items(self):
        item_degrees = self.data.interaction_mat.sum(axis=0).A1
        top_k_items = np.argsort(item_degrees)[-self.top_k:]
        return torch.tensor(top_k_items).cuda()

    def global_pop_records(self, item_emb):
        # Aggregate global top-K items
        top_k_emb = item_emb[self.top_k_items]
        # Mean pooling
        global_pop_emb = top_k_emb.mean(dim=0, keepdim=True)
        # Expand to all users
        return global_pop_emb.expand(self.data.user_num, -1)

    def XSimGCL_pre_train(self):
        pre_trained_model = self.model.cuda()
        optimizer = torch.optim.Adam(pre_trained_model.parameters(), lr=self.lRate)

        print('############## Pre-Training Phase ##############')
        for epoch in range(self.maxPreEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb  = pre_trained_model(True)
                cl_loss = pre_trained_model.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)
                batch_loss =  cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0:
                    print('pre-training:', epoch + 1, 'batch', n, 'cl_loss', cl_loss.item())

    def SimGCL_pre_train(self):
        try:
            self.model.load_state_dict(torch.load('./pretrained_model/SimGCL_douban_pretrain_20.pt'))
            print('############## Pre-Training Phase ##############')
            print('Load pretrained model successfully!')
            return
        except:
            print('No pretrained model, start pre-training...')

        pre_trained_model = self.model.cuda()
        optimizer = torch.optim.Adam(pre_trained_model.parameters(), lr=self.lRate)

        print('############## Pre-Training Phase ##############')
        for epoch in range(self.maxPreEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                cl_loss = pre_trained_model.cal_cl_loss([user_idx,pos_idx])
                batch_loss =  cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0:
                    print('pre-training:', epoch + 1, 'batch', n, 'cl_loss', cl_loss.item())

        torch.save(pre_trained_model.state_dict(), './pretrained_model/SimGCL_gowalla_pretrain_20.pt')    

    def _csr_to_pytorch_dense(self, csr):
        array = csr.toarray()
        dense = torch.Tensor(array)
        return dense.cuda()

    def user_historical_records(self, item_emb):
        user_profiles = torch.mm(self.interaction_mat, item_emb)
        return user_profiles

    def _adjacency_matrix_factorization(self):
        adjacency_matrix = self.data.interaction_mat
        adjacency_matrix = adjacency_matrix.toarray()
        adjacency_matrix = torch.Tensor(adjacency_matrix).cuda().t()

        print('######### Adjacency Matrix Factorization #############')
        nmf = NMF(Vshape=adjacency_matrix.shape, rank=self.emb_size).cuda()
        user_profiles = nmf.W
        item_profiles = nmf.H
        return user_profiles, item_profiles

    def _high_order_relations(self, item_emb, user_emb):
        ego_embeddings = torch.cat((user_emb, item_emb), 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_profiles, item_profiles = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_profiles, item_profiles

    def train(self):
        if self.pretrain_model == 'XSimGCL':
            self.XSimGCL_pre_train()
        elif self.pretrain_model == 'SimGCL':
            self.SimGCL_pre_train()

        model = self.model.cuda()
        params = list(model.parameters()) + list(self.user_attention.parameters()) + \
                 list(self.user_prompt_generator[0].parameters()) + \
                 list(self.user_prompt_generator[1].parameters()) + \
                 list(self.user_prompt_generator[2].parameters()) + \
                 list(self.pop_prompt_generator.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lRate)

        metrics =[]
        print('############## Downstream Training Phase ##############')
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_emb, item_emb = model()

                prompted_user_emb, prompted_item_emb = self.generate_prompts(user_emb, item_emb)

                user_idx, pos_idx, neg_idx = batch
                rec_user_emb = prompted_user_emb
                rec_item_emb = prompted_item_emb

                user_emb_batch, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb_batch, pos_item_emb, neg_item_emb)

                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb_batch, pos_item_emb)
                
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if n % 100==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())
            with torch.no_grad():
                user_emb, self.item_emb = self.model()
                prompted_user_emb, prompted_item_emb = self.generate_prompts(user_emb, self.item_emb)
                self.user_emb = prompted_user_emb
                self.item_emb = prompted_item_emb
            if epoch%5==0:
                metric = self.fast_evaluation(epoch)
                metrics.append(metric)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            user_emb, item_emb = self.model.forward()
            prompted_user_emb, prompted_item_emb = self.generate_prompts(user_emb, item_emb)
            self.best_user_emb = prompted_user_emb
            self.best_item_emb = prompted_item_emb

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
    
    def generate_prompts(self, user_emb, item_emb):
        user_prompts = []
        u = 0
        if self.user_prompt_num != 0:
            if self.user_prompt_H:
                user_prompts.append(self.user_prompt_generator[u](self.user_historical_records(item_emb)))
                u += 1
            if self.user_prompt_M:
                user_prompts.append(self.user_prompt_generator[u](self.user_matrix))
                u += 1
            if self.user_prompt_R:
                user_prompts.append(self.user_prompt_generator[u](self._high_order_relations(item_emb, user_emb)[0]))
            
            # Normal prompt attention
            user_prompt = torch.stack(user_prompts, dim=1)
            P_normal = self.user_attention(user_prompt, user_emb)
            
            # Popularity prompt
            X_pop = self.global_pop_records(item_emb)
            P_pop = self.pop_prompt_generator(X_pop)
            
            # Residual unlearning
            prompted_user_emb = (user_emb * P_normal) - self.gamma * (user_emb * P_pop)
        else:
            prompted_user_emb = user_emb

        return prompted_user_emb, item_emb


class Attention(nn.Module):
    def __init__(self, prompt_size, prompt_num):
        super(Attention, self).__init__()
        initializer = nn.init.xavier_uniform_
        self.prompt_num = prompt_num
        self.A = nn.Parameter(initializer(torch.empty(prompt_size, prompt_num)))

    def forward(self, prompt_base, embeddings):
        alpha = torch.matmul(embeddings, self.A)
        alpha = F.softmax(alpha, dim=1)
        alpha = alpha.unsqueeze(1)
        prompt = torch.bmm(alpha, prompt_base)
        prompt = prompt.squeeze(1)
        return prompt

    
class Prompts_Generator(nn.Module):
    def __init__(self, emb_size, prompt_size):
        super(Prompts_Generator, self).__init__()
        self.W = nn.Linear(emb_size, prompt_size)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        prompts = inputs
        prompts = self.W(prompts)
        prompts = self.activation(prompts)
        
        return prompts
