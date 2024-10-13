

import os
import json
import torch
import random
from utils.tools import *
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected
from utils.bert import get_sent_embedding




class ResGCNTreeDataset(InMemoryDataset):
    def __init__(self, root, word_embedding, word2vec, undirected, transform=None, pre_transform=None,
                 pre_filter=None, args=None):
        self.word_embedding = word_embedding
        self.word2vec = word2vec
        self.args = args
        self.undirected = undirected
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        raw_file_names = self.raw_file_names

        for filename in raw_file_names:
            y = []
            row = []
            col = []
            no_root_row = []
            no_root_col = []

            filepath = os.path.join(self.raw_dir, filename)
            post = json.load(open(filepath, 'r', encoding='utf-8'))
            x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
            
            if 'label' in post['source'].keys():
                y.append(post['source']['label'])
            for i, comment in enumerate(post['comment']):
                x = torch.cat(
                    [x, self.word2vec.get_sentence_embedding(comment['content']).view(1, -1)], 0)
            
                if comment['parent'] != -1:
                    no_root_row.append(comment['parent'] + 1)
                    no_root_col.append(comment['comment id'] + 1)
                row.append(comment['parent'] + 1)
                col.append(comment['comment id'] + 1)

            edge_index = [row, col]
            no_root_edge_index = [no_root_row, no_root_col]
            y = torch.LongTensor(y)
            edge_index = to_undirected(torch.LongTensor(edge_index)) if self.undirected else torch.LongTensor(edge_index)
            no_root_edge_index = torch.LongTensor(no_root_edge_index)

            state = post['state']

            # Populate the matrix with state counts from each hop
            user_state = []
            for hop in sorted(state.keys(), key=lambda x: int(x.split('-')[0])):
                row = [0, state[hop]['state_0'], state[hop]['state_1']]
                user_state.append(row)

            num_hop = len(user_state)

            # padding
            for i in range(self.args.max_hop-len(user_state)):
                row = [0, 0, 0]
                user_state.append(row)

            # Convert the list to a PyTorch tensor for matrix format
            user_state = torch.tensor(user_state, dtype=torch.float32)
            user_state = user_state.unsqueeze(0) # to (1, n, 2)
            

            one_data = Data(x=x, y=y, edge_index=edge_index, no_root_edge_index=no_root_edge_index, user_state=user_state, num_hop=num_hop) if 'label' in post['source'].keys() else \
                Data(x=x, edge_index=edge_index, no_root_edge_index=no_root_edge_index, user_state=user_state, num_hop=num_hop)
            data_list.append(one_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])

class TreeDataset(InMemoryDataset):
    def __init__(self, root, word_embedding, word2vec, transform=None, pre_transform=None,
                 pre_filter=None, args=None):
        self.word_embedding = word_embedding
        self.args = args
        self.word2vec = word2vec
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        raw_file_names = self.raw_file_names

        for filename in raw_file_names:
            y = []
            row = []
            col = []

            filepath = os.path.join(self.raw_dir, filename)
            post = json.load(open(filepath, 'r', encoding='utf-8'))
            if self.word_embedding == 'word2vec':
                x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
            
            if 'label' in post['source'].keys():
                y.append(post['source']['label'])
            for i, comment in enumerate(post['comment']):
                if self.word_embedding == 'word2vec':
                    x = torch.cat(
                        [x, self.word2vec.get_sentence_embedding(comment['content']).view(1, -1)], 0)
                row.append(comment['parent'] + 1)
                col.append(comment['comment id'] + 1)

            edge_index = [row, col]
            y = torch.LongTensor(y)
            edge_index = torch.LongTensor(edge_index)
            
            state = post['state']

            # Populate the matrix with state counts from each hop
            user_state = []
            for hop in sorted(state.keys(), key=lambda x: int(x.split('-')[0])):
                row = [0, state[hop]['state_0'], state[hop]['state_1']]
                user_state.append(row)

            num_hop = len(user_state)

            # padding
            for i in range(self.args.max_hop-len(user_state)):
                row = [0, 0, 0]
                user_state.append(row)

            # Convert the list to a PyTorch tensor for matrix format
            user_state = torch.tensor(user_state, dtype=torch.float32)
            user_state = user_state.unsqueeze(0) # to (1, n, 2)

            one_data = Data(x=x, y=y, edge_index=edge_index, user_state=user_state, num_hop=num_hop) if 'label' in post['source'].keys() else \
                Data(x=x, edge_index=edge_index, user_state=user_state, num_hop=num_hop)
            data_list.append(one_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])

def split_dataset(label_source_path, label_dataset_path, k_shot=10000, split='622'):
    print('Spliting data...')
    if split == '622':
        train_split = 0.6
        test_split = 0.8
    elif split == '802':
        train_split = 0.8
        test_split = 0.8

    train_path, val_path, test_path = dataset_makedirs(label_dataset_path)

    label_file_paths = []
    for filename in os.listdir(label_source_path):
        label_file_paths.append(os.path.join(label_source_path, filename))

    all_post = []
    for filepath in label_file_paths:
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        all_post.append((post['source']['tweet id'], post))
        
    random.shuffle(all_post)
    train_post = []

    multi_class = False
    for post in all_post:
        if post[1]['source']['label'] == 2 or post[1]['source']['label'] == 3:
            multi_class = True

    num0 = 0
    num1 = 0
    num2 = 0
    num3 = 0
    for post in all_post[:int(len(all_post) * train_split)]:
        if post[1]['source']['label'] == 0 and num0 != k_shot:
            train_post.append(post)
            num0 += 1
        if post[1]['source']['label'] == 1 and num1 != k_shot:
            train_post.append(post)
            num1 += 1
        if post[1]['source']['label'] == 2 and num2 != k_shot:
            train_post.append(post)
            num2 += 1
        if post[1]['source']['label'] == 3 and num3 != k_shot:
            train_post.append(post)
            num3 += 1
        if multi_class:
            if num0 == k_shot and num1 == k_shot and num2 == k_shot and num3 == k_shot:
                break
        else:
            if num0 == k_shot and num1 == k_shot:
                break
    if split == '622':
        val_post = all_post[int(len(all_post) * train_split):int(len(all_post) * test_split)]
        test_post = all_post[int(len(all_post) * test_split):]
    elif split == '802':
        val_post = all_post[-1:]
        test_post = all_post[int(len(all_post) * test_split):]
    write_post(train_post, train_path)
    write_post(val_post, val_path)
    write_post(test_post, test_path)
    print('Finished.')