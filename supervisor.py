import torch
from utils.tools import get_obj_from_str
import os
from utils.word2vec import *
from utils.dataloader import *
from model.EIN_ResGCN import ResGCN
from model.EIN_BiGCN import BiGCN
from trainer.EIN_trainer import EINTrainer




def EIN_ResGCN_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    label_source_path = os.path.join('dataset', args.dataset, 'source')
    label_dataset_path = os.path.join('dataset', args.dataset, 'dataset')


    model_path = os.path.join('word2vec',
                        f'w2v_{args.dataset}_{args.tokenize_mode}_{args.vector_size}.model')

    # word2vec
    if not os.path.exists(model_path) and args.word_embedding == 'word2vec':
        sentences = collect_sentences(label_source_path, args.language, args.tokenize_mode)
        w2v_model = train_word2vec(sentences, args.vector_size, args.seed)
        w2v_model.save(model_path)
    
    word2vec = Embedding(model_path, args.language, args.tokenize_mode) if args.word_embedding == 'word2vec' else None

    # load data

    split_dataset(label_source_path, label_dataset_path, k_shot=args.k, split=args.split)

    train_path = os.path.join(label_dataset_path, 'train')
    val_path = os.path.join(label_dataset_path, 'val')
    test_path = os.path.join(label_dataset_path, 'test')

    undirected = args.undirected
    
    train_dataset = ResGCNTreeDataset(train_path, args.word_embedding, word2vec, undirected, args=args)
    val_dataset = ResGCNTreeDataset(val_path, args.word_embedding, word2vec, undirected, args=args)
    test_dataset = ResGCNTreeDataset(test_path, args.word_embedding, word2vec, undirected, args=args)
    
    base_model =  ResGCN(dataset=train_dataset, num_classes=args.num_classes, hidden=args.hidden_dim,
                            num_feat_layers=args.n_layers_feat, num_conv_layers=args.n_layers_conv,
                            num_fc_layers=args.n_layers_fc, gfn=False, collapse=False,
                            residual=args.skip_connection,
                            res_branch=args.res_branch, global_pool=args.global_pool, dropout=args.dropout,
                            edge_norm=args.edge_norm, args=args, device=device).to(device)

    optimizer = base_model.init_optimizer(args)

    datasets = [train_dataset, val_dataset, test_dataset]

    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    trainer.train_process()



def EIN_BiGCN_supervisor(args):
    init_seed(args.seed, need_deepfix=False)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    label_source_path = os.path.join('dataset', args.dataset, 'source')
    label_dataset_path = os.path.join('dataset', args.dataset, 'dataset')


    model_path = os.path.join('word2vec',
                        f'w2v_{args.dataset}_{args.tokenize_mode}_{args.vector_size}.model')

    # word2vec
    if not os.path.exists(model_path) and args.word_embedding == 'word2vec':
        sentences = collect_sentences(label_source_path, args.language, args.tokenize_mode)
        w2v_model = train_word2vec(sentences, args.vector_size, args.seed)
        w2v_model.save(model_path)
    
    word2vec = Embedding(model_path, args.language, args.tokenize_mode) if args.word_embedding == 'word2vec' else None

    # load data

    split_dataset(label_source_path, label_dataset_path, k_shot=args.k, split=args.split)

    train_path = os.path.join(label_dataset_path, 'train')
    val_path = os.path.join(label_dataset_path, 'val')
    test_path = os.path.join(label_dataset_path, 'test')

    train_dataset = TreeDataset(train_path, args.word_embedding, word2vec, args=args)
    val_dataset = TreeDataset(val_path, args.word_embedding, word2vec, args=args)
    test_dataset = TreeDataset(test_path, args.word_embedding, word2vec, args=args)

    base_model = BiGCN(args.in_feats, args.hidden_dim, args.hidden_dim, args.num_classes, args, device).to(device)


    optimizer = base_model.init_optimizer(args)

    datasets = [train_dataset, val_dataset, test_dataset]

    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    trainer.train_process()

