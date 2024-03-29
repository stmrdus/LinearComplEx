from dataloader import Data
import numpy as np
import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import time
from collections import defaultdict
from models import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import pickle
import logging
import os

def set_logger(data_name, save_path):
        directory = save_path + '/' + data_name
        if not os.path.exists(directory):
            os.makedirs(directory)
        log_file = os.path.join(directory, 'train.log')
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


class Experiment:

    def __init__(self, model_name, learning_rate=0.001, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=100, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0., hidden_dropout=0., feature_map_dropout=0.,
                 in_channels=1, out_channels=32, filt_h=3, filt_w=3, label_smoothing=0.):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout": hidden_dropout,
                       "feature_map_dropout": feature_map_dropout, "in_channels":in_channels,
                       "out_channels": out_channels, "filt_h": filt_h, "filt_w": filt_w}

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]],
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:min(
            idx+self.batch_size, len(er_vocab_pairs))]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        logging.info("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(
                predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        logging.info('Hits @10: {0}'.format(np.mean(hits[9])))
        logging.info('Hits @3: {0}'.format(np.mean(hits[2])))
        logging.info('Hits @1: {0}'.format(np.mean(hits[0])))
        logging.info('Mean rank: {0}'.format(np.mean(ranks)))
        logging.info('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))

    def train_and_eval(self):
        logging.info("Training the %s model..." % model_name)
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]                              : i for i in range(len(d.relations))}
        train_data_idxs = self.get_data_idxs(d.train_data)
        logging.info("Number of training data points: %d" % len(train_data_idxs))

        if model_name.lower() == "linearhyper":
            model = LinearHypER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        else:
            model = HypER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)

        logging.info([value.numel() for value in model.parameters()])

        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        logging.info(len(er_vocab_pairs))

        logging.info("Starting training...")

        for it in range(1, self.num_iterations+1):
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(
                    er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0])
                r_idx = torch.tensor(data_batch[:, 1])
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing) *
                               targets) + (1.0/targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
            if self.decay_rate:
                scheduler.step()
            losses.append(loss.item())

            logging.info(it)
            logging.info(np.mean(losses))

            model.eval()
            with torch.no_grad():
                logging.info("Validation:")
                self.evaluate(model, d.valid_data)
                if not it % 2:
                    logging.info("Test:")
                    self.evaluate(model, d.test_data)

# CUDA_VISIBLE_DEVICES=0 python main.py --algorithm LinearHypER --dataset WN18RR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="LinearHypER", nargs="?",
                        help='Which algorithm to use')
    parser.add_argument('--dataset', type=str, default="FB15k-237", nargs="?",
                        help='Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR')
    args = parser.parse_args()
    model_name = args.algorithm
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir, reverse=True)
    if dataset == 'FB15k-237':
        set_logger('fb15k237', 'logs')
    else:
        set_logger('wn18rr', 'logs')
    experiment = Experiment(model_name, num_iterations=2, batch_size=128, learning_rate=0.001, 
                            decay_rate=0.99, ent_vec_dim=200, rel_vec_dim=200, cuda=True, 
                            input_dropout=0.2, hidden_dropout=0.3, feature_map_dropout=0.2,
                            in_channels=1, out_channels=32, filt_h=1, filt_w=9, label_smoothing=0.1)
    experiment.train_and_eval()