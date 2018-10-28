#-*- coding: UTF-8 -*-
#################################################################
#    > File: data.py
#    > Author: Minghua Zhang
#    > Mail: zhangmh@pku.edu.cn
#    > Time: 2018-01-04 23:47:06
#################################################################

import numpy
import copy
import logging


class Data():

    def __init__(self, path_to_data, word2idx, sep=None, batch_size=128, minlen=1, maxlen=150, fresh=False):
        self.sep = sep
        self.batch_size = batch_size
        self.minlen = minlen
        self.maxlen = maxlen
        self.fresh = fresh
        
        self.read_data(path_to_data)
        self.prepare(word2idx)
        self.reset()

    def read_data(self, path_to_data):
        with open(path_to_data, 'rU') as fin:
            lines = fin.readlines()
       
        self.text = list()
        self.total = len(lines)
        for i in xrange(self.total):
            line = lines[i].strip(' \n').decode('utf-8')
            self.text.append(line)
            
            if (i+1) % 10000000 == 0:
                logging.info('reading data line %d' % (i+1))
        logging.info('reading data line %d' % self.total)

    def prepare(self, word2idx):
        self.idxs = list()
        self.lengths = list()
        
        for i in xrange(self.total):
            sent_len, sent_true = self.sent_judge(self.text[i], word2idx)
            if sent_true:
                self.idxs.append( i )
                self.lengths.append( sent_len )
        
        self.qtotal = len(self.idxs)
        self.len_unique = numpy.unique(self.lengths)
        self.len_indices = dict()
        self.len_counts = dict()
        for ll in self.len_unique:
            self.len_indices[ll] = numpy.where(self.lengths == ll)[0]
            self.len_counts[ll] = len(self.len_indices[ll])

    def sent_judge(self, sent, word2idx):
        words = sent.split(self.sep)
        nwords = len(words)
        lenTrue = ((nwords <= self.maxlen) and (nwords >= self.minlen))
        if not lenTrue:
            return nwords, False
        else:
            unk_count = 0
            for w in words:
                if not word2idx.has_key(w):
                    unk_count += 1
            if unk_count > 0:
                return nwords, False
            else:
                return nwords, True

    def reset(self):
        self.len_curr_counts = copy.copy(self.len_counts)
        self.len_unique = numpy.random.permutation(self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0
            self.len_indices[ll] = numpy.random.permutation(self.len_indices[ll])
        self.len_idx = -1

    def next(self):
        count = 0
        while True:
            self.len_idx = numpy.mod(self.len_idx+1, len(self.len_unique))
            if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break
        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration()

        # get the batch size
        curr_batch_size = numpy.minimum(self.batch_size, self.len_curr_counts[self.len_unique[self.len_idx]])
        curr_pos = self.len_indices_pos[self.len_unique[self.len_idx]]
        # get the indices for the current batch
        curr_indices = self.len_indices[self.len_unique[self.len_idx]][curr_pos:curr_pos+curr_batch_size]
        self.len_indices_pos[self.len_unique[self.len_idx]] += curr_batch_size
        self.len_curr_counts[self.len_unique[self.len_idx]] -= curr_batch_size

        if self.fresh:
            self.reset()

        batch_data = [self.text[self.idxs[i]] +' </s>' for i in curr_indices]
        return batch_data

    def __iter__(self):
        return self


def prepare_data(batch_data, word2vec, word2idx, word_dim=300, sep=None):

    batch_data_ = list()
    for i in xrange(len(batch_data)):
        batch_data_.append( [w for w in batch_data[i].split(sep) if w in word2vec] )

    lens = list()
    for i in xrange(len(batch_data_)):
        lens.append( len(batch_data_[i]) )
    max_len = numpy.max(lens)
    n_batches = len(batch_data_)

    x = numpy.zeros((n_batches, max_len, word_dim), dtype='float32')
    x_mask = numpy.zeros((n_batches, max_len), dtype='int32')
    y = numpy.zeros((n_batches, max_len, word_dim), dtype='float32')
    y_mask = numpy.zeros((n_batches, max_len), dtype='int32')
    y_target = numpy.zeros((n_batches, max_len), dtype='int32')

    for i in xrange(n_batches):
        x_mask[i, :lens[i]] = 1
        y_mask[i, :lens[i]] = 1
        for j in xrange(lens[i]):
            x[i, j, :] = word2vec[batch_data_[i][j]]
            y_target[i, j] = word2idx[batch_data_[i][j]]
        
        y[i, 0, :] = word2vec['<s>']
        for j in xrange(lens[i]-1):
            y[i, j+1, :] = word2vec[batch_data_[i][j]]

    return {'x':x, 'x_mask':x_mask, 'y':y, 'y_mask':y_mask, 'y_target':y_target}


