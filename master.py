#-*- coding: UTF-8 -*-
#################################################################
#    > File: master.py
#    > Author: Minghua Zhang
#    > Mail: zhangmh@pku.edu.cn
#    > Time: 2018-01-05 15:51:06
#################################################################

import os
import io
import json
import numpy
import codecs
import logging
import subprocess
from collections import defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize
import tensorflow as tf
import graph
import data


class Master:
    def __init__(self, fconf):
        # init conf
        logging.info('Initializing conf...')
        self.init_conf(fconf)


    def init_conf(self, fconf):
        with codecs.open(fconf, 'r', 'utf-8') as fin:
            self.conf = json.load(fin)
        
        srcpath = os.path.split(os.path.abspath(__file__))[0]
        for k in self.conf['path']:
            self.conf['path'][k] = self.conf['path'][k] % (srcpath)


    def load_vocab(self):
        with codecs.open(self.conf['path']['vocab'], 'r', 'utf-8') as fin:
            lines = fin.readlines()
        
        vocab = defaultdict(lambda : 0)
        for i in xrange(len(lines)):
            word,freq = lines[i].strip('\n').split()
            if int(freq) >= self.conf['option']['word_freq']:
                vocab[word] = 1
        return vocab


    def build_vocab(self, sentences, tokenize=False):
        vocab = defaultdict(lambda : 0)
        sentences = [s.split() if not tokenize else word_tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                vocab[word] = 1
        vocab['<s>'] = 1
        vocab['</s>'] = 1
        return vocab


    def build_emb(self, vocab):
        self.w2v = dict()
        with io.open(self.conf['path']['w2v'], 'r', encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in vocab:
                    self.w2v[word] = numpy.array(list(map(float, vec.split())))
        
        words = self.w2v.keys()
        self.conf['option']['vocab_size'] = len(words)
        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.idx2word = {idx: word for idx, word in enumerate(words)}


    def load_data(self):
        # load corpus
        logging.info('Loading corpus ...')
        self.train_iter = data.Data(self.conf['path']['train'], 
                                    self.word2idx,
                                    sep=None, 
                                    batch_size=self.conf['option']['batch_size'],
                                    minlen=self.conf['option']['minlen'], 
                                    maxlen=self.conf['option']['maxlen'], 
                                    fresh=False)
        logging.info( 'train/Total : %d/%d' % (self.train_iter.qtotal, self.train_iter.total) )
        
        self.valid_iter = data.Data(self.conf['path']['valid'], 
                                    self.word2idx,
                                    sep=None, 
                                    batch_size=self.conf['option']['batch_size'],
                                    minlen=self.conf['option']['minlen'], 
                                    maxlen=self.conf['option']['maxlen'], 
                                    fresh=False)
        logging.info( 'valid/Total : %d/%d' % (self.valid_iter.qtotal, self.valid_iter.total) )
        
        logging.info('Loading decode ...')
        self.decode_iter = data.Data(self.conf['path']['decode'], 
                                    self.word2idx, 
                                    sep=None, 
                                    batch_size=self.conf['option']['decode_bs'],
                                    minlen=self.conf['option']['minlen'], 
                                    maxlen=self.conf['option']['maxlen'], 
                                    fresh=False)
        logging.info( 'decode/Total : %d/%d' % (self.decode_iter.qtotal, self.decode_iter.total) )


    def creat_graph(self):
        # build graph
        logging.info('Build graph...')
        self.g = graph.Graph(self.conf, is_training=self.conf['option']['is_training'])


    def prepare(self):
        config = tf.ConfigProto(log_device_placement=False)
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with self.g.graph.as_default():
            self.sess = tf.Session(graph=self.g.graph, config=config)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=None)
            
            chk_path = self.conf['path']['models'] + self.conf['option']['model']
            if tf.train.checkpoint_exists(chk_path):
                self.saver.restore(self.sess, chk_path)


    def train(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.prepare()
        summary_writer = tf.summary.FileWriter(self.conf['path']['models'], graph=self.sess.graph)

        avg_loss = 0
        batch = self.sess.run(self.g.global_step)
        while batch < self.conf['option']['finish']:
            batch_data = self.train_iter.next()
            np_data = data.prepare_data(batch_data, self.w2v, self.word2idx)
            np_data['drop'] = True

            inps = dict()
            for k in self.g.train_inps:
                inps[ self.g.train_inps[k] ] = np_data[k]

            loss, _ = self.sess.run([self.g.mean_loss, self.g.train_op], feed_dict=inps)
            batch = self.sess.run(self.g.global_step)
            if numpy.isnan(loss) or numpy.isinf(loss):
                logging.info('NaN detected')
                return

            avg_loss += loss
            if numpy.mod(batch, self.conf['option']['dispFreq']) == 0:
                avg_loss /= self.conf['option']['dispFreq']
                logging.info('batch:%-5d  lr=%.4f  loss:%-8.2f' % (batch, self.g.lrate.eval(session=self.sess), avg_loss))
                avg_loss = 0

            if numpy.mod(batch, self.conf['option']['summaryFreq']) == 0:
                summary_writer.add_summary(self.sess.run(self.g.merged, feed_dict=inps), batch)

            if numpy.mod(batch, self.conf['option']['validFreq']) == 0:
                loss, acc = self.valid_loss()
                logging.info('batch:%-5d  loss:%-5.2f  acc:%-5.2f' % (batch, loss, acc))

            if numpy.mod(batch, self.conf['option']['saveFreq']) == 0:
                self.saver.save(self.sess, self.conf['path']['models']+'model', global_step=batch)
                logging.info('batch:%-5d save model' % (batch))

            if numpy.mod(batch, self.conf['option']['decodeFreq']) == 0:
                self.greedy_decode(self.decode_iter, '%s.%d' % (self.conf['path']['models']+'decode', batch))
                logging.info('batch:%-5d greedy decode' % (batch))
        
        self.sess.close()


    def valid_loss(self):
        batch = 0
        dev_loss = 0.
        dev_acc = 0.
        dev_step = 0.
        
        for batch_data in self.valid_iter:
            batch += 1
            np_data = data.prepare_data(batch_data, self.w2v, self.word2idx)
            np_data['drop'] = False

            inps = dict()
            for k in self.g.valid_inps:
                inps[ self.g.valid_inps[k] ] = np_data[k]

            loss, acc = self.sess.run([self.g.mean_loss, self.g.acc], feed_dict=inps)
            dev_loss += loss
            dev_acc += acc
            dev_step += numpy.sum(np_data['y_mask'])

        dev_loss /= batch
        dev_acc /= dev_step
        return dev_loss, dev_acc


    def greedy_decode(self, decode_iter, path_to_decode):
        with codecs.open(path_to_decode, 'w', 'utf-8') as fdecode:
            n_samples = 0
            for batch_data in decode_iter:
                np_data = data.prepare_data(batch_data, self.w2v, self.word2idx)
                np_data['drop'] = False

                inps = dict()
                for k in self.g.decode_inps:
                    inps[ self.g.decode_inps[k] ] = np_data[k]

                batchx, lenx, _ = np_data['x'].shape
                maxlen = int( 1.5 * (lenx-1) )
                preds = numpy.zeros((batchx, maxlen), numpy.int32)
                y = numpy.zeros((batchx, maxlen, self.conf['option']['dim_word']), dtype='float32')
                for j in range(maxlen):
                    for i in xrange(batchx):
                        if j==0:
                            y[i, j, :] = self.w2v[ '<s>' ]
                        else:
                            y[i, j, :] = self.w2v[ self.idx2word[preds[i][j-1]] ]
                    
                    inps[ self.g.y ] = y
                    preds_ = self.sess.run(self.g.preds, feed_dict=inps)
                    preds[:, j] = preds_[:, j]

                for i in xrange(batchx):
                    if preds[i,0] == self.word2idx['</s>']:
                        continue

                    fdecode.write( 'T-%d\t' % (n_samples+i) )
                    for idx in np_data['y_target'][i,:]:
                        if idx == self.word2idx['</s>']:
                            break
                        fdecode.write('%s ' % self.idx2word[idx])
                    fdecode.write('\n')
                    
                    fdecode.write( 'S-%d\t' % (n_samples+i) )
                    for idx in preds[i,1:]:
                        if idx == self.word2idx['</s>']:
                            break
                        fdecode.write('%s ' % self.idx2word[idx])
                    fdecode.write('\n')
                n_samples += batchx

        cmd = r"./bleu.sh %s %s.bleu" % (path_to_decode, path_to_decode)
        subprocess.check_call(cmd, shell=True)
        with codecs.open('%s.bleu'%path_to_decode, 'r', 'utf-8') as fin:
            lines = fin.readlines()
            logging.info(lines[0].strip(' -1=\n'))


    def encode(self, task_data, tokenize=False, use_norm=True):
        features = numpy.zeros((len(task_data), 2*self.conf['option']['dim_model']), dtype='float32')

        ds = defaultdict(list)
        captions = [s.split() if not tokenize else word_tokenize(s) for s in task_data]
        for i,s in enumerate(captions):
            ds[len(s)].append(i)

        for l in ds.keys():
            numbatches = len(ds[l]) / self.conf['option']['encode_bs'] + 1
            for minibatch in range(numbatches):
                caps = ds[l][minibatch::numbatches]
                
                batch_data = [task_data[c] for c in caps]
                np_data = data.prepare_data(batch_data, self.w2v, self.word2idx)
                np_data['drop'] = False
                
                inps = dict()
                for k in self.g.encode_inps:
                    inps[ self.g.encode_inps[k] ] = np_data[k]
                
                ctx = self.sess.run(self.g.ctx, feed_dict=inps)
                if use_norm:
                    for j in range(ctx.shape[0]):
                        for jj in range(ctx.shape[1]):
                            ctx[j, jj] /= norm(ctx[j, jj])
                
                feas = numpy.reshape(ctx, [ctx.shape[0], -1])
                for ind, c in enumerate(caps):
                    features[c] = feas[ind]
        
        return features



