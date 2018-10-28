#-*- coding: UTF-8 -*-
#################################################################
#    > File: graph.py
#    > Author: Minghua Zhang
#    > Mail: zhangmh@pku.edu.cn
#    > Time: 2018-01-04 21:56:06
#################################################################

import tensorflow as tf
from modules import *


class Graph():
    def __init__(self, conf, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Encoder
            self.x = tf.placeholder(tf.float32, shape=[None,None,conf['option']['dim_word']], name='x')
            self.x_mask = tf.placeholder(tf.int32, shape=[None,None], name='x_mask')
            self.y = tf.placeholder(tf.float32, shape=[None,None,conf['option']['dim_word']], name='y')
            self.y_mask = tf.placeholder(tf.int32, shape=[None,None], name='y_mask')
            self.y_target = tf.placeholder(tf.int32, shape=[None,None], name='y_target')
            self.drop = tf.placeholder(tf.bool, shape=[], name='drop')
            
            self.train_inps  = {'x':self.x, 'x_mask':self.x_mask, 'drop':self.drop, 'y':self.y, 'y_mask':self.y_mask, 'y_target':self.y_target}
            self.valid_inps  = {'x':self.x, 'x_mask':self.x_mask, 'drop':self.drop, 'y':self.y, 'y_mask':self.y_mask, 'y_target':self.y_target}
            self.decode_inps = {'x':self.x, 'x_mask':self.x_mask, 'drop':self.drop}
            self.encode_inps = {'x':self.x, 'x_mask':self.x_mask, 'drop':self.drop}
            
            self.enc = self.x
            ## Positional Encoding
            self.px = tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1])
            if conf['option']['position'] == 'sin':
                self.enc += positional_encoding(self.px,
                                                vocab_size=conf['option']['maxlen']+2,
                                                num_units=conf['option']['dim_word'],
                                                zero_pad=False,
                                                scale=False,
                                                scope='enc_pos')
            elif conf['option']['position'] == 'emb':
                self.enc += embedding(self.px,
                                      vocab_size=conf['option']['maxlen']+2,
                                      num_units=conf['option']['dim_word'],
                                      zero_pad=False,
                                      scale=False,
                                      scope='enc_pos')
            else:
                pass
            
            ## Dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=conf['option']['drop_rate'],
                                         training=self.drop)
            
            ## Layers
            for i in range(conf['option']['layer_n']):
                with tf.variable_scope('enc_layers_{}'.format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                   keys=self.enc,
                                                   drop=self.drop,
                                                   dropout_rate=conf['option']['drop_rate'],
                                                   num_units=conf['option']['dim_model'],
                                                   num_heads=conf['option']['head'],
                                                   causality=False)
                    
                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[conf['option']['dim_inner'], conf['option']['dim_model']])

            ## Pooling
            enc_mask = tf.tile(tf.expand_dims(self.x_mask, -1), [1, 1, tf.shape(self.enc)[-1]])
            enc_mask_float = tf.to_float(enc_mask)
            self.enc_mean = tf.reduce_sum(self.enc * enc_mask_float, 1) / tf.reduce_sum(enc_mask_float, 1)
            
            min_paddings = tf.ones_like(self.enc)*(-2**32+1)
            self.enc_max = tf.where(tf.equal(enc_mask, 0), min_paddings, self.enc)
            self.enc_max = tf.reduce_max(self.enc_max, 1)
            
            self.ctx = tf.concat((tf.expand_dims(self.enc_mean, 1), tf.expand_dims(self.enc_max, 1)), 1)

            logits = self.decode(conf, self.y)
            self.probs = tf.nn.softmax(logits)
            self.preds = tf.to_int32(tf.argmax(logits, axis=-1))
            y_istarget = tf.to_float(self.y_mask)
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y_target))*y_istarget)
            tf.summary.scalar('acc', self.acc)
            
            y_smoothed = label_smoothing(tf.one_hot(self.y_target, depth=conf['option']['vocab_size']))
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_smoothed)                
            self.mean_loss = tf.reduce_sum(loss*y_istarget) / (tf.reduce_sum(y_istarget))
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()

            if is_training:
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.lrate = tf.Variable(conf['option']['lrate'], trainable=False)
                if conf['option']['optimizer'] == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lrate, beta1=0.9, beta2=0.98, epsilon=1e-8)
                else:
                    self.optimizer = tf.train.GradientDescentOptimizer(self.lrate)
                
                updates = tf.trainable_variables()
                grads = tf.gradients(self.mean_loss, updates)
                if conf['option']['clip_grad'] > 0.:
                    clip_grads, _ = tf.clip_by_global_norm(grads, conf['option']['clip_grad'])
                else:
                    clip_grads = grads
                self.train_op = self.optimizer.apply_gradients(zip(clip_grads, updates), global_step=self.global_step)


    def decode(self, conf, y):
        dec = y
        ## Positional Encoding
        py = tf.tile(tf.expand_dims(tf.range(tf.shape(y)[1]), 0), [tf.shape(y)[0], 1])
        if conf['option']['position'] == 'sin':
            dec += positional_encoding(py,
                                       vocab_size=conf['option']['maxlen']+2,
                                       num_units=conf['option']['dim_word'],
                                       zero_pad=False,
                                       scale=False,
                                       scope='dec_pos')
        elif conf['option']['position'] == 'emb':
            dec += embedding(py,
                             vocab_size=conf['option']['maxlen']+2,
                             num_units=conf['option']['dim_word'],
                             zero_pad=False, 
                             scale=False,
                             scope='dec_pos')
        else:
            pass

        ## Dropout
        dec = tf.layers.dropout(dec,
                                rate=conf['option']['drop_rate'],
                                training=self.drop)

        ## Layers
        for i in range(conf['option']['layer_n']):
            with tf.variable_scope('dec_s_layers_{}'.format(i)):
                ## Multihead Attention ( self-attention)
                dec = multihead_attention(queries=dec,
                                          keys=dec,
                                          drop=self.drop,
                                          dropout_rate=conf['option']['drop_rate'],
                                          num_units=conf['option']['dim_model'],
                                          num_heads=conf['option']['head'],
                                          causality=True,
                                          scope='self_attention')
                
                ## Multihead Attention ( vanilla attention)
                dec = multihead_attention(queries=dec,
                                          keys=self.ctx,
                                          drop=self.drop,
                                          dropout_rate=conf['option']['drop_rate'],
                                          num_units=conf['option']['dim_model'],
                                          num_heads=conf['option']['head'],
                                          causality=False,
                                          residual=True,
                                          scope='vanilla_attention')
                
                ## Feed Forward
                dec = feedforward(dec, num_units=[conf['option']['dim_inner'], conf['option']['dim_model']])

        # Final linear projection
        logits = tf.layers.dense(dec, conf['option']['vocab_size'])

        return logits


