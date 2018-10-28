#-*- coding: UTF-8 -*-
#################################################################
#    > File: run_eval.py
#    > Author: Minghua Zhang
#    > Mail: zhangmh@pku.edu.cn
#    > Time: 2018-01-07 15:51:06
#################################################################

import os
import json
import codecs
import logging
import ray
import subprocess
import master
import senteval
from collections import defaultdict


def prepare(params, samples):
    vocab = defaultdict(lambda : 0)
    for s in samples:
        for word in s:
            vocab[word] = 1
    vocab['<s>'] = 1
    vocab['</s>'] = 1
    params.master.build_emb(vocab)


def batcher(params, batch):
    batch = [' '.join(sent +['</s>']) for sent in batch]
    embeddings = params.master.encode(batch, use_norm=True)
    return embeddings


@ray.remote
def call_eval(task, call_id):
    fileHandler = logging.FileHandler(os.path.abspath('.')+'/log.eval.'+str(call_id), mode='w', encoding='UTF-8')
    formatter = logging.Formatter('%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S')
    fileHandler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)

    m = master.Master('conf.json')
    m.creat_graph()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    m.prepare()

    params_senteval = {'task_path':m.conf['path']['tasks'], 'usepytorch':False, 'kfold':10}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                     'tenacity': 5, 'epoch_size': 4}
    params_senteval['master'] = m

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    result = se.eval(task)
    m.sess.close()
    return result


if __name__ == '__main__':
    
    with codecs.open('conf.json', 'r', 'utf-8') as fin:
        conf = json.load(fin)

    ray.init(num_gpus=1, redirect_output=True)
    tasks = [['MR'], ['SUBJ'], ['MPQA'], ['CR', 'SST2', 'TREC', 'MRPC', 'SICKRelatedness', 'SICKEntailment', 'STS14']]
    outs = ray.get( [call_eval.remote(tasks[i], i) for i in xrange(len(tasks))] )

    results = dict()
    for result in outs:
        results.update(result)
    resultstr = 'MR:%.2f  CR:%.2f  SUBJ:%.2f  MPQA:%.2f  SST2:%.2f  TREC:%.1f  MRPC:%.2f/%.2f  SICK-E:%.2f  SCIK-R:%.3f  STS14:%.2f/%.2f' % (
                results['MR']['acc'], results['CR']['acc'], results['SUBJ']['acc'], results['MPQA']['acc'], results['SST2']['acc'],
                results['TREC']['acc'], results['MRPC']['acc'], results['MRPC']['f1'], results['SICKEntailment']['acc'],
                results['SICKRelatedness']['pearson'], results['STS14']['all']['pearson']['wmean'], results['STS14']['all']['spearman']['wmean'])
    
    cmd = ''
    for i in xrange(len(tasks)):
        cmd += "cat %s/log.eval.%d >>log.eval;" % (os.path.abspath('.'), i)
    cmd += r"rm %s/log.eval.*; echo %s>>log.eval;" % (os.path.abspath('.'), resultstr)
    subprocess.check_call(cmd, shell=True)


