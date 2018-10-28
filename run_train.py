#-*- coding: UTF-8 -*-
#################################################################
#    > File: run_train.py
#    > Author: Minghua Zhang
#    > Mail: zhangmh@pku.edu.cn
#    > Time: 2018-01-04 16:33:06
#################################################################

import os
import sys
import logging
import master


def main():

    m = master.Master('conf.json')
    vocab = m.load_vocab()
    m.build_emb(vocab)
    m.load_data()
    m.creat_graph()
    m.train()
    logging.info("Done Train !")


if __name__ == '__main__':
    if len(sys.argv) != 1:
        exit(1)
    else:
        # logging
        fileHandler = logging.FileHandler(os.path.abspath('.')+'/log.train', mode='w', encoding='UTF-8')
        formatter = logging.Formatter('%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S')
        fileHandler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(fileHandler)
        
        main()


