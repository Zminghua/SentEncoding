# Mean-Max AAE
Sentence encoder and training code from the paper [Learning Universal Sentence Representations with Mean-Max Attention Autoencoder](https://arxiv.org/abs/1809.06590).

## Dependencies

This code is written in python 2.7. To use it you will need:

* Python 2.7
* [TensorFlow](https://www.tensorflow.org/)
* [NumPy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)
* [Ray](https://ray.readthedocs.io/en/latest/) (for parallel evaluation on transfer tasks)

## Getting started

You will first need to download the model files and word embeddings.

The rest of the document will describe how to run the experiments from the paper. For these, create a folder called 'data' to store each of the datasets.

## Cite
If you use this code for your research, please cite the following paper:
```
  @inproceedings{zhang2018learning,  
  author = {Zhang, Minghua and Wu, Yunfang and Li, Weikang and Li, Wei},  
  title = {Learning Universal Sentence Representations with Mean-Max Attention Autoencoder},  
  booktitle = {EMNLP 2018},  
  year = {2018}  
  }  
 ```
