## Mean-Max AAE
Sentence encoder and training code for the paper [Learning Universal Sentence Representations with Mean-Max Attention Autoencoder](https://arxiv.org/abs/1809.06590).

## Dependencies
This code is written in python. To use it you will need:
* Python 2.7
* [TensorFlow](https://www.tensorflow.org/)
* [NumPy](http://www.numpy.org/)
* [NLTK](https://www.nltk.org/)
* [SciPy](http://www.scipy.org/)
* [Ray](https://ray.readthedocs.io/en/latest/) (for parallel evaluation on transfer tasks)

## Download datasets
The pre-processed Toronto BookCorpus we used for training our model is available [here](http://yknzhu.wixsite.com/mbweb).

To download [GloVe](https://nlp.stanford.edu/projects/glove/) vector:
```bash
curl -Lo data/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip data/glove.840B.300d.zip -d data/
```

To get all the transfer tasks datasets, run (in data/):
```bash
./get_transfer_data.bash
```
This will automatically download and preprocess the transfer tasks datasets, and store them in data/.

## Sentence encoder
We provide a simple interface to encode English sentences. Get started with the following steps:

*1) Download our Mean-Max AAE models:*
```bash
curl -Lo models.zip http://Zminghua/SentEncoding/models.zip
unzip models.zip
```

*2) Make sure you have the NLTK tokenizer by running the following once:*
```python
import nltk
nltk.download('punkt')
```

*3) Load our pre-trained model:*
```python
import master
m = master.Master('conf.json')
m.creat_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
m.prepare()
```

*3) Build the vocabulary of word vectors (i.e keep only those needed):*
```python
vocab = m.build_vocab(sentences, tokenize=True)
m.build_emb(vocab)
```
where *sentences* is your list of **n** sentences.

*4) Encode your sentences:*
```python
embeddings = m.encode(sentences, tokenize=True)
```
This outputs a numpy array with **n** vectors of dimension **4096**.

## Reference
If you found this code useful, please cite the following paper:
```
  @inproceedings{zhang2018learning,
  author = {Zhang, Minghua and Wu, Yunfang and Li, Weikang and Li, Wei},
  title = {Learning Universal Sentence Representations with Mean-Max Attention Autoencoder},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year = {2018}
  }
 ```
