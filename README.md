# Sentence_Restoration
Sentence Restoration from Automated Speech Recognition Transcripts. Unlike Sentence Boundary Disambiguation or Punctuation Restoration, this project has the limited but important (from an NLP perspective) task of taking automated speech transcripts which have zero punctuation and building sentences from them, necessary for all downstream NLP tasks (e.g. usage of BERT and similar large language models, POS taggers, NER, etc.)

Requirements: Tensorflow 2.10.1, and non-conflicting versions of nltk, pyodbc, numpy, sklearn, GloVe pretrained vectors (https://nlp.stanford.edu/projects/glove) - this code used the version from the Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors)
and Ted Talk transcripts (or your own collection of transcripts).
