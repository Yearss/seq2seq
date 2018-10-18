import pickle
from sklearn.model_selection import train_test_split
import random

class Word:
    def __init__(self, val, tf, df):
        self.val = val
        self.tf = tf
        self.df = df


def read_data(data_fname):
    corpus = []
    with open(data_fname, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % 10000 == 0: print("progress:{0}".format(idx))
            try:
                title, abstract = line.strip().split("\t")
            except:
                continue
            corpus.append((title.split(" "), abstract.split(" ")))
    return corpus

def cal_word_tf_df(corpus):
    word_set = {}
    for doc in corpus:
        title, abstract = doc[0], doc[1]
        words = title + abstract
        for w in words:
            if w not in word_set:
                word_set[w] = Word(val=w, tf=1, df=0)
            else:
                word_set[w].tf += 1
        for w in set(words):
            word_set[w].df += 1
    return word_set


id_beg = 0
id_eos = 1
id_emp = 2
id_unk = 3
def build_idx_for_word_tf_df(word_set, tf_thres=12, df_thres=6, vocab_size=100000):
    
    
    start_idx = id_unk + 1
    word2idx = {}
    idx2word = {}
    
    word2idx["<eos>"] = id_eos
    word2idx["<unk>"] = id_unk
    word2idx["<emp>"] = id_emp
    word2idx["<beg>"] = id_beg
    
    word_list = list(filter(lambda w: w.tf > tf_thres and w.df > df_thres, word_set.values()))
    word2idx.update({w.val: start_idx + idx for idx, w in enumerate(word_list[:vocab_size])})
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return word2idx, idx2word


data_fname = "../data/processed.s2s.v1.txt"
corpus = read_data(data_fname)
print("Got {0} doc in corpus".format(len(corpus)))
word_set = cal_word_tf_df(corpus)
print("Got {0} unique word".format(len(word_set)))
top_tf_words = sorted(word_set.values(), key=lambda x: x.tf, reverse=True)
print("The Top 10 are: ")
print("\n".join(["{0}\t{1}\t{2}".format(word.val, word.tf, word.df) for word in top_tf_words[:10]]))

word2idx, idx2word = build_idx_for_word_tf_df(word_set)
titles = [[word2idx.get(w, id_unk) for w in doc[0]] for doc in corpus]
abstracts = [[word2idx.get(w, id_unk) for w in doc[1]] for doc in corpus]

X_train, X_test, Y_train, Y_test = train_test_split(abstracts, titles, test_size=1000)
print("X_train length: {0}\nX_test length: {1}\nY_train length: {2}\nY_test length: {3}".format(len(X_train),
                                                                                                len(X_test),
                                                                                                len(Y_train),
                                                                                                len(Y_test)
                                                                                                   ))
print(len(word2idx))

def prt(label, x):
    print(label + ":")
    for w in x:
        if w == id_emp: continue
        print(idx2word[w], end="")
    print()

data_fn = "../data/finance150.pkl"
with open(data_fn, 'wb') as f:
    pickle.dump((word2idx, idx2word, X_train, X_test, Y_train, Y_test), f, -1)
print("vocab size: {0}".format(len(word2idx)))
idx = random.randint(0, len(X_train))
prt("abstract", X_train[idx])
prt("title", Y_train[idx])
