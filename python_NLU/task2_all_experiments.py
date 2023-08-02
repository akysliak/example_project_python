import glob
import torch
from torchnlp.word_to_vector import GloVe
from torchnlp.word_to_vector import FastText
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from collections import Counter

from data_utils import read_data
from data_utils import DELIM_SLOT_TASK_2

PAD = "#######" # padding symbol, for beginning and ends of sentences
UNIGRAM = "unigram"
BIGR_LEFT = "bigram-left"
BIGR_RIGHT = "bigram-right"
TRIGR = "trigram"
K_SPLITS = 10 # number of folds in a k-fold cross-validation
N_NEARESTN = 1 # number of nearest neighbours in knn

def get_flattened_no_pad(y):
    out = []
    for sent in y:
        out.extend(sent)
    return out

def get_flattened_with_pad(x):
    out = [PAD]
    for sent in x:
        out.extend(sent)
        out.append(PAD)
    return out

def get_x_unigram(x):
    return [[el] for sent in x for el in sent]

def get_x_trigr(x_flattened):
    return [x_flattened[i-1:i+2] for i in range(1, len(x_flattened)-1) if not x_flattened[i] == PAD]

def get_x_bi_left(x_flattened):
    return [x_flattened[i-1:i+1] for i in range(1, len(x_flattened)-1) if not x_flattened[i] == PAD]

def get_x_bi_right(x_flattened):
    return [x_flattened[i:i+2] for i in range(1, len(x_flattened)-1) if not x_flattened[i] == PAD]

def get_embedded_Ngram(ngrams, embedding_vectors):
    out = []
    for ngram in ngrams:
        embeddings = [embedding_vectors[token] for token in ngram]
        embed_ngram = torch.cat(embeddings, dim = 0)
        embed_ngram = embed_ngram.numpy()
        out.append(embed_ngram)
    return out

def prepare_data(filename):
    x_Ngrams = {}
    tokens, slots, intents = read_data(filename, DELIM_SLOT_TASK_2)
    y = get_flattened_no_pad(slots)
    x_fl = get_flattened_with_pad(tokens)
    x_Ngrams[UNIGRAM] = get_x_unigram(tokens)
    x_Ngrams[BIGR_LEFT] = get_x_bi_left(x_fl)
    x_Ngrams[BIGR_RIGHT] = get_x_bi_right(x_fl)
    x_Ngrams[TRIGR] = get_x_trigr(x_fl)
    return x_Ngrams, y, tokens, intents

def get_eval_results(pred, y_test):
    f1_macro = f1_score(y_test, pred, average='macro')
    f1_micro = f1_score(y_test, pred, average='micro')
    return f1_macro, f1_micro

def get_ngram_baseline_model(model_type, x_train, y_train, x_test, y_test):
    '''Creates and evaluates baseline model based on N-grams counts'''
    x_train = x_train[model_type]
    x_test = x_test[model_type]
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    print("Model type:", model_type)
    model = {}
    for i in range(len(x_train)):
        ngr = tuple(x_train[i])
        if ngr not in model.keys():
            model[ngr] = Counter()
        model[ngr].update([y_train[i]])
    pred = []
    for ngr in x_test:
        key = tuple(ngr)
        if key in model.keys():
            pred.append(model[key].most_common(1)[0][0])
        else:
            pred.append("O")
    f1_macro, f1_micro = get_eval_results(pred, y_test)
    print("f1 macro:", round(f1_macro, 2), "f1 micro:", round(f1_micro, 2))
    return model

def run_exp_baselines(x_train, y_train, x_test, y_test):
    print("BASELINES")
    print("Simple BASELINE models: no embedding used")
    model_uni = get_ngram_baseline_model(UNIGRAM, x_train, y_train, x_test, y_test)
    model_bi_l = get_ngram_baseline_model(BIGR_LEFT, x_train, y_train, x_test, y_test)
    model_bi_r = get_ngram_baseline_model(BIGR_RIGHT, x_train, y_train, x_test, y_test)
    model_trigr = get_ngram_baseline_model(TRIGR, x_train, y_train, x_test, y_test)
    print("Advanced BASELINE model: no embedding used, back-off")
    #make predictions for the test set with a combined model
    pred = []
    for i in range(len(y_test)):
        key_trigr = tuple(x_test[TRIGR][i])
        key_bi_l = tuple(x_test[BIGR_LEFT][i])
        key_bi_r = tuple(x_test[BIGR_RIGHT][i])
        key_uni = tuple(x_test[UNIGRAM][i])
        if key_trigr in model_trigr.keys():
            pred.append(model_trigr[key_trigr].most_common(1)[0][0])
        elif key_bi_l in model_bi_l.keys():
            pred.append(model_bi_l[key_bi_l].most_common(1)[0][0])
        elif key_bi_r in model_bi_r.keys():
            pred.append(model_bi_r[key_bi_r].most_common(1)[0][0])
        elif key_uni in model_uni.keys():
            pred.append(model_uni[key_uni].most_common(1)[0][0])
        else:
            pred.append("O")
    assert len(pred) == len(y_test)
    f1_macro, f1_micro = get_eval_results(pred, y_test)
    print("f1 macro:", round(f1_macro, 2), "f1 micro:", round(f1_micro, 2))
    print("---------------------------------------")

def run_exp_embeddings(emb_name, ngram_type, embedding_vectors, x_train, y_train, x_test, y_test, n_nearestN = N_NEARESTN):
    #print("Embedding", emb_name, ngram_type)
    x_train = get_embedded_Ngram(x_train, embedding_vectors)
    x_test = get_embedded_Ngram(x_test, embedding_vectors)
    knn = KNeighborsClassifier(n_neighbors=n_nearestN)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    return get_eval_results(pred, y_test), pred

def run_kfold_experiments(x_train_Ngr, y_train, emb_dict):
    print("{}-FOLD CROSS_VALIDATION".format(K_SPLITS))
    res = {}
    for key in x_train_Ngr:
        res[key] = {}
        for emb in emb_dict:
            res[key][emb] = {}
            res[key][emb]["f1_macro"] = 0
            res[key][emb]["f1_micro"] = 0
    kf = KFold(n_splits=K_SPLITS, shuffle=True, random_state=42)
    for train_inds, test_inds in kf.split(y_train):
        #print(len(train_inds), len(test_inds))
        y_tr = [y_train[i] for i in train_inds]
        y_te = [y_train[i] for i in test_inds]
        for key in x_train_Ngr.keys():
            x_tr = [x_train_Ngr[key][i] for i in train_inds]
            x_te = [x_train_Ngr[key][i] for i in test_inds]
            for emb in emb_dict:
                (f1_macro, f1_micro), _ = run_exp_embeddings(emb, key, emb_dict[emb], x_tr, y_tr,\
                                   x_te, y_te)
                res[key][emb]["f1_macro"] += f1_macro
                res[key][emb]["f1_micro"] += f1_micro
    for key in x_train_Ngr:
        for emb in emb_dict:
            res[key][emb]["f1_macro"] /= K_SPLITS
            res[key][emb]["f1_micro"] /= K_SPLITS
            print(key, emb, "f1_macro=", round(res[key][emb]["f1_macro"], 2), "f1_micro=", round(res[key][emb]["f1_micro"], 2))
    return res

def run_full_experiments(x_train_Ngr, y_train, x_test_Ngr, y_test, emb_dict):
    print("FULL EXPERIMETNS")
    for ngram in x_train_Ngr.keys():
        for emb, emb_vectors in emb_dict.items():
            (f1_macro, f1_micro), _ = run_exp_embeddings(emb, ngram, emb_vectors, x_train_Ngr[ngram], y_train, \
                                                    x_test_Ngr[ngram], y_test)
            print(ngram, emb, ": f1_macro = ", round(f1_macro, 2), "f1_micro = ", round(f1_micro, 2))

def run_experimnet_set(file_train, file_test, file_train_augm, emb_dict):
    x_train_Ngr, y_train, _, _ = prepare_data(file_train)
    x_test_Ngr, y_test, _, _ = prepare_data(file_test)
    run_kfold_experiments(x_train_Ngr, y_train, emb_dict)
    run_full_experiments(x_train_Ngr, y_train, x_test_Ngr, y_test, emb_dict)
    run_exp_baselines(x_train_Ngr, y_train, x_test_Ngr, y_test)
    print("--------EXPERIMENT WITH AUGMENTED DATA-------------")
    allowed_labels = ["O", "playlist", "artist", "music_item"]
    x_train_Ngr_augm, y_train_augm, _, _ = prepare_data(file_train_augm)
    allowed_indices = [i for i in range(len(y_train_augm)) if y_train_augm[i] in allowed_labels]
    print(len(allowed_indices), "more training examples")
    y_train.extend([y_train_augm[i] for i in allowed_indices])
    for key, ngrams in x_train_Ngr_augm.items():
        x_train_Ngr[key].extend([ngrams[i] for i in allowed_indices])
    run_kfold_experiments(x_train_Ngr, y_train, emb_dict)
    run_full_experiments(x_train_Ngr, y_train, x_test_Ngr, y_test, emb_dict)
    run_exp_baselines(x_train_Ngr, y_train, x_test_Ngr, y_test)

def main():
    glove_emb = GloVe(name='6B', dim=300) # get GloVe pre-trained embeddings
    fasttext_emb = FastText(language='en', aligned=True) # get FastText pre-trained embeddings
    emb_dict = {"GloVe" : glove_emb, "FastText" : fasttext_emb}
    file_test = glob.glob('LowResource/PlayMusic/test.txt')
    for size_type in ["twenty", "fifteen"]:
        file_train = glob.glob('LowResource/PlayMusic/{}_proto.txt'.format(size_type))
        file_train_augm = glob.glob('LowResource/AddToPlaylist/{}_proto.txt'.format(size_type))
        print("SETTING", size_type.capitalize(), ", knn with k =", N_NEARESTN)
        run_experimnet_set(file_train, file_test, file_train_augm, emb_dict)

if __name__ == "__main__":
    main()