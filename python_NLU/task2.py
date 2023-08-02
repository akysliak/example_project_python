import glob
from torchnlp.word_to_vector import GloVe
from torchnlp.word_to_vector import FastText

from task2_all_experiments import TRIGR
from task2_all_experiments import run_exp_embeddings
from task2_all_experiments import prepare_data
from task2_all_experiments import run_exp_baselines

from data_utils import make_file_with_predictions
from data_utils import DELIM_SLOT_TASK_2
N_NEARESTN = 1 # number of nearest neighbours in knn

def run_experiment(x_train_Ngr, y_train, x_test_Ngr, y_test, emb_dict, test_tokens, test_intents, \
                   ngrams=[TRIGR], augm=""):
    for ngram in ngrams:
        for emb, emb_vectors in emb_dict.items():
            (f1_macro, f1_micro), pred = run_exp_embeddings(emb, ngram, emb_vectors, x_train_Ngr[ngram], y_train, \
                                                    x_test_Ngr[ngram], y_test, n_nearestN = N_NEARESTN)
            print(ngram, emb, ": f1_macro = ", round(f1_macro, 2), "f1_micro = ", round(f1_micro, 2))
            filename = "task_2_test_pred_"+emb+"_"+augm
            make_file_with_predictions(filename, test_tokens, pred, test_intents, DELIM_SLOT_TASK_2)


def run_experimnet_set(file_train, file_test, file_train_augm, emb_dict):
    x_train_Ngr, y_train, _, _ = prepare_data(file_train)
    x_test_Ngr, y_test, test_tokens, test_intents = prepare_data(file_test)
    run_experiment(x_train_Ngr, y_train, x_test_Ngr, y_test, emb_dict, test_tokens, test_intents, [TRIGR], augm="no_augm")
    run_exp_baselines(x_train_Ngr, y_train, x_test_Ngr, y_test)
    print("--------EXPERIMENT WITH AUGMENTED DATA-------------")
    allowed_labels = ["O", "playlist", "artist", "music_item"]
    x_train_Ngr_augm, y_train_augm, _, _ = prepare_data(file_train_augm)
    allowed_indices = [i for i in range(len(y_train_augm)) if y_train_augm[i] in allowed_labels]
    print(len(allowed_indices), "more training examples")
    y_train.extend([y_train_augm[i] for i in allowed_indices])
    for key, ngrams in x_train_Ngr_augm.items():
        x_train_Ngr[key].extend([ngrams[i] for i in allowed_indices])
    run_experiment(x_train_Ngr, y_train, x_test_Ngr, y_test, emb_dict, test_tokens, test_intents, [TRIGR], augm="with_augm")
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