from torchnlp.word_to_vector import GloVe
from torchnlp.word_to_vector import FastText
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
import glob
import random

from data_utils import read_data
from data_utils import DELIM_SLOT_TASK_1
from data_utils import make_file_with_predictions

torch.manual_seed(42)

N_INTENTS = 7
N_SLOTS = 72
EMB_DIM=300
HIDDEN_DIM=100
N_LAYERS=3
DROPOUT=0.2
LR = 0.1 # learning rate
TRAIN = "train"
DEV = "valid"
TEST = "test"
INPUT_SENTS = "input_sents"
OUTPUT_SLOTS = "output_slots"
OUTPUT_INTENTS = "output_intents"
EMBEDDED_SENTS = "embedded_sents"
SLOTS_INDS = "slots_inds"
INTENTS_INDS = "intent_inds"

class MyModel(nn.Module):

    def __init__(self, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, batch_1st=False, \
                 dr=DROPOUT, n_slots=N_SLOTS, n_intents=N_INTENTS):
        super(MyModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, batch_first=batch_1st, \
                            dropout=dr, bidirectional=True)
        self.hidden2slots = nn.Linear(2*hidden_dim, n_slots)
        self.hidden2intents = nn.Linear(2*hidden_dim, n_intents)

    def forward(self, sent_embeds, batch_size=1):
        sent_len = sent_embeds.shape[0] 
        lstm_out, (h_n, c_n) = self.lstm(sent_embeds)
        slots_space = self.hidden2slots(lstm_out.view(sent_len, -1))
        slots_scores = F.log_softmax(slots_space, dim=1)
        h_n = h_n.view(self.n_layers, 2, batch_size, self.hidden_dim)
        last_h = h_n[-1]
        # concatenate hidden states of the last biLSTM layer
        last_h = last_h.transpose(1, 0).contiguous().view(batch_size, -1)
        intents_space = self.hidden2intents(last_h)
        intents_scores = F.log_softmax(intents_space, dim=1)
        return slots_scores, intents_scores

def embed_sentences(sents, embs):
    '''Embeds all sentences with the given embeddings.

    Returns data in the format required for training.  
    '''
    out = []
    for sent in sents:
        tokens_emb = [embs[token] for token in sent]
        tokens_emb = torch.stack(tokens_emb)
        tokens_emb = tokens_emb.unsqueeze(1) # insert a batch dimension to get dim: [sent_len, batch_size, embedding_size]
        out.append(tokens_emb)
    return out

def get_labels2inds(target_data):
    '''Creates indices for labels in the given dataset.

    Target data has to be a list (or other container) with entries corresponding to sentences.
    Returns mappings between indices and labels.
    '''
    unique_labels = set()
    labels2inds = {}
    inds2labels = {}
    for sent in target_data:
        if not isinstance(sent, str):
            unique_labels.update(sent)
        else:
            unique_labels.add(sent)
    cnt = 0
    for label in sorted(unique_labels):
        labels2inds[label] = cnt
        inds2labels[cnt] = label
        cnt += 1
    return labels2inds, inds2labels

def get_labels_from_inds(inds, inds2labels):
    '''Transfers indices to corresponding labels.'''
    out = [inds2labels[ind] for ind in inds]
    return out

def prepare_targets(labels_all, labels2inds, dataset):
    '''Transfers labels to their indices and returns data in required format.'''
    out = []
    for lables_per_sent in labels_all:
        if not isinstance(lables_per_sent, str):
            ids = [labels2inds[label] for label in lables_per_sent]
        else:
            ids = [labels2inds[lables_per_sent]]
        if dataset == TRAIN:
            out.append(torch.tensor(ids, dtype=torch.long))
        else:
            out.extend(ids)
    return out

def flatten_slots(sent_seq):
    return [el for sent in sent_seq for el in sent]

def evaluate_model(model, embedded_sents, slots_true, intents_true):
    '''Evaluates the given model on a given dataset in terms of micro-averaged f1-score.

    Returns the score for the slots and itents as well as predicted indices for these labels.
    '''
    with torch.no_grad():
        pred_slots = []
        pred_intents = []
        for i in range(len(embedded_sents)):
            slots_scores, intents_scores = model(embedded_sents[i])
            _, slots_inds = torch.max(slots_scores, 1)
            pred_slots.append(slots_inds)
            _, intents_inds = torch.max(intents_scores, 1)
            pred_intents.append(intents_inds)
        pred_slots = torch.cat(pred_slots, dim=0)
        pred_intents = torch.cat(pred_intents, dim=0)
        f1_micro_slots = f1_score(slots_true, pred_slots, average='micro')
        f1_micro_intents = f1_score(intents_true, list(pred_intents), average='micro')
    return f1_micro_slots, f1_micro_intents, (pred_slots, pred_intents)

def run_experiment(data, emb, emb_name, inds2slots, inds2intents):
    '''Creates a neural model, trains and evaluates it with the given pre-trained embeddings and datasets,
    stores predictions for the test file. 
    '''
    print(emb_name)
    embedded_sents = embed_sentences(data[TRAIN][INPUT_SENTS], emb)
    slots_true = data[TRAIN][SLOTS_INDS]
    intents_true = data[TRAIN][INTENTS_INDS]
    embedded_sents_dev = embed_sentences(data[DEV][INPUT_SENTS], emb)
    assert len(embedded_sents) == len(slots_true) == len(intents_true)
    model = MyModel(EMB_DIM, HIDDEN_DIM, N_LAYERS, False, DROPOUT, N_SLOTS, N_INTENTS)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    loss_fnc1 = nn.NLLLoss()
    loss_fnc2 = nn.NLLLoss()

    random.seed(42)
    for epoch in range(10):
        print("EPOCH", epoch+1)
        #shuffle indices
        inds = list(range(len(embedded_sents)))
        random.shuffle(inds)
        for i in inds:
            model.zero_grad() # clear accumulated gradients
            slots_scores, intents_scores = model(embedded_sents[i]) # forward pass.
            #computing joint loss and updating parameters
            loss1 = loss_fnc1(slots_scores, slots_true[i])
            loss2 = loss_fnc2(intents_scores, intents_true[i])
            loss = loss1 + loss2 # different weights can be applied here
            loss.backward()
            optimizer.step()
        #print("epoch {}: train_loss {}".format(epoch+1, loss))
        #f1_slots_train, f1_intents_train = evaluate_model(model, embedded_sents, flat_slots_train, flat_intents_train)
        #print("train f1: slots", round(f1_slots_train, 3), "intents", f1_intents_train)
        f1_slots_dev, f1_intents_dev, _ = evaluate_model(model, embedded_sents_dev, data[DEV][SLOTS_INDS], data[DEV][INTENTS_INDS])
        print("dev f1: slots", round(f1_slots_dev, 3), "intents", round(f1_intents_dev, 3))
    embedded_sents_test = embed_sentences(data[TEST][INPUT_SENTS], emb)
    f1_slots_test, f1_intents_test, (slots, intents) = evaluate_model(model, embedded_sents_test, data[TEST][SLOTS_INDS], data[TEST][INTENTS_INDS])
    print("test f1: slots", round(f1_slots_test, 3), "intents", round(f1_intents_test, 3))
    slots = get_labels_from_inds(slots.numpy(), inds2slots)
    intents = get_labels_from_inds(intents.numpy(), inds2intents)
    filename = "task1_test_pred_"+emb_name
    make_file_with_predictions(filename, data[TEST][INPUT_SENTS], slots, intents, DELIM_SLOT_TASK_1)
    return model

def main():
    datasets = [TRAIN, DEV, TEST]
    # read data and create indices for slot and intent labels
    all_data = {}
    for ds in datasets:
        file = glob.glob("SNIPS/" +ds)
        input_sents, output_slots, output_intent = read_data(file, DELIM_SLOT_TASK_1)
        assert len(input_sents) == len(output_slots) == len(output_intent)
        all_data[ds] = {}
        all_data[ds][INPUT_SENTS] = input_sents
        all_data[ds][OUTPUT_SLOTS] = output_slots
        all_data[ds][OUTPUT_INTENTS] = output_intent
    slots2inds, inds2slots = get_labels2inds(all_data[TRAIN][OUTPUT_SLOTS])
    intents2inds, inds2intents = get_labels2inds(all_data[TRAIN][OUTPUT_INTENTS])
    for key in all_data:
        all_data[key][SLOTS_INDS] = prepare_targets(all_data[key][OUTPUT_SLOTS], slots2inds, key)
        all_data[key][INTENTS_INDS] = prepare_targets(all_data[key][OUTPUT_INTENTS], intents2inds, key)
    # load embeddings and run corresponding experiments
    glove_emb = GloVe(name='6B', dim=300) # get GloVe pre-trained embeddings
    run_experiment(all_data, glove_emb, "GloVe", inds2slots, inds2intents)
    fasttext_emb = FastText(language='en', aligned=True) # get FastText pre-trained embeddings
    run_experiment(all_data, fasttext_emb, "FastText", inds2slots, inds2intents)

if __name__ == "__main__":
    main()
