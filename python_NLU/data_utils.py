from collections import Counter
import glob

DELIM_INTENT = "<=>"
DELIM_SLOT_TASK_1 =":"
DELIM_SLOT_TASK_2 ="||"

def print_dict(d, dictionary_name = ""):
    if not isinstance(d, dict):
        print("NOT A DICTIONARY")
        return
    print("printing DICTIONARY", dictionary_name)
    print("length=", len(d))
    for k, v in d.items():
        if isinstance(v, dict):
            print(k, ":", len(v), v)
        else:
            print(k, ":", v)

def update_dict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = Counter()
    dictionary[key].update(value)

def read_data(filenames, delimiter):
    input_tokens = []
    output_slots = []
    output_intent = []
    for filename in filenames:
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                # dealing with annotation inconsistency: "::" instead of ":" as separator
                if delimiter == DELIM_SLOT_TASK_1:
                    line = line.replace("::", ":")
                # separate input sentence from the intent label
                slots, intent = line.split(DELIM_INTENT)
                intent = intent.strip()
                output_intent.append(intent)
                # separate input tokens from their slot labels
                slots = slots.split()
                token_label_pairs = [[pair for pair in slot.rsplit(delimiter, 1)] for slot in slots] # rsplit() to account for annotations like "16:01:04:B-timeRange"
                tokens, slot_labels = zip(*token_label_pairs)
                input_tokens.append(tokens)
                output_slots.append(slot_labels)
    return input_tokens, output_slots, output_intent

def analyse_data(input_tkns, slot_lbls, intent_lbls):
    slots_cnt = Counter()
    intents_cnt = Counter()
    intents_with_slots = {}
    slots_with_intents = {}
    assert len(input_tkns) == len(slot_lbls), "len(input_tkns) != len(slot_lbls)"
    assert len(slot_lbls) == len(intent_lbls), "len(slot_lbls) != len(intent_lbls)"
    for i in range(len(input_tkns)):
        assert len(input_tkns[i]) == len(slot_lbls[i])
        intent = intent_lbls[i]
        intents_cnt.update([intent])
        update_dict(intents_with_slots, intent, slot_lbls[i])
        for slot_label in slot_lbls[i]:
            slots_cnt.update([slot_label])
            update_dict(slots_with_intents, slot_label, [intent])
    print("Intents:", len(intents_cnt), intents_cnt)
    print("Slots:", len(slots_cnt), slots_cnt)
    print_dict(intents_with_slots, "Intents")
    print_dict(slots_with_intents, "Slots")

def make_file_with_predictions(filename, all_tokens, all_slots, all_intents, delimiter):
    out = ""
    i = 0 # iterator for slots
    j = 0 # iterator for intents
    for sent in all_tokens:
        cur_line = []
        for token in sent:
            cur_line.append(delimiter.join([token, all_slots[i]]))
            i += 1
        out += " ".join(cur_line) + "<=>" + all_intents[j] + "\n"
        j += 1
    out = out.strip()
    with open(filename + ".txt", "w", encoding="utf8") as f:
        f.write(out)

def main():
    file = glob.glob("SNIPS/train")
    x, y_slots, y_intent = read_data(file, DELIM_SLOT_TASK_1)
    #assert len(x) == len(y_slots), "len(x) != len(y_slots)"
    #assert len(y_slots) == len(y_intent), "len(y_slots) != len(y_intent)"
    #for i in range(len(x)):
    #    assert len(x[i]) == len(y_slots[i])
    analyse_data(x, y_slots, y_intent)
    #files = glob.glob('LowResource/AddToPlaylist/fifteen_proto.txt') #('LowResource/*/fifteen_proto.txt')
    #x, y_slots, y_intent = read_data(files, DELIM_SLOT_TASK_2)
    #analyse_data(x, y_slots, y_intent)

if __name__ == "__main__":
    main()