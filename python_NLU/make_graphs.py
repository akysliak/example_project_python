import matplotlib.pyplot as plt

def make_lines(ys, x_ticks, legend, titel, x_label, ylabel, show_legend=True, show_ticks=True):
    palette = ["green", "red", "yellow", "blue", "violet"]
    num = 0
    for y in ys:
        plt.plot(x_ticks, y, marker='', color=palette[num], linewidth=1, alpha=0.9, \
             label=legend[num])
        num += 1
    if show_legend:
        plt.legend()
    plt.title(titel)
    plt.xticks(x_ticks)
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.show()

def make_plots_task2():
    x_ticks = [1, 5, 10, 20] # k in knn on the x-axis
    glove_twentyProto_trigr_no_augm_f1_micro = [0.77, 0.76, 0.75, 0.74]
    fasttext_twentyProto_trigr_no_augm_f1_micro = [0.77, 0.73, 0.71, 0.67]
    advanced_baseline = [0.72, 0.72, 0.72, 0.72]

    glove_twentyProto_trigr_with_augm_f1_micro = [0.8, 0.78, 0.73, 0.74]
    fasttext_twentyProto_trigr_with_augm_f1_micro = [0.75, 0.73, 0.72, 0.68]
    #adv_baseline = [0.72, 0.72, 0.72, 0.72] # with augmentation
    title = "Twenty_Proto Setting"
    x_label = "number of nearest neighbours"
    ylabel = "f1 micro"
    ys = [glove_twentyProto_trigr_no_augm_f1_micro, fasttext_twentyProto_trigr_no_augm_f1_micro, \
          glove_twentyProto_trigr_with_augm_f1_micro, fasttext_twentyProto_trigr_with_augm_f1_micro, advanced_baseline]
    legend = ["GloVe, no augmentation", "FastText, no augmentation", "GloVe, with augmentation", \
              "FastText, with augmentation", "baseline with back-off"]
    make_lines(ys, x_ticks, legend, title, x_label, ylabel)

    title = "Fifteen_Proto Setting"
    glove_15Proto_trigr_no_augm_f1_micro = [0.79, 0.76, 0.75, 0.72 ]
    fasttext_15Proto_trigr_no_augm_f1_micro = [0.78, 0.71, 0.7, 0.65]
    advanced_baseline = [0.69, 0.69, 0.69, 0.69]

    glove_15Proto_trigr_with_augm_f1_micro = [0.77, 0.72, 0.71, 0.73]
    fasttext_15Proto_trigr_with_augm_f1_micro = [0.73, 0.72, 0.71, 0.67]
    #adv_baseline = [0.69, 0.69, 0.69, 0.69] # with augmentation
    ys = [glove_15Proto_trigr_no_augm_f1_micro, fasttext_15Proto_trigr_no_augm_f1_micro, \
          glove_15Proto_trigr_with_augm_f1_micro, fasttext_15Proto_trigr_with_augm_f1_micro, advanced_baseline]
    legend = ["GloVe, no augmentation", "FastText, no augmentation", "GloVe, with augmentation", \
              "FastText, with augmentation", "baseline with back-off"]
    make_lines(ys, x_ticks, legend, title, x_label, ylabel)

def make_plots_task1():
    x_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # k in knn on the x-axis
    x_label = "epochs"
    ylabel = "f1 micro"
    legend = ["GloVe", "FastText"]
    title = "Learning curves: slots"
    GloVE_slots_f1_micro = [0.867, 0.91, 0.929, 0.941, 0.948, 0.942, 0.95, 0.943, 0.953, 0.948]
    FastText_slots_f1_micro = [0.76, 0.855, 0.883, 0.896, 0.917, 0.913, 0.927, 0.926, 0.928, 0.931]
    ys = [GloVE_slots_f1_micro, FastText_slots_f1_micro]
    make_lines(ys, x_ticks, legend, title, x_label, ylabel)
    title = "Learning curves: intents"
    GloVE_intents_f1_micro = [0.94, 0.976, 0.977, 0.983, 0.984, 0.977, 0.983, 0.977, 0.981, 0.983]
    FastText_intents_f1_micro = [0.861, 0.95, 0.964, 0.966, 0.969, 0.967, 0.981, 0.974, 0.971, 0.98]
    ys = [GloVE_intents_f1_micro, FastText_intents_f1_micro]
    make_lines(ys, x_ticks, legend, title, x_label, ylabel)

def main():
    make_plots_task1()
    make_plots_task2()

if __name__ == "__main__":
    main()



