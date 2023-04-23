import os
from os.path import join as pjoin
import dw.eval as dwe
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from yaml import load, Loader
from dw.eval import get_confusion_metrics as gcm


def try_epoch_to_int(s):
    try:
        i = int(s.split('_')[-1].split('.')[0])
    except ValueError:
        i = 0
    return i


def plot_results_nrw(names, test=True, train=True, val=True, tdom=False):
    """Example to show training curves for different experiment runs.

    Experiment hyperparameters can be extracted from the stored yaml file.
    :param names: A list of experiment names.
    :param test: If True, plot test results.
    :param train: If True, plot training results.
    :param val: If True, plot validation results.
    :param tdom: If True, plot target domain results.
    """

    data = pd.DataFrame()

    class_names = ["BU", "FO", "RD", "AG", "UR", "GL", "HW"]
    cases = []
    if test: cases.append('TESTING')
    if train: cases.append('TRAINING')
    if val: cases.append('VALIDATION')
    if tdom: cases.append('TARGET_DOM_VAL')

    # ---------------------------------------------------------------------------------------- Prepare data
    for name in names:
        root = pjoin(name, 'confusion_matrices/')
        if not os.path.exists(root):
            print(f'Path {root} does not exist')
            continue
        n = max([try_epoch_to_int(txt) for txt in os.listdir(root)]) + 1
        config_file = sorted([f for f in os.listdir(name) if f.endswith('.yaml')])[-1]
        with open(pjoin(name, config_file), 'r') as file:
            yaml = load(file.read(), Loader)

        exp_attributes = {
            "BTSZ": f"{yaml['TRAIN']['BTSZ']}",
            "LR": f"{yaml['TRAIN']['SEG']['LR']}",
            "BTSZ/LR": f"{yaml['TRAIN']['BTSZ']} / {yaml['TRAIN']['SEG']['LR']}",
        }

        for subset in cases:
            for i in range(1, n):
                try:
                    M = gcm(np.load(root + f'CM_{subset}_{i}.npy'))
                except IOError:
                    continue

                data = pd.concat((data, pd.DataFrame(exp_attributes |
                                                     {"Subset": subset, "ep": i, "Class": "glob", "OA": M['oa'],
                                                      "mF1": M['mf1']}, index=[0])), ignore_index=True)
                for c in range(len(M['f1s'])):
                    data = pd.concat(
                        (data, pd.DataFrame(exp_attributes | {"Subset": subset, "ep": i, "F1": M['f1s'][c] * 100,
                                                              "Class": class_names[c]}, index=[0])), ignore_index=True)

    # ----------------------------------------------------------------------------------------- Plot data
    data = data.loc[(data['Class'] == "glob")]  # & (data['Class'] != "CL")]
    sns.set_theme()
    sns.set_context("paper")
    sns.relplot(x="ep", y="mF1", data=data, kind="line", row="Subset", col="BTSZ", hue="LR",
                legend="full", facet_kws={"sharex": True, "sharey": True})
    plt.show()


def print_cm(path: str, print_normalized_cms=False):
    """Load a confusion matrix and present several class-wise and global metrics.

    :param path: Path to the confusion matrix to load.
    :param print_normalized_cms: Whether to print normalized variants.
    """
    print('Loading confusion matrix from', path)

    cm = np.load(path)
    dwe.print_metrics(cm)

    if print_normalized_cms:
        np.set_printoptions(precision=2, suppress=True)

        print('\n---------- P\\R (normalised over predictions)\n')
        nCM = (100.0 * cm) / np.sum(cm, axis=1, keepdims=True)
        print(nCM)

        print('\n---------- P\\R (normalised over reference)\n')
        nCM = (100.0 * cm) / np.sum(cm, axis=0, keepdims=True)
        print(nCM)

    return cm


if __name__ == "__main__":
    # print_cm('../runs/source_training/bochum/eval_heinsberg/confusion_matrices/CM_TESTING.npy', False)
    variant_root = '../runs/tuning_example/variants/'
    plot_results_nrw([pjoin(variant_root,f) for f in os.listdir(variant_root) if os.path.isdir(pjoin(variant_root,f))], test=False, train=True, val=True, tdom=False)
