import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

'''
This script needs two files ready.

test_corrlabel: File that constis of the correct label for each line in the test set
results.csv: Comma seperated file where each column contains values predicted by coresponding model.

These two commands can be used to gernerate the files in each fold:
awk '{print substr($1,2,3)}' <path testfold> test_corrlabel
paste *.conc.pred -d',' > results.csv
'''


def eval(labelfile, predictionfile):
    # Setup variable for stats
    bigcm = 0
    bigmiss = 0
    totalSamples = 0

    # Loop over all folds and loss functions
    for foldnr in [0, 1, 2, 3]:
        print("Fold: " + str(foldnr))
        filepath = "INSERT BASE PATH" + "fold" + \
                   str(foldnr) + "/"
        y_true = genfromtxt(filepath + labelfile, delimiter=',')
        y_pred = genfromtxt(filepath + predictionfile, delimiter=',')

        # Evaluate predicted class by choose class with max probability
        mmax = np.argmax(y_pred, axis=1)

        # Count missclassification
        missclass = sum(y_true != (mmax + 1))
        print("Number of missclassification: ", missclass)

        # Create confusion matrix
        cm = confusion_matrix(y_true, mmax + 1)
        print(cm)

        # Aggregate results over folds
        bigcm = bigcm + cm
        bigmiss = bigmiss + missclass

        # Zero index true labels
        y_true = y_true - 1

        n_samples = y_pred.shape[0]

        print("ACC: ", (n_samples - missclass) / n_samples)
        print()
        print("Datapoints in fold: ", n_samples)
        print()
        totalSamples += n_samples

    print("Total datapoint: ", totalSamples)
    print(bigcm)
    print("total ACC: ", (totalSamples - bigmiss) / totalSamples)
    print()


def plot(foldnr):
    # Plot of Number of nodes explored vs iterations
    # Plot of loss vs iterations

    plt.rcParams['figure.figsize'] = (10, 13)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(7, 7))

    filepath = "INSERT BASE PATH" + "fold" + \
               str(foldnr) + "/"
    fig, axes = plt.subplots(nrows=5, ncols=2)
    fig.tight_layout()
    fig2, axes2 = plt.subplots(nrows=5, ncols=2)
    fig2.tight_layout()
    classnr = 1
    df = {}
    for classnr in range(1, 10):
        df[classnr] = pd.read_csv(filepath + "MMC_" + str(foldnr) + "_" + str(classnr) + ".itrStats.csv",
                                skiprows=1, header=None,
                                names=['iteration', 'numFeatures', 'rewritten', 'prone', 'total', 'step', 'motif', 'lossvalue', 'regloss', 'convrate'])
        markerstep = 10
        print(classnr)
        (df[classnr]['lossvalue'] / df[classnr]['lossvalue'].max()).plot(label=classnr,
                                                                         use_index=False,
                                                                         logy=1,
                                                                         ax=axes[(classnr - 1) // 2,
                                                                                 (classnr - 1) % 2],
                                                                         title="Class " +
                                                                         str(classnr),
                                                                         linewidth=2.0,
                                                                         markevery=slice(
                                                                             0, -1, markerstep),
                                                                         markerfacecolor='none')
        pllt = (df[classnr]['total'].plot(label=classnr,
                                          use_index=False,
                                          ax=axes2[(classnr - 1) // 2,
                                                   (classnr - 1) % 2],
                                          title="Class " + str(classnr),
                                          markevery=slice(0, -1, markerstep),
                                          alpha=1))
        axes2[(classnr - 1) // 2, (classnr - 1) % 2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    axes[4, 0].set_xlabel('Iterations')
    axes[3, 1].set_xlabel('Iterations')
    axes[2, 0].set_ylabel('Loss function value')
    axes2[4, 0].set_xlabel('Iterations')
    axes2[3, 1].set_xlabel('Iterations')
    axes2[2, 0].set_ylabel('Nodes explored')
    axes[-1, -1].axis('off')
    axes2[-1, -1].axis('off')
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig("fig1.png")
    fig2.savefig("fig2.png")

eval("test_corrlabel", "results.csv")
plot(0)
