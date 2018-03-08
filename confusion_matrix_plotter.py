import sys
sys.path.append('/home/myo/grip_detection/classification/training_dir')

import matplotlib.pyplot as plt
import numpy as np
import pickle
import networkCalculator
import os
import itertools
from sklearn.metrics import f1_score
import argparse

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,showColorMap=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    hfont = {'fontname':'Helvetica','size':16}

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,**hfont)
    if showColorMap:
        plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,**hfont)
    plt.yticks(tick_marks, classes,**hfont)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",**hfont)

    plt.tight_layout()
    plt.ylabel('True label',**hfont)
    plt.xlabel('Predicted label',**hfont)


parser = argparse.ArgumentParser("Startup training script")
parser.add_argument('-f','--file', help="name of file that contains the protocol",required=True)
parser.add_argument('-v','--verbose', action="store_true", help="verbosity")
args = parser.parse_args()

# Class names for combined case
class_names1 = ['empty','combined']

# Class names for more cases
#class_names2 = ['empty','handle','form','force']
class_names2 = ['empty','screwdriver','hammer','weightP']
#class_names2 = ['empty','force','form','reib']


# File name for file to plot
#outputF_name = "networkperformance_log_is[12]_fr[6]_fn100.p"
#outputF_name = "networkperformance_log_is[12]_fr[6]_fn100_intersample.p"
#outputF_name = 'iao_networkperformance_log_is[12]_fr[6]_fn100_intersample.p'
#outputF_name = 'iao_networkperformance_log_is[12]_fr[6]_fn100_interdata.p'

outputF_name = args.file

with open(outputF_name,'rb') as f:
    while True:
        try:
            data = pickle.load(f)

            if isinstance(data, str):
                raise ValueError

            print(data[0])
            print(data[1])
            aa1 = data[0]
            aa2 = data[1]
            trace = np.trace(aa2)
            total = np.sum(aa2)
            trace2 = np.trace(aa1)
            total2 = np.sum(aa1)
            acc2 = trace2/total2
            acc = trace/total
            print("Accuracy: {0:2.2%}".format(acc))
            config = data[2]
            print(config[0])
            print(config[1])

            plt.figure(figsize=(8,8))
            # Plot normalized confusion matrix for combined classes
            plot_confusion_matrix(aa1, classes=class_names1, normalize=True,title='Normalized confusion matrix, Acc:{0:2.2%}'.format(acc2),showColorMap=False)
            plt.savefig(os.path.join( *[p, date, "comb", "{0}-ks{1}-is{2}_comb".format(date,config[0], config[1])] )  ,bbox_inches="tight")

            plt.figure(figsize=(8,8))
            plot_confusion_matrix(aa2, classes=class_names2, normalize=True,title='Normalized confusion matrix, Acc:{0:2.2%}'.format(acc),showColorMap=False)
            plt.savefig(os.path.join( *[p, date, "full", "{0}-ks{1}-is{2}_full".format(date,config[0], config[1])] ) ,bbox_inches="tight")

        except EOFError:
            break

        except ValueError:
            date = data
            print("\n"+date+"\n")
            p, _ = os.path.split(os.path.realpath(__file__))
            os.makedirs(os.path.join(p,date),exist_ok=True)
            os.makedirs(os.path.join(p,date,"comb"),exist_ok=True)
            os.makedirs(os.path.join(p,date,"full"),exist_ok=True)

        except:
            date = "brokenDatestring"
            print("\n"+date+"\n")
            p, _ = os.path.split(os.path.realpath(__file__))
            os.makedirs(os.path.join(p,date),exist_ok=True)
            os.makedirs(os.path.join(p,date,"comb"),exist_ok=True)
            os.makedirs(os.path.join(p,date,"full"),exist_ok=True)
