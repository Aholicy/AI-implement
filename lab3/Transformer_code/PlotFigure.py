import matplotlib.pyplot as plt
import pickle
from datetime import datetime

def PlotFigure(result, use_save=False):
    train_loss = result['train loss']
    # test_loss = result['test loss']
    # train_acc = result['train acc']

    test_acc = result['test acc']

    fig = plt.figure(1)
    # 字体
    font = {'family' : 'serif', 'color': 'black', 'weight': 'bold', 'size': 16,}

    ax1 = fig.add_subplot(111)
    ln1 = ax1.plot(train_loss, 'r', label='Training Loss')
    ax2 = ax1.twinx()
    ln2 = ax2.plot(test_acc, 'k--', label='Testing Accuracy')

    lns = ln1 + ln2

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=7)

    ax1.set_ylabel('Loss', fontdict=font)
    ax1.set_title("Text Classification", fontdict=font)
    ax1.set_xlabel('Epoch', fontdict=font)

    ax2.set_ylabel('Accuracy', fontdict=font)

    plt.show()
    if use_save:
        figname = 'figure/LSTM_classifier_' + datetime.now().strftime("%d-%H-%M-%S") + '.png'
        fig.savefig(figname)
        print('Figure %s is saved.' % figname)

