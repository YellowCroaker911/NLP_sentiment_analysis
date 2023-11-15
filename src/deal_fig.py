import matplotlib.pyplot as plt
import pickle


def save_fig(ax, model_name):
    with open(f'../picture/{model_name}.pkl', 'wb') as f:
        pickle.dump(ax, f)


def show_fig(model_name):
    with open(f'../picture/{model_name}.pkl', 'rb') as f:
        ax = pickle.load(f)
    plt.show()


if __name__ == '__main__':
    for model_name in ('BiRNN', 'TextCNN'):
        show_fig(model_name)
