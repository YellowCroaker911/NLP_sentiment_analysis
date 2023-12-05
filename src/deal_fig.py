import matplotlib.pyplot as plt
import pickle

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

def save_fig(ax, model_name):
    with open(f'../picture/{model_name}.pkl', 'wb') as f:
        pickle.dump(ax, f)


def show_fig(model_name):
    with open(f'../picture/{model_name}.pkl', 'rb') as f:
        ax = pickle.load(f)
    plt.show()


if __name__ == '__main__':
    # for model_name in ('BiRNN', 'TextCNN'):
    #     show_fig(model_name)
    # show_fig('BiRNN')
    show_fig('TextCNN')
