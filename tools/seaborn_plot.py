import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import os
import matplotlib.pyplot as plt


def hist_of_speed():
    # sns.set(style='white', font_scale=1.5, font='consolas')
    sns.set(font_scale=1, font='consolas')
    iris = np.load(os.path.join('..', 'carla_tools', 'velocity_list_v0.npy'))
    iris = np.asarray(iris)[20:-50]
    l2_distance_velocity = np.linalg.norm(iris, axis=1) # * 1.2
    # num_samples = iris.shape[0]
    # print('num_samples: ', num_samples)
    print('l2_distance_velocity.mean(): {}, l2_distance_velocity.std(): {}, l2_distance_velocity.median(): {}, '
          'l2_distance_velocity.max(): {}, l2_distance_velocity.min(): {}'
          .format(l2_distance_velocity.mean(), l2_distance_velocity.std(), np.median(l2_distance_velocity), l2_distance_velocity.max(), l2_distance_velocity.min()))
    print('l2_distance_velocity.shape: ', l2_distance_velocity.shape)  # iris.shape:  (600, 3)
    index = np.asarray(list(range(iris.shape[0])))
    data = np.concatenate((index[:, None], l2_distance_velocity[:, None], np.zeros_like(l2_distance_velocity[:, None])), axis=1)
    iris = pd.DataFrame(data=data,
                        columns=['timestep','velocity (m/s)', 'target']
                        )

    # sns.displot(data=iris, x='velocity (m/s)', kind='kde', fill=True,
    #             palette=sns.color_palette('bright')[:3], height=5, aspect=1.5) # hue='target',
    sns.histplot(data=iris, x="velocity (m/s)", binwidth=0.1)  # , color='red')
    plt.show()
    # sns.lineplot(x='timestep', y='velocity (m/s)', data=iris)
    # plt.show()

if __name__ == "__main__":
    hist_of_speed()