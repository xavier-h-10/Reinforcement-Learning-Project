import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import json

# path = "BoxingNoFrameskip-v4_reward.json"
# name = "BoxingNoFrameskip-v4"
path = "PongNoFrameskip-v4_reward.json"
name = "PongNoFrameskip-v4"


def draw(score, title, ylabel, smooth=True):
    plt.xlabel("Training Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.plot(score, marker='o',
    #          mfc="white", ms=3)
    score = gaussian_filter1d(score, sigma=1)
    # plt.plot(score, marker='o', markerfacecolor='white')
    plt.plot(score)
    plt.savefig(name + ".jpg", dpi=1000, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    with open(path, "r") as f:
        data = json.load(f)

    data = np.array(data)
    score = data[:, 2]
    score1 = []
    for i in range(len(score)):
        score1.append(np.mean(score[0:i + 1]))

    draw(score, "Reward on " + name, "Reward per Episode", smooth=False)
    # draw(score1, "Average Reward on " + name, "Average Reward per Episode")

    print("max_reward=", np.max(score))
