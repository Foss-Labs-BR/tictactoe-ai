import matplotlib.pyplot as plt


def showScores(scores, name, color='C1', figure=0, save=True):
    plt.figure(figure)
    plt.clf()
    plt.title('Learning TicTacToe')
    plt.xlabel('Sessions')
    plt.ylabel('Mean Rewards')
    plt.plot(scores, color=color)
    # plt.pause(0.001)
    if save:
        plt.savefig(name)
