import tensorflow as tf
import tqdm
import statistics
import collections
from modules.globals import env
from modules.train_step import trainStep
from modules.scores import showScores


def singleTrain(
        replay,
        policy: tf.keras.Model,
        target: tf.keras.Model,
        loss: tf.keras.losses,
        optimizer: tf.keras.optimizers,
        maxEpisodes: int,
        maxStepsPerEpisodes: int,
        numAction: int,
        gamma: float,
        epsilon: float,
        epsilonFinal: float,
        epsilonDecay: float,
        learnEpisodesMod: int = 5000):

    update = False
    actionEpsilon = False

    max_reward = 0

    episodes_reward: collections.deque = collections.deque(
        maxlen=1000)

    scores = []

    # Roda os episódios
    with tqdm.trange(maxEpisodes) as episodes:
        # episódio em episódioe
        for episode in episodes:
            # Inicia um novo game e obtem o estado inicial
            state = tf.constant(env.newGame(), dtype=tf.int64)

            # Roda um episódio e obtem a recompensa
            episode_reward, epsilon = trainStep(
                replay,
                policy,
                target,
                loss,
                gamma,
                numAction,
                optimizer,
                epsilon,
                epsilonFinal,
                epsilonDecay,
                state,
                update,
                actionEpsilon,
                maxStepsPerEpisodes)

            env.appendGameBoard()

            if episode % 10000 == 0 and episode > 0:
                # Salva o tabuleiro
                env.saveBoardGame()

                showScores(scores, name=env.config['plotPath'])

            # Verifica se é hora de atualizar a rede neural
            if episode % 300 == 0 and episode > 0:
                update = True
            else:
                update = False

            actionEpsilon = False
            if episode % 300 == 0 and episode > 0:
                actionEpsilon = True

            # Converte para inteiro
            episode_reward = float(episode_reward)

            # Insere a recompensa do episódio
            episodes_reward.append(episode_reward)

            # Calcula a média de recompensas
            running_reward = statistics.mean(episodes_reward)

            # Scores
            scores.append(sum(episodes_reward) / (len(episodes_reward) + 1.))

            episodes.set_description(f'Episode {episode}')

            if running_reward > max_reward and epsilonFinal >= epsilon:
                max_reward = running_reward

            # Exibe os dados
            episodes.set_postfix(
                episilon="{:.3f}".format(epsilon),
                episode_reward="{:.1f}".format(episode_reward),
                running_reward="{:.2f}".format(running_reward),
                max="{:.2f}".format(max_reward))

            if episode % 100000 == 0 and episode > 0:
                policy.save_weights("/content/files/policy.h5")
                target.save_weights("/content/files/target.h5")
