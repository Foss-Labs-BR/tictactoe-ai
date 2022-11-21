import tensorflow as tf
from modules.episode_versus import runEpisodeVersus
from modules.optimize_model import optimizeModel

@tf.function
def trainStep(
        replay,
        policy: tf.keras.Model,
        target: tf.keras.Model,
        loss: tf.keras.losses,
        gamma: tf.Tensor,
        numAction: tf.Tensor,
        optimizer: tf.keras.optimizers,
        epsilon: tf.Tensor,
        epsilonFinal: tf.Tensor,
        epsilonDecay: tf.Tensor,
        state: tf.Tensor,
        update: tf.Tensor,
        action_episilon: tf.Tensor,
        maxSteps: int):

    rewards = runEpisodeVersus(
        replay, policy, epsilon, epsilonFinal, state, maxSteps)

    if tf.cast(update, tf.bool):
        epsilon = optimizeModel(
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
            action_episilon,
            30)

    # Soma as recompensas
    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward, epsilon
