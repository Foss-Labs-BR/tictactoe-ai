import tensorflow as tf
import numpy as np
from typing import Tuple, List
from modules.globals import env

def getSignal() -> tf.Tensor:
    return tf.numpy_function(getValidSignal, [], tf.int64)


def getValidSignal() -> Tuple[np.ndarray]:
    actions = env.onlyValidSignal()
    np.random.shuffle(actions)
    action = actions[0]
    return (np.array(action, np.int64))


def tfEnvStepVersus(
        action: tf.Tensor,
        player: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(envStepVersus, [action, player],
                             [tf.int64, tf.float32, tf.bool])


def envStepVersus(
        action: np.ndarray,
        player: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    state, reward, done = env.step(action, player)

    return (state.astype(np.int64),
            np.array(reward, np.float32),
            np.array(done, np.bool))
