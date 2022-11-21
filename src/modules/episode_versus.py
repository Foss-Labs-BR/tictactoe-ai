import tensorflow as tf
from modules.step_functions import getSignal, tfEnvStepVersus

def runEpisodeVersus(
        replay,
        policy: tf.keras.Model,
        epsilon: tf.Tensor,
        epsilonFinal: tf.Tensor,
        state: tf.Tensor,
        maxSteps: int):

    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    playerO = tf.constant('O', dtype=tf.string)
    playerX = tf.constant('X', dtype=tf.string)

    played = tf.constant('O', dtype=tf.string)

    # Guarda o shape inicial do estado inicial
    initialStateShape = state.shape

    #print( initialStateShape )
    # sys.exit()

    # Hora de processar cada jogada,
    # levando em consideração o número máximo de jogadas permitidas por episódio
    for T in tf.range(maxSteps):

        # Convert o tensor para 1 dimensão
        state = tf.expand_dims(state, 0)

        if tf.math.equal(played, playerX):
            played = playerO

            action = getSignal()
            action = tf.reshape(action, [])
        else:
            played = playerX

            # Seleciona uma ação
            if tf.math.greater(epsilon, epsilonFinal):
                # Seleciona uma ação randomica
                #action  =  tf.random.uniform(shape=(), minval=1, maxval=9, dtype=tf.int64)
                action = getSignal()
                action = tf.reshape(action, [])
            else:
                # Obtem a probabilidade das ações
                actions = policy(state)

                # Seleciona uma ação com base na distribuição
                action = tf.random.categorical(actions, 1)[0, 0]

        # Realiza a ação no ambiente
        nextState, reward, done = tfEnvStepVersus(action, played)

        if tf.math.equal(played, playerX):
            # Atualiza o rewards
            rewards = rewards.write(T, reward)

            nextState = tf.reshape(nextState, [40, 50, 2])
            reward = tf.reshape(reward, [1])
            done = tf.reshape(done, [1])

            action = tf.expand_dims(action, 0)
            state = tf.squeeze(state)

            replay.add(state, nextState, action, reward, done)

        nextState = tf.expand_dims(nextState, 0)

        # Set initial state shape
        nextState = tf.squeeze(nextState)
        state = nextState

        # Seta o shape inicial do estado
        state.set_shape(initialStateShape)

        # Verifica se o jogo terminou
        if tf.cast(done, tf.bool):
            break

    # Empilha os tensores
    rewards = rewards.stack()

    # Retorna
    return rewards
