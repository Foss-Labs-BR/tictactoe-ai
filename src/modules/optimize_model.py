import tensorflow as tf

@tf.function
def optimizeModel(
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
        action_episilon: tf.Tensor,
        sampleBatchSize: int):
    """
    Atualiza o modelo
    Arguments:
        int sampleBatchSize - Quantidade amostrar a treinar
    Returns:
        Void
    """

    # Obtem as amostras
    data = replay.get(sampleBatchSize)

    # Itera as amostras
    for states, nextStates, actions, rewards, dones in data:
        for i in tf.range(sampleBatchSize):
            state = states[i]
            nextState = nextStates[i]
            action = actions[i]
            reward = rewards[i]
            done = dones[i]

            # Usando o modelo alvo, processa o próximo estado
            nextStateActions = target(nextState)

            # Obtem as probabilidades das ações
            na_probs = tf.nn.softmax(nextStateActions)

            # Obtem a ação com maior probabilidade
            q_s_a_prime = tf.argmax(na_probs, axis=1)

            # Converte o tensor da ação para float32
            q_s_a_prime = tf.cast(q_s_a_prime, dtype=tf.float32)

            """
            Faz a verificação:
                Se for um estado terminal (done == true)
                    A recompensa é a que existe hoje (Segundo parametro da tf.where)
                Se não for um estado terminal (done == false)
                    Aplica o fator de desconto (gamma), multiplicado pela maior probabilidade encontrada para a ação,
                    depois, soma com a recompensa obtida com a execução da ação atual
            """
            q_s_a_target = tf.where(done, reward, reward+(gamma*q_s_a_prime))

            # Cria a fita do gradiente
            with tf.GradientTape() as tape:

                # Com o modelo da politica, processamos o estado atual
                q_s_a = policy(state)

                # Multiplica todas as distribuições
                q_s_a = q_s_a * tf.one_hot(action, numAction)

                # Faz a soma dos tensores
                q_s_a = tf.math.reduce_sum(q_s_a)

                _loss = loss(q_s_a_target, q_s_a)

                '''
                # Calcula o custo e remove as dimensões do tensor
                loss = tf.square( q_s_a_target - q_s_a )

                # Faz a soma dos custos
                loss = tf.math.reduce_mean( loss )
                '''

            # Gera os parametros do gradiente usando os pesos do modelo de politica
            gradients = tape.gradient(_loss, policy.trainable_variables)

            # Aplica os gradientes para otimizar a descida do gradiente
            optimizer.apply_gradients(
                zip(gradients, policy.trainable_variables))

        # Verifica os valors de episilon
        if tf.cast(action_episilon, tf.bool) and tf.math.greater(epsilon, epsilonFinal):

            # Se for menos, aplica uma multiplicador de episilon pelo valor de decaimento
            epsilon = tf.multiply(epsilon, epsilonDecay)

        return epsilon
