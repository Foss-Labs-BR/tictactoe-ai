from typing import Tuple
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import tensorflow as tf

class Replay:
    def __init__(self,
                 batchSize: int,
                 maxLength: int,
                 specs: Tuple,
                 itemsQtd=5):
        """
        Inicia a classe do replay
        Arguments:
            Int batchSize:   Tamanho do batch
            Int maxLength:   Tamanho total do buffer
            Tuple specs:     Tupla de especificação do conteúdo do buffer
            Int   itemsQtd:  Quantidade de itens no batch
        Returns:
            Void
        """

        # Seta o tamanho do batch
        self.batchSize = batchSize

        # Seta a quantidade de itens por batch
        self.itemsQtd = itemsQtd

        # Cria o buffer
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            specs,
            batch_size=batchSize,
            max_length=maxLength
        )

    def add(self,
            state: tf.Tensor,
            nextState: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            done: tf.Tensor,
            standardize: bool = True):
        """
        Adiciona batchs ao buffer
        Arguments:
            Tensor state:     Estado inicial (Estado antes da execução da ação)
            Tensor nextState: Próximo estado (Estado após a execução da ação)
            Tensor action:    Ação executada
            Tensor reward:    Recompensa recebida
            Tensor done:      Flag de continuação/termino do game
        Returns:
            Void
        """

        # Gera a tupla dos tensores
        values = (state, nextState, action, reward, done)

        # Monta o batch
        values_batched = tf.nest.map_structure(
            lambda t: tf.stack([t] * self.batchSize), values)

        # Adiciona o batch
        self.buffer.add_batch(values_batched)

    def get(self,
            sampleBatchSize: int):
        """
        Obtem uma determinada quantidade amostras do buffer
        Arguments:
            Int sampleBatchSize: Tamanho do batch
        Returns:
            Void
        """

        # Obtem o batch em tf.data.dataset
        dataset = self.buffer.as_dataset(
            single_deterministic_pass=False,
            sample_batch_size=sampleBatchSize,
            num_steps=1)

        # Itera os itens
        iterator = iter(dataset)

        # Data a ser retornada
        data = []

        # Obtem os dados
        for _ in range(self.itemsQtd):

            # Obtem o dado
            d, _ = next(iterator)

            # Seta na lista
            data.append(d)

        # Retorna os dados
        return data

    def getData(self,
                sampleBatchSize: int,
                numSteps=1):
        """
        Obtem uma determinada quantidade amostras do buffer e separa por:
        states, nextStates, actions, rewards, dones
        Arguments:
            Int sampleBatchSize: Tamanho do batch
            Int numSteps: Número de steps a retornar
        Returns:
            Void
        """
        # Obtem o batch em tf.data.dataset
        dataset = self.buffer.as_dataset(
            single_deterministic_pass=False,
            sample_batch_size=sampleBatchSize,
            num_steps=numSteps)

        # Itera os itens
        iterator = iter(dataset)

        # Dados a serem retornados
        states, nextStates, actions, rewards, dones = [], [], [], [], []

        # Obtem os dados
        for _ in range(self.itemsQtd):

            # Obtem o dado
            data, _ = next(iterator)

            # Obtem os dados
            state, nextState, action, reward, done = data

            # Itera e seta os dados
            for s, n, a, r, d in zip(state, nextState, action, reward, done):
                states.append(s)
                nextStates.append(n)
                actions.append(a)
                rewards.append(r)
                dones.append(d)

        # Retorna os dados
        return states, nextStates, actions, rewards, dones
