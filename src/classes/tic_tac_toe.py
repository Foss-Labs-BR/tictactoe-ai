import os
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

class TicTacToe:
    def __init__(self, config):
        """
        Inicia a classe do game
        Arguments:
            Array config: Configurações de recompensas
        Returns:
            Void
        """

        # Configurações do game
        self.config = config

        # Matriz da partida
        self.game = []

        # Usuário que está jogando
        self.player = ''

        # Estado do jogo (False - Partida terminada)
        self.done = True

        # Jogada anterior
        self.lastAction = None

        # Quantas jogadas foram realizadas na partida
        self.played = 0

        self.board = ''

        # Represeta os jogadores na partida
        self.pX = 1
        self.pO = 2

    def getRewardLossGame(self):
        return self.config['rewardPositive']

    def newGame(self, player='X'):
        """
        Inicia uma nova partida
        Arguments:
            String player: Usuário que inicia a partida
        Returns:
            Void
        """
        self.lastAction = -1
        self.done = False
        self.player = player
        self.game = np.zeros(9).astype(int)
        #self.game    =  np.array([1, 2, 1, 2, 2, 1, 1, 1, 2])
        self.played = 0

        return self.getObservable()

    def onlyValidSignal(self):
        """
        Retorna somente os locais válidos para jogar
        Returns:
            Array
        """

        # Lista com os itens válidos
        valid = []

        # Contador da lista
        j = 0

        # Loop na matriz do jogo
        for i in self.game:

            # Verifica as posições validas
            if i == 0:

                # Coloca o item na lista
                valid.append(j+1)

            # Incrementa o contador
            j += 1

        # Retorna  a lista dos itens válidos
        return valid

    def getGameStatus(self):
        if self.played == 9:
            self.done = True
        return self.done

    def checkWinner(self, player):
        """
        Verifica se a partida tem um vencedor
        Arguments:
            String player: Usuário que será verificado
        Returns:
            Boolean
        """

        # Obtem o valor do player repsentado na matriz
        p = self.pX if player == 'X' else self.pO

        # Seta a partida como terminada
        self.done = True

        # Verifica se houve um vencedor
        if self.game[0] == p and self.game[1] == p and self.game[2] == p:
            return True
        elif self.game[3] == p and self.game[4] == p and self.game[5] == p:
            return True
        elif self.game[6] == p and self.game[7] == p and self.game[8] == p:
            return True
        elif self.game[0] == p and self.game[3] == p and self.game[6] == p:
            return True
        elif self.game[1] == p and self.game[4] == p and self.game[7] == p:
            return True
        elif self.game[2] == p and self.game[5] == p and self.game[8] == p:
            return True
        elif self.game[0] == p and self.game[4] == p and self.game[8] == p:
            return True
        elif self.game[2] == p and self.game[4] == p and self.game[6] == p:
            return True

        # Não venceu, então o jogo continua
        self.done = False

        # Retorna o boolean, informando que o jogo nao terminou
        return False

    def checkPlayerWinner(self):
        """
        Verifica se a partida tem um vencedor
        Returns:
            Integer
        """

        if self.checkWinner('X'):
            # X venceu
            return [self.getObservable(), self.config['rewardPositive'], self.done]
        elif self.checkWinner('O'):
            # O venceu
            return [self.getObservable(), self.config['rewardPositive'], self.done]
        elif self.getGameStatus():
            # Empate
            return [self.getObservable(), self.config['rewardDraw'], self.done]

        # Ninguém venceu e o jogo não terminou
        return [self.getObservable(), self.config['rewardEachStep'], self.done]

    def getObservable(self):
        """
        Retorna a matriz do game com a jogada anterior
        Returns:
            Array
        """

        if self.config['stateImage']:
            return self.getStateImage()

        # Seta o estado como sendo a matriz do game
        state = self.game

        # Adiciona a jogada anterior
        #state  = np.append(state, self.lastAction)

        # Retorna o estado
        return state

    def appendGameBoard(self):
        self.board += self.showCurrentGame()+"\n"

    def showCurrentGame(self):
        currentGame = ''
        separator = ''
        for arr in self.game.reshape(3, 3):
            currentGame += '\n'
            j = 0
            for i in arr:
                separator = ' | ' if j < 2 else ''
                if i == self.pX:
                    currentGame += 'X'+separator
                elif i == self.pO:
                    currentGame += 'O'+separator
                else:
                    currentGame += ' '+separator
                j += 1
        return currentGame

    def saveBoardGame(self, append=True):
        if append:
            f = open(self.config['pathBoardFile'], 'a')
        else:
            f = open(self.config['pathBoardFile'], 'w')
        f.write(self.board)
        f.write("\n")
        f.close()
        self.board = ''

    def deleteBoardFile(self):
        # Verifica se o arquivo do jogo existe
        if os.path.exists(self.config['pathBoardFile']):
            # Apaga o arquivo
            os.remove(self.config['pathBoardFile'])

    def getStateImage(self):
        board = self.showCurrentGame()
        self.generateImage(board)
        return self.getImage(self.config['stateImagePath'])

    def generateImage(self, board):
        img = Image.new('RGB', (50, 40))
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype(self.config['font'], 8)

        d.text((3, -10), board, fill=(255, 255, 255), font=font)
        #img.save(self.config['stateImagePath'], "JPEG", quality=100)
        img.convert('LA').save(self.config['stateImagePath'], 'png')

    def getImage(self, path):
        img = Image.open(path)
        img.load()
        data = np.asarray(img, dtype="int32")
        return data

    def step(self, index: int, player='X'):
        """
        Faz as jogadas na partida
        Arguments:
            Integer index: Usuário que será verificado
            String  player: Usuário que será verificado
        Returns:
            Array
        """

        # Faz a troca para a resentação do player na matriz
        player0 = player

        player = self.pX if player == b'X' else self.pO

        # Ultima ação
        self.lastAction = index

        # Verifica se a jogada está dentro do intervalor 0-8
        if (index-1) > len(self.game):
            # Valor inválido
            return [self.getObservable(), self.config['rewardErrorStep'], self.done]
        elif self.game[index-1] == 0:

            # Atualiza a quantidade de jogada
            self.played += 1

            # Faz a jogada na posição da matriz
            self.game[index-1] = player

            # Retorna os feedback
            return self.checkPlayerWinner()

        elif self.getGameStatus():

            # Retorna os feedback
            return self.checkPlayerWinner()

        # Valor inválido
        return [self.getObservable(), self.config['rewardInvalidStep'], self.done]
