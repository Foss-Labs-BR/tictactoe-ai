from datetime import datetime

from classes.tic_tac_toe import TicTacToe

# Configuração de recompensa para o ambiente
config = {
    'stateImage': True,
    'stateImagePath': '/content/logs/out.png',
    'font': '/content/fonts/VeraMono.ttf',
    'pathBoardFile': "/content/logs/game.txt",
    'plotPath': '/content/logs/brain{}.png'.format(datetime.now().strftime("%d%m%Y%H%M%S")),
    "rewardPositive": 9,
    "rewardNegative": -9,
    "rewardEachStep": 0,  # -0.001,
    "rewardErrorStep": -0.01,
    "rewardInvalidStep": -1,
    "rewardDraw": 0  # -0.1
}

# Instancia o jogo
env = TicTacToe(config)
