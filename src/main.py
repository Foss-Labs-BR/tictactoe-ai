import tensorflow as tf
from tensorflow import keras
from classes.replay import Replay
from modules.weights import copyWeights
from modules.single_train import singleTrain
from modules.globals import env

# Instancia o jogo
env.deleteBoardFile()

# Especificação dos dados de replay experience
dataSpec = (
    tf.TensorSpec([40, 50, 2], tf.int64, 'state'),
    tf.TensorSpec([40, 50, 2], tf.int64, 'nextState'),
    tf.TensorSpec([1], tf.int64, 'action'),
    tf.TensorSpec([1], tf.float32, 'reward'),
    tf.TensorSpec([1], tf.bool, 'done')
)

# Instancia a classe de replay
replay = Replay(1, 500, dataSpec)

# Configurando o otimizador da descida do gradiente
optimizer  =  tf.keras.optimizers.Adam(learning_rate=0.001)

# Configurando a função de custo
loss       =  tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

# Numero de ações no jogo
numAction = 9

# Numero de neurônios em cada camada oculta
numHidden = 64

# Número máximo de episódios
maxEpisodes = 10000000

# Número máximo de jogadas por episódio
maxStepsPerEpisodes = 9

# Parametro gamma
gamma = 0.999

# Parametros da estratégia E-Greedy
epsilon = 1.000
epsilonFinal = 0.800
epsilonDecay = 0.995

visible = keras.layers.Input(shape=(40, 50, 2))
conv1   = keras.layers.Conv2D(128, kernel_size=3, padding='SAME', activation='relu')(visible)
pool1   = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2   = keras.layers.Conv2D(128, kernel_size=3, padding='SAME', activation='relu')(pool1)
pool2   = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
flat    = keras.layers.Flatten()(pool2)
hidden1 = keras.layers.Dense(128, activation='relu')(flat)
hidden2 = keras.layers.Dense(128, activation='relu')(hidden1)
output  = keras.layers.Dense(numAction, activation='relu')(hidden2)

# Cria os modelos
target  =  keras.Model(inputs=visible, outputs=output)
policy  =  keras.Model(inputs=visible, outputs=output)

# Igualando os pesos e bias dos modelos
policy, target  =  copyWeights( policy, target )

# Roda o treinamento
singleTrain(
    replay,
    policy,
    target,
    loss,
    optimizer,
    maxEpisodes,
    maxStepsPerEpisodes,
    numAction,
    gamma,
    epsilon,
    epsilonFinal,
    epsilonDecay
)