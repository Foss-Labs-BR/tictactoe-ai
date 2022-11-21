def copyWeights(
        model0,
        model1):
    """
    Copia os pesos de um modelo para o outro
    Arguments:
        model0 - Modelo Keras 1
        model1 - Modelo Keras 2
    Returns:
        Void
    """

    # Copias os pesos do modelo 1
    weights2 = model0.trainable_variables

    # Copias os pesos do modelo 2
    weights1 = model1.trainable_variables

    # Itera nos valores
    for v1, v2 in zip(weights1, weights2):

        # Seta os pesos
        v1.assign(v2.numpy())

    # Retorna os modelos
    return model0, model1
