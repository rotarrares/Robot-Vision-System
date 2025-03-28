class FloorPredictions:
    def __init__(self, predictions, num_labels):
        self.__num_labels = num_labels
        # Store the original shape
        self.__original_shape = predictions.shape
        # Ensure predictions maintains its shape
        if len(predictions.shape) != 2:
            raise ValueError(f"Predictions must be 2D array, got shape {predictions.shape}")
        self.__predictions = predictions

    def get_predictions(self):
        # Ensure we return with the original shape
        return self.__predictions.reshape(self.__original_shape)

    def get_num_labels(self):
        return self.__num_labels