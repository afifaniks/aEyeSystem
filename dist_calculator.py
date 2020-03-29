import pickle


class DistanceCalculator:
    def __init__(self, weight_path):
        # load
        with open(weight_path, 'rb') as file:
            self.dist_model = pickle.load(file)

    def predict(self, y_axis_co):
        """
        :param y_axis_co: Receives LOWER Y axis co ordinate of an object
        :return: Predicted distance from camera
        """
        return self.dist_model.predict([[y_axis_co]])