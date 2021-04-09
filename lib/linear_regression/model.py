class Model():
    def __init__(self, table, x_columns, y_column, state=None):
        self.id = "m0"
        self.name = self.id
        # States: 0 empty, 1 estimation available, 2 prediction available, 3 score available
        self.state = 0
        self.input_table = table
        self.prediction_table = self.input_table
        self.x_columns = x_columns
        self.y_column = y_column
        self.prediction_columns = self.x_columns
        self.input_size = len(x_columns) + 1
        self.ohe_columns = [] # TODO manage model handling (save, load)
