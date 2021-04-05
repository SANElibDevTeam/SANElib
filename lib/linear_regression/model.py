class Model:
    def __init__(self, table, x_columns, y_column):
        self.id = "m0"
        self.name = self.id
        self.input_table = table
        self.prediction_table = self.input_table
        self.x_columns = x_columns
        self.y_column = y_column
        self.prediction_columns = self.x_columns
        self.input_size = len(x_columns) + 1

