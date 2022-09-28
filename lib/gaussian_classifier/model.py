class Model:
    def __init__(self, table, x_columns, y_column, state=None):
        self.id = "m0"
        self.name = self.id
        # States: 0 empty, 1 estimation available, 2 prediction available, 3 score available
        self.state = 0
        self.input_table = table
        self.prediction_table = self.input_table
        self.x_columns = x_columns
        self.y_column = y_column
        self.x_map = self.map_x_columns()
        self.prediction_columns = self.x_columns
        self.input_size = len(x_columns)
        self.no_of_rows_input = 0
        self.y_classes = []
        self.no_of_rows_prediction = 0



    def update_input_size(self):
        self.input_size = len(self.x_columns) + 1
    def map_x_columns(self):
        x_col_map = {}
        for i in range(len(self.x_columns)):
            x_col_map[self.x_columns[i]]= i+1
        return x_col_map


