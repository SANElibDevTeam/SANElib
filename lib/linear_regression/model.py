class Model:
    def __init__(self, x_columns, y_column):
        self.id = "m1"
        self.x_columns = x_columns
        self.y_column = y_column
        self.input_size = 1

    def get_id(self):
        return self.id

    def get_x_columns(self):
        return self.x_columns

    def get_y_column(self):
        return self.y_column

    def get_input_size(self):
        return self.input_size
