class Model:
    def __init__(self, table, x_columns, y_column):
        self.id = "m0"
        self.name = self.id
        self.input_table = table
        self.x_columns = x_columns
        self.y_column = y_column
        self.input_size = len(x_columns) + 1

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_id(self):
        return self.id

    def get_x_columns(self):
        return self.x_columns

    def get_y_column(self):
        return self.y_column

    def get_input_size(self):
        return self.input_size

    def get_coefficients(self):
        pass

    def get_accuracy(self):
        pass
