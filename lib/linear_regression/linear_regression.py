from lib.linear_regression.model import Model
import numpy as np
from lib.linear_regression import sql_templates


class LinearRegression:
    def __init__(self, db):
        self.db_connection = db
        self.database = db.database_name
        self.model = None

    def create_model(self, table, x_columns, y_column, model_name=None):
        self.model = Model(table, x_columns, y_column)
        if model_name is not None:
            model_list = self.get_model_list()
            model_names = model_list[:, 1]
            if model_name in model_names:
                raise Exception('Model {} already exists!'.format(model_name))
            if len(self.get_model_list()) < 1:
                next_model_id = 0
            else:
                next_model_id = int(model_list[-1, 0].replace('m', '')) + 1
            self.model.id = "m" + str(next_model_id)
            self.model.name = model_name

        self.__manage_one_hot_encoding()
        self.__save_model()
        return self

    def load_model(self, model_id=None):
        if model_id is None:
            model_id = 'm0'
        elif model_id not in self.get_model_list():
            raise Exception('Provided model_id not found!')

        sql_statement = sql_templates.tmpl['get_all_from_where_id'].render(database=self.database, table='linreg_model',
                                                                           where_statement=model_id)
        data = np.asarray(self.db_connection.execute_query(sql_statement))[0]

        x_columns = []
        for x in data[5].split(','):
            x_columns.append(x)
        y_column = [data[6]]

        prediction_columns = []
        for x in data[7].split(','):
            prediction_columns.append(x)

        ohe_columns = []
        for x in data[9].split(','):
            prediction_columns.append(x)

        self.model = Model(data[3], x_columns[:-1], y_column)
        self.model.id = data[0]
        self.model.name = data[1]
        self.model.state = int(data[2])
        self.model.prediction_table = data[4]
        self.model.prediction_columns = prediction_columns[:-1]
        self.model.input_size = int(data[8])
        self.model.ohe_columns = ohe_columns[:-1]
        return self

    def drop_model(self, model_id=None):
        if model_id is None:
            model_id = 'm0'

        tables = ['_calculation', '_prediction', '_result', '_score']
        for x in tables:
            sql_statement = sql_templates.tmpl['drop_table'].render(table='linreg_' + model_id + x)
            self.db_connection.execute(sql_statement)

        sql_statement = sql_templates.tmpl['delete_from_table_where_id'].render(table='linreg_model',
                                                                                where_statement=model_id)
        try:
            self.db_connection.execute(sql_statement)
        except Exception as e:
            raise Exception('No models available! {}'.format(e))

    def get_model_list(self):
        self.__init_model_table("linreg_model")
        sql_statement = sql_templates.tmpl['get_model_list'].render(database=self.database, table='linreg_model')
        try:
            data = self.db_connection.execute_query(sql_statement)
        except Exception as e:
            raise Exception('No models available! {}'.format(e))

        model_list = [['ID', 'Name']]
        for x in data:
            model_list.append(x)
        return np.asarray(model_list)

    def get_active_model_description(self):
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')

        return "Model " + self.model.id + "\n" + "Name: " + self.model.name + "\n" + "Input table: " + self.model.input_table + "\n" + "X columns: " + str(
            self.model.x_columns) + "\n" + "Y column: " + str(self.model.y_column)

    def estimate(self, table=None, x_columns=None, y_column=None, ohe_handling=False):
        if table is not None or x_columns is not None or y_column is not None:
            self.model = Model(table, x_columns, y_column)
        elif self.model is None:
            raise Exception(
                'No model parameters available! Please load/create a model or provide table, x_columns and y_column as parameters to this function!')

        if ohe_handling:
            self.__manage_one_hot_encoding()

        self.__add_ones_column()
        self.__init_calculation_table()
        self.__init_result_table()
        self.__calculate_equations()
        equations = self.__get_equations()
        xtx = equations[:, 1:self.model.input_size + 1]
        xty = equations[:, self.model.input_size + 1]
        theta = np.linalg.solve(xtx, xty)

        for x in theta:
            sql_statement = sql_templates.tmpl['save_theta'].render(table="linreg_" + self.model.id + "_result",
                                                                    value=x)
            self.db_connection.execute(sql_statement)

        if self.model.state < 1:
            self.model.state = 1
            self.__save_model()

        return self

    def predict(self, table=None, x_columns=None):
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 1:
            raise Exception('Model not trained! Please use estimate method first!')
        self.__init_prediction_table("linreg_" + self.model.id + "_prediction")

        if table is not None and x_columns is not None:
            self.model.prediction_table = table
            self.model.prediction_columns = x_columns
            self.__manage_prediction_one_hot_encoding()
            self.__save_model()
        input_table = self.model.prediction_table
        coefficients = self.get_coefficients()
        prediction_statement = str(coefficients[0][0])

        for i in range(self.model.input_size - 1):
            prediction_statement = prediction_statement + " + " + self.model.prediction_columns[i] + "*" + \
                                   str(coefficients[i + 1][0])
        sql_statement = sql_templates.tmpl['predict'].render(table="linreg_" + self.model.id + "_prediction",
                                                             input_table=input_table,
                                                             prediction_statement=prediction_statement)
        self.db_connection.execute(sql_statement)

        if self.model.state < 2:
            self.model.state = 2
            self.__save_model()

        return self

    def score(self):
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 2:
            raise Exception('No predictions available! Please use predict method first!')

        self.__init_score_table("linreg_" + self.model.id + "_score")
        sql_statement = sql_templates.tmpl['calculate_save_score'].render(table_id='linreg_' + self.model.id,
                                                                          input_table=self.model.input_table,
                                                                          y=self.model.y_column[0])
        self.db_connection.execute(sql_statement)

        if self.model.state < 3:
            self.model.state = 3
            self.__save_model()

        return self

    def get_prediction_array(self):
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 2:
            raise Exception('No predictions available! Please use predict method first!')

        sql_statement = sql_templates.tmpl['select_x_from'].render(x='y_prediction', database=self.database,
                                                                   table='linreg_' + self.model.id + '_prediction')
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data)

    def get_coefficients(self):
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 1:
            raise Exception('Model not trained! Please use estimate method first!')

        sql_statement = sql_templates.tmpl['select_x_from'].render(x='theta', database=self.database,
                                                                   table='linreg_' + self.model.id + '_result')
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data)

    def get_score(self):
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 3:
            raise Exception('No score available! Please use score method first!')

        sql_statement = sql_templates.tmpl['select_x_from'].render(x='score', database=self.database,
                                                                   table='linreg_' + self.model.id + '_score')
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data)[0][0]

    def __manage_one_hot_encoding(self):
        self.__manage_ohe_columns(self.model.input_table, self.model.x_columns)

        # Update input_size
        self.model.update_input_size()

    def __manage_prediction_one_hot_encoding(self):
        self.__manage_ohe_columns(self.model.prediction_table, self.model.prediction_columns)

    def __manage_ohe_columns(self, table, columns):
        # Check all columns
        for x in columns:
            sql_statement = sql_templates.tmpl['column_type'].render(table=table, column=x)
            column_type = np.asarray(self.db_connection.execute_query(sql_statement))[0][0]

            # Check if ohe necessary
            ohe_options = []
            if column_type == 'varchar':
                if x not in self.model.ohe_columns:
                    self.model.ohe_columns.append(x)
                sql_statement = sql_templates.tmpl['select_x_from'].render(database=self.database,
                                                                           table=table, x=x)
                data = np.asarray(self.db_connection.execute_query(sql_statement))[:, 0]
                for y in data:
                    if y not in ohe_options:
                        ohe_options.append(y)

            # Save ohe options per column
            if x in self.model.ohe_columns:
                self.model.ohe_options[x] = ohe_options

            # Create and fill in ohe columns
            sql_statement = sql_templates.tmpl['set_safe_updates'].render(value=0)
            self.db_connection.execute(sql_statement)
            for z in ohe_options:
                # Add ohe column to columns
                if 'linreg_ohe_' + x + '_' + z not in columns:
                    columns.append('linreg_ohe_' + x + '_' + z)

                # Add ohe columns in table
                if 'linreg_ohe_' + x + '_' + z not in self.__get_column_names(table):
                    sql_statement = sql_templates.tmpl['add_column'].render(table=table,
                                                                            column='linreg_ohe_' + x + '_' + z,
                                                                            type='INT')
                    self.db_connection.execute(sql_statement)
                sql_statement = sql_templates.tmpl['set_ohe_column'].render(table=table,
                                                                            ohe_column='linreg_ohe_' + x + '_' + z,
                                                                            input_column=x, value=z)
                self.db_connection.execute(sql_statement)
            sql_statement = sql_templates.tmpl['set_safe_updates'].render(value=1)
            self.db_connection.execute(sql_statement)

        # Remove varchar columns from columns
        for x in self.model.ohe_columns:
            if x in columns:
                columns.remove(x)

    def __save_model(self):
        self.__init_model_table("linreg_model")

        x_columns_string = ""
        for x in self.model.x_columns:
            x_columns_string = x_columns_string + x + ","

        y_column_string = self.model.y_column[0]

        prediction_columns_string = ""
        for x in self.model.x_columns:
            prediction_columns_string = prediction_columns_string + x + ","

        ohe_columns_string = ""
        for x in self.model.ohe_columns:
            ohe_columns_string = ohe_columns_string + x + ","

        sql_statement = sql_templates.tmpl['save_model'].render(id=self.model.id, name=self.model.name,
                                                                state=self.model.state,
                                                                input_table=self.model.input_table,
                                                                prediction_table=self.model.prediction_table,
                                                                x_columns=x_columns_string,
                                                                y_column=y_column_string,
                                                                prediction_columns=prediction_columns_string,
                                                                input_size=self.model.input_size,
                                                                ohe_columns=ohe_columns_string)
        self.db_connection.execute(sql_statement)

        return self

    def __get_equations(self):
        sql_statement = sql_templates.tmpl['get_all_from'].render(database=self.database,
                                                                  table="linreg_" + self.model.id + "_calculation")
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data)

    def __get_column_names(self, table):
        sql_statement = sql_templates.tmpl['table_columns'].render(database=self.database, table=table)
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data)

    def __init_calculation_table(self):
        sql_statement = sql_templates.tmpl['drop_table'].render(table='linreg_' + self.model.id + '_calculation')
        self.db_connection.execute(sql_statement)

        x = []
        for i in range(self.model.input_size):
            x.append('x' + str(i))
        sql_statement = sql_templates.tmpl['init_calculation_table'].render(database=self.database,
                                                                            table='linreg_' + self.model.id + '_calculation',
                                                                            x_columns=x)
        self.db_connection.execute(sql_statement)

    def __init_model_table(self, table):
        sql_statement = sql_templates.tmpl['init_model_table'].render(database=self.database, table=table)
        self.db_connection.execute(sql_statement)

    def __init_result_table(self):
        sql_statement = sql_templates.tmpl['drop_table'].render(table='linreg_' + self.model.id + '_result')
        self.db_connection.execute(sql_statement)

        sql_statement = sql_templates.tmpl['init_result_table'].render(database=self.database,
                                                                       table='linreg_' + self.model.id + '_result')
        self.db_connection.execute(sql_statement)

    def __init_prediction_table(self, table):
        sql_statement = sql_templates.tmpl['drop_table'].render(table=table)
        self.db_connection.execute(sql_statement)

        sql_statement = sql_templates.tmpl['init_prediction_table'].render(database=self.database, table=table)
        self.db_connection.execute(sql_statement)

    def __init_score_table(self, table):
        sql_statement = sql_templates.tmpl['drop_table'].render(table=table)
        self.db_connection.execute(sql_statement)

        sql_statement = sql_templates.tmpl['init_score_table'].render(database=self.database, table=table)
        self.db_connection.execute(sql_statement)

    def __add_ones_column(self):
        if 'linreg_ones' not in self.__get_column_names(self.model.input_table):
            sql_statement = sql_templates.tmpl['add_ones_column'].render(table=self.model.input_table,
                                                                         column='linreg_ones')
            self.db_connection.execute(sql_statement)

    def __calculate_equations(self):
        columns = ['linreg_ones']
        for i in range(len(self.model.x_columns)):
            columns.append(self.model.x_columns[i])
        columns.append(self.model.y_column[0])

        x = []
        for i in range(self.model.input_size):
            x.append('x' + str(i))

        for i in range(self.model.input_size):
            sum_statements = []
            for j in range(self.model.input_size + 1):
                if j < self.model.input_size:
                    sum_statements.append(
                        "sum(" + columns[i] + "*" + columns[j] + ") FROM " + self.model.input_table + "),")
                else:
                    sum_statements.append(
                        "sum(" + columns[i] + "*" + columns[j] + ") FROM " + self.model.input_table + ")")

            sql_statement = sql_templates.tmpl['calculate_equations'].render(
                table='linreg_' + self.model.id + '_calculation', table_input=self.model.input_table,
                sum_statements=sum_statements, x_columns=x)
            self.db_connection.execute(sql_statement)
