from lib.linear_regression.model import Model
import numpy as np
import logging


class GaussianClassifier:
    def __init__(self, db):
        self.db_connection = db
        self.database = db.database_name
        self.model = None
        if db.driver_name == 'mysql+mysqlconnector':
            from lib.linear_regression.sql_templates.mysql import tmpl_mysql
            self.sql_templates = tmpl_mysql

    def set_log_level(self, level):
        if level == "INFO":
            level = logging.INFO
        elif level == "DEBUG":
            level = logging.DEBUG
        elif level == "NONE":
            level = logging.WARN
        else:
            raise Exception('Invalid log level provided. Please select one of the following: INFO, DEBUG, NONE!')

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=level)

    def create_model(self, table, x_columns, y_column, model_name=None):
        logging.info("\n-----\nCREATING MODEL " + str(model_name))
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
        logging.info("\nMODEL CREATED\n-----")
        return self

    def load_model(self, model_id=None):
        logging.info("\n-----\nLOADING MODEL " + str(model_id))
        if model_id is None:
            model_id = 'm0'
        elif model_id not in self.get_model_list():
            raise Exception('Provided model_id not found!')

        sql_statement = self.sql_templates['get_all_from_where_id'].render(database=self.database, table='gaussian_model',
                                                                           where_statement=model_id)
        logging.debug("SQL: " + str(sql_statement))
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
        logging.info("\nMODEL LOADED\n-----")
        return self

    def drop_model(self, model_id=None):
        if model_id is None:
            model_id = 'm0'
        logging.info("\n-----\nDROPPING MODEL " + str(model_id))

        tables = ['_calculation', '_prediction', '_result', '_score']
        for x in tables:
            sql_statement = self.sql_templates['drop_table'].render(table='gaussian_' + model_id + x)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['delete_from_table_where_id'].render(table='gaussian_model',
                                                                                where_statement=model_id)
        logging.debug("SQL: " + str(sql_statement))
        try:
            self.db_connection.execute(sql_statement)
        except Exception as e:
            raise Exception('No models available! {}'.format(e))
        logging.info("\nMODEL DROPPED\n-----")

    def get_model_list(self):
        logging.info("\n-----\nGETTING MODEL LIST")
        self.__init_model_table("gaussian_model")
        sql_statement = self.sql_templates['get_model_list'].render(database=self.database, table='gaussian_model')
        logging.debug("SQL: " + str(sql_statement))
        try:
            data = self.db_connection.execute_query(sql_statement)
        except Exception as e:
            raise Exception('No models available! {}'.format(e))

        model_list = [['ID', 'Name']]
        for x in data:
            model_list.append(x)
        logging.info("\nMODEL LIST RECEIVED\n-----")
        return np.asarray(model_list)

    def get_active_model_description(self):
        logging.info("\n-----\nGETTING ACTIVE MODEL DESCRIPTION")
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')

        logging.info("\nACTIVE MODEL DESCRIPTION RECEIVED\n-----")
        return "Model " + self.model.id + "\n" + "Name: " + self.model.name + "\n" + "Input table: " + self.model.input_table + "\n" + "X columns: " + str(
            self.model.x_columns) + "\n" + "Y column: " + str(self.model.y_column)

    def estimate(self, table=None, x_columns=None, y_column=None, ohe_handling=False):
        logging.info("\n-----\nESTIMATING")
        if table is not None or x_columns is not None or y_column is not None:
            self.model = Model(table, x_columns, y_column)
        elif self.model is None:
            raise Exception(
                'No model parameters available! Please load/create a model or provide table, x_columns and y_column as parameters to this function!')

        if ohe_handling:
            self.__manage_one_hot_encoding()


        self.__init_calculation_table()
        self.__init_result_table()
        self.__calculate_equations()
        equations = self.__get_equations()
        xtx = equations[:, 1:self.model.input_size + 1]
        xty = equations[:, self.model.input_size + 1]
        theta = np.linalg.lstsq(xtx, xty, rcond=None)[0]

        for x in theta:
            sql_statement = self.sql_templates['save_theta'].render(table="gaussian_" + self.model.id + "_result",
                                                                    value=x)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)

        if self.model.state < 1:
            self.model.state = 1
            self.__save_model()
        logging.info("\nESTIMATING FINISHED\n-----")
        return self

    def predict(self, table=None, x_columns=None):
        logging.info("\n-----\nPREDICTING")
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 1:
            raise Exception('Model not trained! Please use estimate method first!')
        self.__init_prediction_table("gaussian_" + self.model.id + "_prediction")

        if table is not None and x_columns is not None:
            self.model.prediction_table = table
            self.model.prediction_columns = x_columns
            self.__manage_prediction_one_hot_encoding()
            if self.model.input_size - 1 != len(self.model.prediction_columns):
                raise Exception(
                    'Please ensure the number of columns to be predicted matches the columns used in estimate!')
            self.__save_model()
        input_table = self.model.prediction_table
        coefficients = self.get_coefficients()
        prediction_statement = str(coefficients[0][0])

        for i in range(self.model.input_size - 1):
            prediction_statement = prediction_statement + " + " + self.model.prediction_columns[i] + "*" + \
                                   str(coefficients[i + 1][0])
        sql_statement = self.sql_templates['predict'].render(table="gaussian_" + self.model.id + "_prediction",
                                                             input_table=input_table,
                                                             prediction_statement=prediction_statement)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        if self.model.state < 2:
            self.model.state = 2
            self.__save_model()

        logging.info("\nPREDICTING FINISHED\n-----")
        return self

    def score(self):
        logging.info("\n-----\nCALCULATING SCORE")
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 2:
            raise Exception('No predictions available! Please use predict method first!')

        self.__init_score_table("gaussian_" + self.model.id + "_score")
        sql_statement = self.sql_templates['calculate_save_score'].render(table_id='gaussian_' + self.model.id,
                                                                          input_table=self.model.input_table,
                                                                          y=self.model.y_column[0])
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        if self.model.state < 3:
            self.model.state = 3
            self.__save_model()

        logging.info("\nCALCULATION FINISHED\n-----")
        return self

    def get_prediction_array(self):
        logging.info("\n-----\nGETTING PREDICTION ARRAY")
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 2:
            raise Exception('No predictions available! Please use predict method first!')

        sql_statement = self.sql_templates['select_x_from'].render(x='y_prediction', database=self.database,
                                                                   table='gaussian_' + self.model.id + '_prediction')
        logging.debug("SQL: " + str(sql_statement))
        data = self.db_connection.execute_query(sql_statement)
        logging.info("\nPREDICTION ARRAY RECEIVED\n-----")
        return np.asarray(data)

    def get_coefficients(self):
        logging.info("\n-----\nGETTING COEFFICIENTS")
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 1:
            raise Exception('Model not trained! Please use estimate method first!')

        sql_statement = self.sql_templates['select_x_from'].render(x='theta', database=self.database,
                                                                   table='gaussian_' + self.model.id + '_result')
        logging.debug("SQL: " + str(sql_statement))
        data = self.db_connection.execute_query(sql_statement)
        logging.info("\nCOEFFICIENTS RECEIVED\n-----")
        return np.asarray(data)

    def get_score(self):
        logging.info("\n-----\nGETTING SCORE")
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 3:
            raise Exception('No score available! Please use score method first!')

        sql_statement = self.sql_templates['select_x_from'].render(x='score', database=self.database,
                                                                   table='gaussian_' + self.model.id + '_score')
        logging.debug("SQL: " + str(sql_statement))
        data = self.db_connection.execute_query(sql_statement)
        logging.info("\nSCORE RECEIVED\n-----")
        return np.asarray(data)[0][0]

    def __manage_one_hot_encoding(self):
        logging.info("MANAGING ONE HOT ENCODING")
        self.__manage_ohe_columns(self.model.input_table, self.model.x_columns)

        # Update input_size
        self.model.update_input_size()

    def __manage_prediction_one_hot_encoding(self):
        logging.info("MANAGING ONE HOT ENCODING FOR PREDICTION")
        self.__manage_ohe_columns(self.model.prediction_table, self.model.prediction_columns)

    def __manage_ohe_columns(self, table, columns):
        # Check all columns
        for x in columns:
            sql_statement = self.sql_templates['column_type'].render(table=table, column=x)
            logging.debug("SQL: " + str(sql_statement))
            column_type = np.asarray(self.db_connection.execute_query(sql_statement))[0][0]

            # Check if ohe necessary
            ohe_options = []
            if column_type == 'varchar':
                if x not in self.model.ohe_columns:
                    self.model.ohe_columns.append(x)
                sql_statement = self.sql_templates['select_x_from'].render(database=self.database,
                                                                           table=table, x=x)
                logging.debug("SQL: " + str(sql_statement))
                data = np.asarray(self.db_connection.execute_query(sql_statement))[:, 0]
                for y in data:
                    if y not in ohe_options:
                        ohe_options.append(y)

            # Save ohe options per column
            if x in self.model.ohe_columns:
                self.model.ohe_options[x] = ohe_options

            # Create and fill in ohe columns
            sql_statement = self.sql_templates['set_safe_updates'].render(value=0)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)
            for z in ohe_options:
                # Add ohe column to columns
                if 'gaussian_ohe_' + x + '_' + z not in columns:
                    columns.append('gaussian_ohe_' + x + '_' + z)

                # Add ohe columns in table
                if 'gaussian_ohe_' + x + '_' + z not in self.__get_column_names(table):
                    sql_statement = self.sql_templates['add_column'].render(table=table,
                                                                            column='gaussian_ohe_' + x + '_' + z,
                                                                            type='INT')
                    logging.debug("SQL: " + str(sql_statement))
                    self.db_connection.execute(sql_statement)
                sql_statement = self.sql_templates['set_ohe_column'].render(table=table,
                                                                            ohe_column='gaussian_ohe_' + x + '_' + z,
                                                                            input_column=x, value=z)
                logging.debug("SQL: " + str(sql_statement))
                self.db_connection.execute(sql_statement)
            sql_statement = self.sql_templates['set_safe_updates'].render(value=1)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)

        # Remove varchar columns from columns
        for x in self.model.ohe_columns:
            if x in columns:
                columns.remove(x)

    def __save_model(self):
        logging.info("SAVING MODEL")
        self.__init_model_table("gaussian_model")

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

        sql_statement = self.sql_templates['save_model'].render(id=self.model.id, name=self.model.name,
                                                                state=self.model.state,
                                                                input_table=self.model.input_table,
                                                                prediction_table=self.model.prediction_table,
                                                                x_columns=x_columns_string,
                                                                y_column=y_column_string,
                                                                prediction_columns=prediction_columns_string,
                                                                input_size=self.model.input_size,
                                                                ohe_columns=ohe_columns_string)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        return self

    def __get_equations(self):
        logging.info("GETTING EQUATIONS")
        sql_statement = self.sql_templates['get_all_from'].render(database=self.database,
                                                                  table="gaussian_" + self.model.id + "_calculation")
        logging.debug("SQL: " + str(sql_statement))
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data, dtype='double')

    def __get_column_names(self, table):
        logging.info("GETTING COLUMN NAMES")
        sql_statement = self.sql_templates['table_columns'].render(database=self.database, table=table)
        logging.debug("SQL: " + str(sql_statement))
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data)

    def __init_calculation_table(self):
        logging.info("INITIALIZING CALCULATION TABLE")
        sql_statement = self.sql_templates['drop_table'].render(table='gaussian_' + self.model.id + '_calculation')
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        x = []
        for i in range(self.model.input_size):
            x.append("mean_" + self.model.x_columns[i])
            x.append("std_" + self.model.x_columns[i])
        sql_statement = self.sql_templates['init_calculation_table'].render(database=self.database,
                                                                            table='gaussian_' + self.model.id + '_calculation',
                                                                            x_columns=x)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __init_model_table(self, table):
        logging.info("INITIALIZING MODEL TABLE")
        sql_statement = self.sql_templates['init_model_table'].render(database=self.database, table=table)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __init_result_table(self):
        logging.info("INITIALIZING RESULT TABLE")
        sql_statement = self.sql_templates['drop_table'].render(table='gaussian_' + self.model.id + '_result')
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['init_result_table'].render(database=self.database,
                                                                       table='gaussian_' + self.model.id + '_result')
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __init_prediction_table(self, table):
        logging.info("INITIALIZING PREDICTION TABLE")
        sql_statement = self.sql_templates['drop_table'].render(table=table)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['init_prediction_table'].render(database=self.database, table=table)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __init_score_table(self, table):
        logging.info("INITIALIZING SCORE TABLE")
        sql_statement = self.sql_templates['drop_table'].render(table=table)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['init_score_table'].render(database=self.database, table=table)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)
    def __add_ones_column(self):
        return self

    def __calculate_equations(self):
        return self