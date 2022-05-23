import itertools

from lib.gaussian_classifier.model import Model
import numpy as np
import logging
import sqlparse


class GaussianClassifier:
    def __init__(self, db):
        self.db_connection = db
        self.database = db.database_name
        self.model = None
        if db.driver_name == 'mysql+mysqlconnector':
            from lib.gaussian_classifier.sql_templates.mysql import tmpl_mysql
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

        self.__save_model()
        logging.info("\nMODEL CREATED\n-----")
        return self

    def load_model(self, model_id=None):
        logging.info("\n-----\nLOADING MODEL " + str(model_id))
        if model_id is None:
            model_id = 'm0'
        elif model_id not in self.get_model_list():
            raise Exception('Provided model_id not found!')

        sql_statement = self.sql_templates['get_all_from_where_id'].render(database=self.database,
                                                                           table='gaussian_model',
                                                                           where_statement=model_id)
        logging.debug("SQL: " + str(sql_statement))
        data = np.asarray(self.db_connection.execute_query(sql_statement))[0]

        x_columns = []
        for x in data[5].split(','):
            x_columns.append(x)

        y_classes = []
        for y in data[6].split(','):
            if y != '':
                y_classes.append(int(y))

        y_column = [data[7]]

        prediction_columns = []
        for x in data[8].split(','):
            prediction_columns.append(x)

        self.model = Model(data[3], x_columns[:-1], y_column)
        self.model.id = data[0]
        self.model.name = data[1]
        self.model.state = int(data[2])
        self.model.prediction_table = data[4]
        self.model.prediction_columns = prediction_columns[:-1]
        self.model.input_size = int(data[9])
        self.model.no_of_rows_input = int(data[10])
        self.model.y_classes = y_classes[:-1]

        logging.info("\nMODEL LOADED\n-----")
        return self

    def drop_model(self, model_id=None):
        if model_id is None:
            model_id = 'm0'
        logging.info("\n-----\nDROPPING MODEL " + str(model_id))

        tables = ['_estimation', '_prediction', '_uni_gauss_prob']
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

    def train_test_split(self, ratio=0.8, seed=1):
        self.ratio = ratio
        self.seed = seed

        sql_statement = self.sql_templates['drop_table'].render(table='gaussian_' + self.model.id + '_train',database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['_train'].render(train_table='gaussian_' + self.model.id + '_train',
                                                            input_table=self.model.input_table, input_seed=seed,
                                                            input_ratio=ratio,database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['drop_table'].render(table='gaussian_' + self.model.id + '_eval',database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['_eval'].render(train_table='gaussian_' + self.model.id + '_eval',
                                                           input_table=self.model.input_table, input_seed=seed,
                                                           input_ratio=ratio,database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        self.model.input_table = 'gaussian_' + self.model.id + '_train'
        self.model.prediction_table = 'gaussian_' + self.model.id + '_eval'

    def estimate(self, table=None, x_columns=None, y_column=None, multivariate=True):
        logging.info("\n-----\nESTIMATING")
        if table is not None or x_columns is not None or y_column is not None:
            self.model = Model(table, x_columns, y_column)
            self.model.multivariate = multivariate
            self.model.y_classes = self.__get_targets()
        elif self.model is None:
            raise Exception(
                'No model parameters available! Please load/create a model or provide table, x_columns and y_column as parameters to this function!')

        if self.model.multivariate:
            self.model.id = "m0m"
            self.train_test_split()
            self.model.no_of_rows_input = self.__get_no_of_rows(self.model.input_table)
            self.model.no_of_rows_prediction = self.__get_no_of_rows(self.model.prediction_table)
            self.__init_mean_table()
            self.__calculate_means()
            self.__calculate_gaussian_probabilities_multivariate()


        else:
            self.model.id = "m0u"
            self.train_test_split()
            self.model.no_of_rows_input = self.__get_no_of_rows(self.model.input_table)
            self.model.no_of_rows_prediction = self.__get_no_of_rows(self.model.prediction_table)
            self.__init_mean_table()
            self.__calculate_means()
            self.__init_variance_table()
            self.__calculate_variances()
            self.__init_uni_gauss_prob_table()
            self.__calculate_gaussian_probabilities_univariate()

        if self.model.state < 1:
            self.model.state = 1
            self.__save_model()
        logging.info("\nESTIMATING FINISHED\n-----")
        return self

    def predict(self):
        logging.info("\n-----\nPREDICTING")
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 1:
            raise Exception('Model not trained! Please use estimate method first!')

        if self.model.multivariate:
            self.__init_vector_tables(self.model.no_of_rows_prediction)
            self.__get_mahalonibis_distance()
            self.__multivariate_probability()

            self.__drop_prediction_table("gaussian_" + self.model.id + "_prediction")

            sql_statement = self.sql_templates['predict'].render(table="gaussian_" + self.model.id + "_prediction",
                                                                 row_no=list(range(self.model.no_of_rows_prediction)),
                                                                 estimation_table="gaussian_" + self.model.id + "_estimation",database=self.database)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)




        else:
            self.__drop_prediction_table("gaussian_" + self.model.id + "_prediction")

            sql_statement = self.sql_templates['predict'].render(table="gaussian_" + self.model.id + "_prediction",
                                                                 row_no=list(range(self.model.no_of_rows_prediction)),
                                                                 estimation_table='gaussian_' + self.model.id + '_uni_gauss_prob',database=self.database)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)

        if self.model.state < 2:
            self.model.state = 2
            self.__save_model()

        logging.info("\nPREDICTING FINISHED\n-----")
        return self

    def clean_up(self,multivariate=True):
        if multivariate:
            self.__drop_mean_table()
            self.__drop_vector_tables(self.model.no_of_rows_prediction)
            self.__drop_determinante_table('gaussian_' + self.model.id + "_determinante")
            self.__drop_inverse_covariance_matrix_table(self.model.y_classes)
            self.__drop_covariance_matrix_table(self.model.y_classes)
        else:
            self.__drop_variance_table()
            self.__drop_covariance_input_table()
            self.__drop_mean_table()


    def score(self):
        logging.info("\n-----\nCALCULATING SCORE")
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 2:
            raise Exception('No predictions available! Please use predict method first!')

        sql_statement = self.sql_templates['add_column'].render(table="gaussian_" + self.model.id + "_prediction",
                                                                column="y", type="DOUBLE NULL",database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['add_column'].render(table="gaussian_" + self.model.id + "_prediction",
                                                                column="score", type="DOUBLE NULL",database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['insert_target'].render(table="gaussian_" + self.model.id + "_prediction",
                                                                   column="y", input_table=self.model.prediction_table,
                                                                   row_no=list(range(self.model.no_of_rows_prediction)),
                                                                   y_column=self.model.y_column[0],database=self.database)
        for statement in sqlparse.split(sql_statement):
            if statement:
                logging.debug("SQL: " + str(statement))
                self.db_connection.execute(statement)

        sql_statement = self.sql_templates['substract_columns'].render(
            table="gaussian_" + self.model.id + "_prediction",
            column="score", feature_1="y", feature_2="y_prediction",database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['insert_score'].render(
            table="gaussian_" + self.model.id + "_prediction",
            column="score",database=self.database)
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

    def get_target_array(self):
        logging.info("\n-----\nGETTING TARGET ARRAY")
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')

        sql_statement = self.sql_templates['get_target_array'].render(database=self.database,
                                                                      table=self.model.prediction_table,
                                                                      y=self.model.y_column[0])
        logging.debug("SQL: " + str(sql_statement))
        data = self.db_connection.execute_query(sql_statement)
        logging.info("\nTARGET ARRAY RECEIVED\n-----")
        return np.asarray(data)

    def get_accuracy(self):
        logging.info("\n-----\nGETTING ACCURACY")
        if self.model is None:
            raise Exception('No model parameters available! Please load/create a model!')
        elif self.model.state < 3:
            raise Exception('No score available! Please use score method first!')

        sql_statement = self.sql_templates['get_accuracy'].render(x='score', database=self.database,
                                                                  table='gaussian_' + self.model.id + '_prediction',
                                                                  no_rows_prediction=self.model.no_of_rows_prediction)
        logging.debug("SQL: " + str(sql_statement))
        data = self.db_connection.execute_query(sql_statement)
        logging.info("\nACCURACY RECEIVED\n-----")
        return np.asarray(data)[0][0]

    def __save_model(self):
        logging.info("SAVING MODEL")
        self.__init_model_table("gaussian_model")

        x_columns_string = ""
        for x in self.model.x_columns:
            x_columns_string = x_columns_string + x + ","

        y_classes_string = ""
        for y in self.model.y_classes:
            y_classes_string = y_classes_string + str(y) + ","

        y_column_string = self.model.y_column[0]

        prediction_columns_string = ""
        for x in self.model.x_columns:
            prediction_columns_string = prediction_columns_string + x + ","

        sql_statement = self.sql_templates['save_model'].render(id=self.model.id, name=self.model.name,
                                                                state=self.model.state,
                                                                input_table=self.model.input_table,
                                                                prediction_table=self.model.prediction_table,
                                                                x_columns=x_columns_string,
                                                                y_classes=y_classes_string,
                                                                y_column=y_column_string,
                                                                prediction_columns=prediction_columns_string,
                                                                input_size=self.model.input_size,
                                                                no_of_rows_input=self.model.no_of_rows_input,
                                                                no_of_rows_prediction=self.model.no_of_rows_prediction)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        return self

    def __get_column_names(self, table):
        logging.info("GETTING COLUMN NAMES")
        sql_statement = self.sql_templates['table_columns'].render(database=self.database, table=table)
        logging.debug("SQL: " + str(sql_statement))
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data)

    def __init_mean_table(self):
        logging.info("INITIALIZING MEAN TABLE")
        sql_statement = self.sql_templates['drop_table'].render(table='gaussian_' + self.model.id + '_mean',database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['init_calculation_table'].render(database=self.database,
                                                                            table='gaussian_' + self.model.id + '_mean',
                                                                            x_columns=self.model.x_columns)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __init_variance_table(self):
        logging.info("INITIALIZING CALCULATION TABLE")
        sql_statement = self.sql_templates['drop_table'].render(table='gaussian_' + self.model.id + '_variance',database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['init_calculation_table'].render(database=self.database,
                                                                            table='gaussian_' + self.model.id + '_variance',
                                                                            x_columns=self.model.x_columns)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __init_model_table(self, table):
        logging.info("INITIALIZING MODEL TABLE")
        sql_statement = self.sql_templates['init_model_table'].render(database=self.database, table=table)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __init_uni_gauss_prob_table(self):
        logging.info("INITIALIZING _uni_gauss_prob TABLE")
        sql_statement = self.sql_templates['drop_table'].render(table='gaussian_' + self.model.id + '_uni_gauss_prob',database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['init_uni_gauss_prob_table'].render(database=self.database,
                                                                               table='gaussian_' + self.model.id + '_uni_gauss_prob',
                                                                               x_columns=self.model.x_columns)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __drop_prediction_table(self, table):
        logging.info("INITIALIZING PREDICTION TABLE")
        sql_statement = self.sql_templates['drop_table'].render(table=table,database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)


    def __get_targets(self):
        logging.info("GETTING TARGET CLASSES")
        sql_statement = self.sql_templates['get_targets'].render(table=self.model.input_table, y=self.model.y_column[0],database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        query_return = self.db_connection.execute_query(sql_statement)
        data = []
        for element in query_return:
            data.append(element[0])
        return np.asarray(data)

    def __get_no_of_rows(self, table):
        logging.info("GETTING NUMBER OF ROWS")
        sql_statement = self.sql_templates['get_no_of_rows'].render(table=table,database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        query_return = self.db_connection.execute_query(sql_statement)
        data = []
        for element in query_return:
            data.append(element[0])
        return data[0]

    def __remove_help_row(self, table):
        sql_statement = self.sql_templates['drop_row'].render(table=table,database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __calculate_means(self):
        logging.info("CALCULATING MEANS")
        y_classes = self.model.y_classes

        sql_statement = self.sql_templates['calculate_means'].render(
            table='gaussian_' + self.model.id + '_mean', input_table=self.model.input_table,
            y_classes=y_classes, x_columns_means=self.model.x_columns, x_columns=self.model.x_columns,
            target=self.model.y_column[0],database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)
        self.__remove_help_row('gaussian_' + self.model.id + '_mean')


    def __calculate_variances(self):
        logging.info("CALCULATING VARIANCES")
        y_classes = self.model.y_classes

        sql_statement = self.sql_templates['calculate_variances'].render(
            table='gaussian_' + self.model.id + '_variance', input_table=self.model.input_table,
            y_classes=y_classes, x_columns_variances=self.model.x_columns, x_columns=self.model.x_columns,
            target=self.model.y_column[0],database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)
        self.__remove_help_row('gaussian_' + self.model.id + '_variance')

    def __calculate_gaussian_probabilities_univariate(self):
        logging.info("CALCULATING UNIVARIATE GAUSSIAN PROBABILITIES")
        y_classes = self.model.y_classes
        rows = self.model.no_of_rows_prediction
        no_of_rows = list(range(0, rows))

        for n in no_of_rows:
            gauss_statements = []
            for y in y_classes:
                gauss_prob = []
                for x in self.model.x_columns:
                    gauss_prob.append(
                        f"(SELECT (1 / SQRT(2*PI()*(SELECT {x} from {self.database}.gaussian_{self.model.id}_variance where y = {y}))*EXP(-POW((SELECT {x} from {self.model.prediction_table} LIMIT {n},1)-(SELECT {x} from  gaussian_{self.model.id}_mean where y = {y}),2)/(2*(SELECT {x} from gaussian_{self.model.id}_variance where y = {y}))))),"
                    )
                str_gauss_prob = ''.join(str(e) for e in gauss_prob)
                gauss_statements.append(f"({n}," + str_gauss_prob + f"{y}),"

                                        )

            gauss_statements[-1] = gauss_statements[-1][:-1]
            sql_statement = self.sql_templates['calculate_gauss_prob_univariate'].render(
                table='gaussian_' + self.model.id + '_uni_gauss_prob', x_columns=self.model.x_columns,
                gauss_statements=gauss_statements,database=self.database)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)
            # self.__remove_help_row('gaussian_' + self.model.id + '_uni_gauss_prob')

        multiplication_string = ""
        for x in self.model.x_columns:
            multiplication_string = multiplication_string + x + " * "
        multiplication_string = multiplication_string[:-2]

        sql_statement = self.sql_templates['calculate_univariate_density'].render(
            table='gaussian_' + self.model.id + '_uni_gauss_prob', column="gaussian_distribution",
            multiplication_string=multiplication_string,database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['p_y'].render(
            table='gaussian_' + self.model.id + '_uni_gauss_prob',
            y_column=self.model.y_column[0],
            y_classes=self.model.y_classes,
            input_table=self.model.input_table,
            row_no=self.model.no_of_rows_input,database=self.database)

        for statement in sqlparse.split(sql_statement):
            if statement:
                logging.debug("SQL: " + str(statement))
                self.db_connection.execute(statement)

        sql_statement = self.sql_templates['multiply_columns'].render(
            table='gaussian_' + self.model.id + '_uni_gauss_prob',
            column="probability",
            feature_1="p_y",
            feature_2="gaussian_distribution",
            database=self.database
        )
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __calculate_gaussian_probabilities_multivariate(self):
        y_classes = self.model.y_classes
        matrix = self.__create_matrix()
        self.__get_diff_from_mean()
        self.__multiply_columns(matrix)
        self.__init_covariance_matrix_table(y_classes)
        self.__fill_covariance_matrix(matrix, y_classes)

        self.__init_determinante_table('gaussian_' + self.model.id + "_determinante")
        self.__init_inverse_covariance_matrix_table(y_classes)
        for y_class in y_classes:
            self.__get_matrix_inverse(matrix, f"gaussian_{self.model.id}_covariance_matrix_{y_class}")

    def __create_matrix(self):
        matrix = []
        for i in range(len(self.model.x_columns)):
            row = []
            for j in range(len(self.model.x_columns)):
                column = []
                column.append(self.model.x_columns[i])
                column.append(self.model.x_columns[j])
                row.append(column)
            matrix.append(row)
        return matrix

    def __init_covariance_matrix_table(self, y_classes):
        logging.info("INITALIZING COVARIANCE MATRIX TABLE")
        for y_class in y_classes:
            sql_statement = self.sql_templates['drop_table'].render(
                table='gaussian_' + self.model.id + '_covariance_matrix_' + str(y_class),database=self.database)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)

            sql_statement = self.sql_templates['init_covariance_table'].render(
                table='gaussian_' + self.model.id + '_covariance_matrix_' + str(y_class),
                x_columns=self.model.x_columns,database=self.database)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)
            for column in self.model.x_columns:
                column = '"' + column + '"'
                sql_statement = self.sql_templates['insert_id'].render(
                    table='gaussian_' + self.model.id + '_covariance_matrix_' + str(y_class), id=column,database=self.database)
                logging.debug("SQL: " + str(sql_statement))
                self.db_connection.execute(sql_statement)

    def __drop_covariance_matrix_table(self, y_classes):
        logging.info("DROPPING COVARIANCE MATRIX TABLE")
        for y_class in y_classes:
            sql_statement = self.sql_templates['drop_table'].render(
                table='gaussian_' + self.model.id + '_covariance_matrix_' + str(y_class),database=self.database)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)

    def __drop_mean_table(self):
        logging.info("DROPPING MEAN MATRIX TABLE")
        sql_statement = self.sql_templates['drop_table'].render(
            table='gaussian_' + self.model.id + '_mean',database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __drop_variance_table(self):
        logging.info("DROPPING VARIANCE TABLE")
        sql_statement = self.sql_templates['drop_table'].render(
            table='gaussian_' + self.model.id + '_variance', database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __drop_covariance_input_table(self):
        logging.info("DROPPING COVARIANCE INPUT TABLE")
        sql_statement = self.sql_templates['drop_table'].render(
            table='gaussian_' + self.model.id + '_covariance_input', database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __init_inverse_covariance_matrix_table(self, y_classes):
        logging.info("INITALIZING INVERSE COVARIANCE MATRIX TABLE")
        for y_class in y_classes:
            sql_statement = self.sql_templates['drop_table'].render(
                table='gaussian_' + self.model.id + '_covariance_matrix_' + str(y_class) + "_inverse_matrix",database=self.database)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)

            sql_statement = self.sql_templates['init_covariance_table'].render(
                table='gaussian_' + self.model.id + '_covariance_matrix_' + str(y_class) + "_inverse_matrix",
                x_columns=self.model.x_columns,database=self.database)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)

            sql_statement = self.sql_templates['drop_table'].render(
                table='gaussian_' + self.model.id + '_covariance_matrix_' + str(y_class) + "_inverse",database=self.database)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)

            sql_statement = self.sql_templates['init_inverse_table'].render(
                table='gaussian_' + self.model.id + '_covariance_matrix_' + str(y_class) + "_inverse",
                row_name="k", col_name="j",database=self.database)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)

    def __drop_inverse_covariance_matrix_table(self, y_classes):
        logging.info("DROPPING INVERSE COVARIANCE MATRIX TABLES")
        for y_class in y_classes:
            sql_statement = self.sql_templates['drop_table'].render(
                table='gaussian_' + self.model.id + '_covariance_matrix_' + str(y_class) + "_inverse_matrix",database=self.database)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)
            sql_statement = self.sql_templates['drop_table'].render(
                table='gaussian_' + self.model.id + '_covariance_matrix_' + str(y_class) + "_inverse",
                database=self.database)
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)

    def __init_vector_tables(self, no_rows):
        logging.info("INITIALIZING VECTOR TABLES")
        for y in self.model.y_classes:
            for row in list(range(no_rows)):
                sql_statement = self.sql_templates['drop_table'].render(
                    table='gaussian_' + self.model.id + '_vector_' + str(row)+ "_" + str(y),database=self.database)
                logging.debug("SQL: " + str(sql_statement))
                self.db_connection.execute(sql_statement)

                sql_statement = self.sql_templates['init_inverse_table'].render(
                    table='gaussian_' + self.model.id + '_vector_' + str(row)+ "_" + str(y),
                    row_name="i", col_name="k",database=self.database)
                logging.debug("SQL: " + str(sql_statement))
                self.db_connection.execute(sql_statement)

    def __drop_vector_tables(self, no_rows):
        logging.info("DROPPING VECTOR TABLES")
        for row in list(range(no_rows)):
            for y in self.model.y_classes:
                sql_statement = self.sql_templates['drop_table'].render(
                    table='gaussian_' + self.model.id + '_vector_' + str(row)+ "_" + str(y),database=self.database)
                logging.debug("SQL: " + str(sql_statement))
                self.db_connection.execute(sql_statement)

    def __get_diff_from_mean(self):
        logging.info("CALCULATING feature - avg(feature) where class")

        sql_statement = self.sql_templates['drop_table'].render(table='gaussian_' + self.model.id + '_covariance_input',database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['create_table_like'].render(
            new_table='gaussian_' + self.model.id + '_covariance_input', original_table=self.model.input_table,database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['copy_table'].render(
            new_table='gaussian_' + self.model.id + '_covariance_input', original_table=self.model.input_table,database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)
        y_classes = self.model.y_classes
        for y_class in y_classes:
            for column in self.model.x_columns:
                sql_statement = self.sql_templates['add_column'].render(
                    table='gaussian_' + self.model.id + '_covariance_input',
                    column=column + '_diff_mean_' + str(y_class), type="DOUBLE NULL",database=self.database)
                logging.debug("SQL: " + str(sql_statement))
                self.db_connection.execute(sql_statement)

                sql_statement = self.sql_templates['diff_mean'].render(
                    table='gaussian_' + self.model.id + '_covariance_input',
                    column=column + '_diff_mean_' + str(y_class), feature_1=column, y_class=y_class,
                    mean_table='gaussian_' + self.model.id + '_mean',database=self.database)
                logging.debug("SQL: " + str(sql_statement))
                self.db_connection.execute(sql_statement)


    def __multiply_columns(self, matrix):
        y_classes = self.model.y_classes
        for row in matrix:
            for y_class in y_classes:
                for permutation in row:
                    sql_statement = self.sql_templates['add_column'].render(
                        table='gaussian_' + self.model.id + '_covariance_input',
                        column=str(self.model.x_map[permutation[0]]) + '_x_' + str(
                            self.model.x_map[permutation[1]]) + "_" + str(y_class), type="DOUBLE NULL",database=self.database)
                    logging.debug("SQL: " + str(sql_statement))
                    self.db_connection.execute(sql_statement)

                    sql_statement = self.sql_templates['multiply_columns'].render(
                        table='gaussian_' + self.model.id + '_covariance_input',
                        column=str(self.model.x_map[permutation[0]]) + '_x_' + str(
                            self.model.x_map[permutation[1]]) + "_" + str(y_class),
                        feature_1=permutation[0] + '_diff_mean_' + str(y_class),
                        feature_2=permutation[1] + '_diff_mean_' + str(y_class),database=self.database)
                    logging.debug("SQL: " + str(sql_statement))
                    self.db_connection.execute(sql_statement)

    def __fill_covariance_matrix(self, matrix, y_classes):
        for y_class in y_classes:
            for row in matrix:
                for permutation in row:
                    covariance = str(self.model.x_map[permutation[0]]) + '_x_' + str(
                        self.model.x_map[permutation[1]]) + "_" + str(y_class)
                    sql_statement = self.sql_templates['fill_covariance_matrix'].render(
                        table='gaussian_' + self.model.id + '_covariance_matrix_' + str(y_class),
                        gaussian_input_table='gaussian_' + self.model.id + '_covariance_input',
                        covariance=covariance,
                        feature_1='"' + permutation[0] + '"',
                        feature_2=permutation[1],
                        y_column=self.model.y_column[0],
                        y_class=y_class,
                        input_table=self.model.input_table,
                        no_rows=self.model.no_of_rows_input,
                    database=self.database)
                    logging.debug("SQL: " + str(sql_statement))
                    self.db_connection.execute(sql_statement)

    def __get_matrix_determinante(self, m, covariance_table):
        # base case for 2x2 matrix
        if len(m) == 2:
            m01 = f'SELECT {m[0][1][1]} from {covariance_table} WHERE id= "{m[0][1][0]}"'
            m10 = f'SELECT {m[1][0][1]} * ({m01}) from {covariance_table} WHERE id= "{m[1][0][0]}"'
            m11 = f'SELECT {m[1][1][1]} from {covariance_table} WHERE id= "{m[1][1][0]}"'
            m00 = f'SELECT {m[0][0][1]} * ({m11}) - ({m10}) from {covariance_table} WHERE id= "{m[0][0][0]}"'
            logging.debug("SQL: " + str(m00))
            data = list(self.db_connection.execute_query(m00)[0])
            if data[0] == 0.0:
                data[0] = 0.0001
            return data[0]
        else:
            determinant = 0
            for c in range(len(m)):
                sql_statement = self.sql_templates['m0c'].render(
                    covariance_matrix=covariance_table,
                    m0c0='"' + m[0][c][0] + '"',
                    m0c1=m[0][c][1],database=self.database)
                logging.debug("SQL: " + str(sql_statement))
                m0c = list(self.db_connection.execute_query(sql_statement)[0])
                determinant += ((-1) ** c) * m0c[0] * self.__get_matrix_determinante(self.__get_matrix_minor(m, 0, c),
                                                                                     covariance_table)
            if determinant == 0:
                determinant = 0.0001
            return determinant

    def __get_matrix_minor(self, m, i, j):
        return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]

    def __transpose_matrix(self, covariance_table):
        sql_statement = self.sql_templates['transpose_matrix'].render(
            covariance_matrix=covariance_table,
            features=self.model.x_columns,
            inverse_matrix=covariance_table,database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __init_determinante_table(self, table):
        sql_statement = self.sql_templates['drop_table'].render(table=table,database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['init_determinante_table'].render(table=table,database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __drop_determinante_table(self, table):
        logging.info("DROPPING DETERMINANTE TABLE")
        sql_statement = self.sql_templates['drop_table'].render(table=table,database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __get_matrix_inverse(self, m, covariance_table):
        determinant = self.__get_matrix_determinante(m, covariance_table)
        sql_statement = self.sql_templates['insert_determinante'].render(table="'" + covariance_table + "'",
                                                                         determinante=determinant,
                                                                         determinante_table='gaussian_' + self.model.id + "_determinante",database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)
        # special case for 2x2 matrix:
        if len(m) == 2:
            logging.debug("INSERTING 2x2 INVERSE COVARIANCE MATRIX")
            m00 = f'SELECT {m[0][0][1]} / {determinant} from {covariance_table} WHERE id= "{m[0][0][0]}"'
            m01 = f'SELECT {m[0][1][1]}*(-1) / {determinant} from {covariance_table} WHERE id= "{m[0][1][0]}"'
            m10 = f'SELECT {m[1][0][1]}*(-1) / {determinant} from {covariance_table} WHERE id= "{m[1][0][0]}"'
            m11 = f'SELECT {m[1][1][1]} / {determinant} from {covariance_table} WHERE id= "{m[1][1][0]}"'
            sql_statement = self.sql_templates['insert_inverse_2_2'].render(
                m00=m00,
                m01=m01,
                m10=m10,
                m11=m11,
                features=self.model.x_columns,
                inverse_matrix=covariance_table + "_inverse_matrix",database=self.database
            )
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)

        else:
            # find matrix of cofactors
            cofactors = []
            for r in range(len(m)):
                cofactorRow = []
                for c in range(len(m)):
                    minor = self.__get_matrix_minor(m, r, c)
                    cofactorRow.append(((-1) ** (r + c)) * self.__get_matrix_determinante(minor, covariance_table))
                cofactors.append(cofactorRow)
            cofactors[:] = [[x / determinant for x in cofactor] for cofactor in cofactors]

            sql_statement = self.sql_templates['insert_inverse_n_n'].render(
                cofactor_matrix=cofactors,
                features=self.model.x_columns,
                inverse_matrix=covariance_table + "_inverse_matrix",database=self.database
            )
            logging.debug("SQL: " + str(sql_statement))
            self.db_connection.execute(sql_statement)
        self.__transpose_matrix(covariance_table + "_inverse_matrix")
        self.__rearrange_inverse(covariance_table)

    def __rearrange_inverse(self, covariance_table):
        for row in self.model.x_columns:
            for column in self.model.x_columns:
                sql_statement = self.sql_templates['select_x_from_where'].render(
                    x=column,
                    where_statement=row,
                    table=covariance_table + "_inverse_matrix",database=self.database
                )
                logging.debug("SQL: " + str(sql_statement))
                data = list(self.db_connection.execute_query(sql_statement)[0])

                sql_statement = self.sql_templates['insert_inverse'].render(
                    row=self.model.x_map[row],
                    column=self.model.x_map[column],
                    actual_value=data[0],
                    inverse_table=covariance_table + "_inverse",database=self.database
                )
                logging.debug("SQL: " + str(sql_statement))
                self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['drop_table'].render(
            table=covariance_table + "_inverse_matrix",database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

    def __get_mahalonibis_distance(self):
        logging.info("CALCULATING MAHALONOBIS DISTANCE")
        n = list(range(self.model.no_of_rows_prediction))
        sql_statement = self.sql_templates['insert_vector'].render(
            vector_table='gaussian_' + self.model.id + '_vector_',
            x_columns=self.model.x_columns,
            row_no=n,
            y_classes=self.model.y_classes,
            input_table='gaussian_' + self.model.id + '_covariance_input',database=self.database, target=self.model.y_column[0]
        )
        for statement in sqlparse.split(sql_statement):
            if statement:
                logging.debug("SQL: " + str(statement))
                self.db_connection.execute(statement)

        sql_statement = self.sql_templates['drop_table'].render(
            table='gaussian_' + self.model.id + '_estimation',database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['init_multi_gauss_prob_table'].render(
            table='gaussian_' + self.model.id + '_estimation',database=self.database)
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)

        sql_statement = self.sql_templates['calculate_mahalonobis_distance'].render(
            estimation_table='gaussian_' + self.model.id + '_estimation',
            row_no=n,
            y_classes=self.model.y_classes,
            vector_table='gaussian_' + self.model.id + '_vector_',
            covariance_matrix=f"gaussian_{self.model.id}_covariance_matrix",database=self.database

        )
        for statement in sqlparse.split(sql_statement):
            if statement:
                logging.debug("SQL: " + str(statement))
                self.db_connection.execute(statement)

    def __multivariate_probability(self):
        logging.info("CALCULATION MULTIVARIATE PROBABILITY")
        n = list(range(self.model.no_of_rows_prediction))
        sql_statement = self.sql_templates['calculate_multivariate_density'].render(
            table='gaussian_' + self.model.id + '_estimation',
            row_no=n,
            y_classes=self.model.y_classes,
            determinante_table='gaussian_' + self.model.id + "_determinante",
            vector_table='gaussian_' + self.model.id + '_vector_',
            covariance_matrix=f"gaussian_{self.model.id}_covariance_matrix",
            k=self.model.input_size,
            id='gaussian_' + self.model.id,database=self.database

        )
        for statement in sqlparse.split(sql_statement):
            if statement:
                logging.debug("SQL: " + str(statement))
                self.db_connection.execute(statement)

        sql_statement = self.sql_templates['p_y'].render(
            table='gaussian_' + self.model.id + '_estimation',
            y_column=self.model.y_column[0],
            y_classes=self.model.y_classes,
            input_table=self.model.input_table,
            row_no=self.model.no_of_rows_input,database=self.database)

        for statement in sqlparse.split(sql_statement):
            if statement:
                logging.debug("SQL: " + str(statement))
                self.db_connection.execute(statement)

        sql_statement = self.sql_templates['multiply_columns'].render(
            table='gaussian_' + self.model.id + '_estimation',
            column="probability",
            feature_1="p_y",
            feature_2="gaussian_distribution",database=self.database
        )
        logging.debug("SQL: " + str(sql_statement))
        self.db_connection.execute(sql_statement)
