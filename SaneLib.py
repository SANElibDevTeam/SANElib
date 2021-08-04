from DataBase import DataBase
from Mdh import Mdh

class SaneLib:
    def __init__(self, connection):
        self.db = DataBase(connection)
    def mdh(self, model_id):
        return Mdh(self, model_id, )
