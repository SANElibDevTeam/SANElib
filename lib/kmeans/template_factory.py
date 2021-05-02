
from lib.kmeans.sql_templates import SqlTemplates
from lib.kmeans.sqlite_templates import SqliteTemplates


# factory method
def get_templates(driver_name):
    if driver_name== "sqlite":
        return SqliteTemplates()
    else:
        return SqlTemplates()
