
from lib.kmeans.sql_templates import SqlTemplates
from lib.kmeans.sqlite_templates import SqliteTemplates


def get_templates(driver_name):
    """
    This function returns the correct template implementations depending on the used driver.
    """
    
    if driver_name == "sqlite":
        return SqliteTemplates()
    else:
        return SqlTemplates()
