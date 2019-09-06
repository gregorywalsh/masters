import pymysql.cursors
import pandas as pd
from pandas import DataFrame


def execute_query(connection, sql, insert_data=None):

    """
    ARGS
    connection      a pymysql.connect connection object
    query           a SQL statement as a string
    insert_data     a tuple of string strings to inject into the SQL string using % format

    RETURNS
    an iterable cursor
    """

    with connection.cursor() as cursor:
        if insert_data:
            cursor.execute(sql, insert_data)
        else:
            cursor.execute(sql)
        return cursor


def get_connection(user, password, database, port=3306):

    """
    ARGS
    user        name of mysql user account
    password    account password

    RETURNS
    a pymysql connection object
    """

    return pymysql.connect(host='localhost',
                           user=user,
                           password=password,
                           db=database,
                           charset='utf8mb4',
                           port=port,
                           cursorclass=pymysql.cursors.DictCursor)
