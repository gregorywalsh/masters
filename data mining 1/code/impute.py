import DBConnect as db
# 0. connect to the database(call DBConnect)
connection = db.get_connect()

# 1. get the data(from the raw_Master FDM)
raw_master_FDM_cursor = db.execute_query()
raw_master_FDM = raw_master_FDM_cursor.iterate()

"""
set missing value as -1
"""

