import pymysql
 
def new_table(name, df, columns, types, null, keys, connection):   
    """
    ---What it does---
        + Creates a new table in the DB, from an existing DF.
    ---What it needs---
        + tname: name of the new table, as string.
        + columns:
        + types:
        + null:
        + keys: 
    ---What it returns---
        + new_db: new DB in the SQLserver.   
    """
    le = len(keys)
    query = "DROP TABLE IF EXISTS " + name + " CREATE TABLE " + name + " ("
    for i in range(df.shape[1]):
        query += str(df.columns[i] + " " + types[i] + " " + null[i]) + ", "
    if len(keys) >= 1:
        query += " PRIMARY KEY "
        for j in range(le):
            query += str(keys[j]) + ", "
    query = query[:-2] + ");"
    connection.cursor.execute(query)
    connection.commit()

def insert_data(name_table, df, connection):   
    """
    ---What it does---
        + Creates a new table in the DB, from an existing DF.
    ---What it needs---
        + name: name of the new table, as string.
        + columns:
        + types:
        + null:
        + keys: 
    ---What it returns---
        + new_db: new DB in the SQLserver.   
    """
    query = "INSERT INTO " + name_table + " VALUES "

    for i in range(df.shape[0]): 
        query += "("
        for j in range(df.shape[1]): 
            query += str(df[df.columns.values[j]][i]) + ", "
        query = query[:-2] + "), " 
    query = query[:-2] + ";"
    connection.cursor.execute(query)
    connection.commit()   

def deconnection(connection):
    """
    ---What it does---
        + Closes the connection between Py & SQL server.
    """
    connection.close() 