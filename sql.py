import pymysql

def new_db(name, connection = conn):
    """
    ---What it does---
        + Creates a new_db in a SQLserver via Py.
    ---What it needs---
        + name: new_db name.    
    ---What it returns---
        + new_db: new DB in the SQLserver.
    """
    query = "CREATE database " + name + " ;" 
    connection.cursor.execute(query)
    connection.commit()   

def new_table(name, connection = conn, df, columns, types, null, keys):   
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

def deconnection(connection = conn):
    """
    ---What it does---
        + Closes the connection between Py & SQL server.
    """
    connection.close() 