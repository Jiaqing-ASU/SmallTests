import psycopg2
import numpy as np
import pandas as pd

t_host = "localhost"
t_port = "5432"
t_dbname = "postgres"
t_user = "postgres"
t_pw = "postgres"
db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
db_cursor = db_conn.cursor()

try:
    db_cursor.execute("CREATE TABLE imdb(id int, array_data bytea);")
except psycopg2.Error as e:
    print(e)

A = np.random.randint(0, 1, (100, 1009375))

try:
    for i in range(100):
        a = A[i]
        a_bytes = a.tobytes()
        # execute the INSERT statement
        db_cursor.execute("INSERT INTO imdb(id,array_data) " +
                    "VALUES(%s,%s)",(i, a_bytes))
        # commit the changes to the database
        db_conn.commit()
    # close the communication with the PostgresQL database
    db_cursor.close()
except(Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if db_conn is not None:
        db_conn.close()
