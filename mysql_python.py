import os
import mysql.connector

auth_id = '123'

host = os.environ['DATABASE_HOST']
user = os.environ['DATABASE_USER']
password = os.environ['DATABASE_PASS']
database_name = os.environ['DATABASE_NAME']
conn = mysql.connector.connect(
    host=host,
    port=3306,
    user=user,
    password=password,
    database=database_name
)
cur = conn.cursor()
stmt = "SELECT key_name FROM table_name WHERE value_name = %s"
cur.execute(stmt, (auth_id, ))
got_value = cur.fetchone()[0]
print(got_value)