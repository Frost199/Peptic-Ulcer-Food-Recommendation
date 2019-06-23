import sqlite3

connection = sqlite3.connect('Food_Logs.db')
cursor = connection.cursor()

create_table = "CREATE TABLE IF NOT EXISTS food_logs (id INTEGER PRIMARY KEY," \
               " food text," \
               " food_group)"
connection.execute(create_table)
connection.commit()
connection.close()
