import psycopg2
import numpy as np
import pandas as pd
import csv

A = np.random.randint(0, 1, (1000, 597540))
A = np.float64(A)

with open('input_double.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(A)

B = np.float32(A)

with open('input_float.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(B)