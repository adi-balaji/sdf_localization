import csv
import json
import os

CSV_FILE = '/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/partial_results.csv'
JSON_FILE = '/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/Rt_error_add_test_partial_results.json'

with open(JSON_FILE) as f:
    data = json.load(f)

for k, v in data.items():
    R_errs = v["R_err"]
    t_errs = v["t_err"]
    add_errs = v["ADD_err"]

    #rounded to 4 decimal places
    avg_R_err = round(sum(R_errs) / len(R_errs), 4)
    avg_t_err = round(sum(t_errs) / len(t_errs), 4)
    avg_add_err = round(sum(add_errs) / len(add_errs), 5)

    with open(CSV_FILE, mode='a') as f:
        writer = csv.writer(f)
        writer.writerow([k, avg_R_err, avg_t_err, avg_add_err])
