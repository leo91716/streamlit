
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

d = datetime.strptime('2020-06', '%Y-%m')
print(d)
print(d + relativedelta(months=+1))
print(d + relativedelta(years=+1))


# from datetime import date, datetime
# from dateutil.relativedelta import relativedelta


# # ✅ add months to a date

# date_1 = date(2023, 6, 24)
# print(date_1)  # 👉️ 2023-06-24

# result_1 = date_1 + relativedelta(months=+3)
# print(result_1)  # 👉️ 2023-09-24

# # ----------------------------------------

# # ✅ add months to current date

# date_2 = date.today()
# print(date_2)  # 👉️ 2022-06-20

# result_2 = date_2 + relativedelta(months=+2)
# print(result_2)  # 👉️ 2022-08-20

# # ----------------------------------------

# # ✅ add months to date (using datetime object)

# my_str = '09-24-2023'  # 👉️ (mm-dd-yyyy)

# date_3 = datetime.strptime(my_str, '%m-%d-%Y')
# print(date_3)  # 👉️ 2023-09-24 00:00:00


# result_3 = date_3 + relativedelta(months=+2)
# print(result_3)  # 👉️ 2023-11-24 00:00:00

# # ----------------------------------------

# # ✅ add months to current date (using datetime object)

# date_4 = datetime.today()
# print(date_4)  # 👉️ 2022-06-20 10:48:05.892375

# result_4 = date_4 + relativedelta(months=+3)
# print(result_4)  # 👉️ 2022-09-20 10:48:05.892375