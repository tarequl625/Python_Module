import time

# -----------------------------
# gmtime() → returns UTC time as struct_time
# -----------------------------
print(time.gmtime())
# Example output (UTC):
# time.struct_time(tm_year=2025, tm_mon=8, tm_mday=29, tm_hour=12, tm_min=45, tm_sec=10,
#                  tm_wday=4, tm_yday=241, tm_isdst=0)

# -----------------------------
# localtime() → returns system local time as struct_time
# -----------------------------
print(time.localtime(time.time()))
# Example output (Bangladesh system local time, UTC+6):
# time.struct_time(tm_year=2025, tm_mon=8, tm_mday=29, tm_hour=18, tm_min=45, tm_sec=10,
#                  tm_wday=4, tm_yday=241, tm_isdst=0)

# -----------------------------
# time() → returns current Unix timestamp (seconds since 1970-01-01 UTC)
# -----------------------------
print(time.time())
# Example output:
# 1745928310.123456

# -----------------------------
# asctime() → human-readable string from struct_time (UTC)
# -----------------------------
print(time.asctime(time.gmtime(time.time())))
# Example output:
# Fri Aug 29 12:45:10 2025

# -----------------------------
# ctime() → human-readable string from localtime
# -----------------------------
print(time.ctime(time.time()))
# Example output:
# Fri Aug 29 18:45:10 2025

# -----------------------------
# Sleep example
# -----------------------------
print("Hello", end=" ")
time.sleep(5)  # pauses execution for 5 seconds
print("World")
# Output (waits 5 seconds):
# Hello World

# -----------------------------
# strftime formatting examples
# -----------------------------
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))
# Example output (UTC):
# Fri, 29 Aug 2025 12:45:10

print(time.strftime("%A, %D %B %Y %X %T", time.gmtime()))
# Example output:
# Friday, 08/29/25 August 2025 12:45:10 12:45:10

print(time.strftime("%x, %d %b %Y %I:%M:%S", time.gmtime()))
# Example output:
# 08/29/25, 29 Aug 2025 12:45:10

print(time.strftime("%c"))
# Example output (system local time):
# Fri Aug 29 18:45:10 2025

print(time.strftime("%r %z %Z"))
# Example output:
# 12:45:10 PM  +0000 UTC

print(time.strftime("%r, %V, %W, %w"))
# Example output:
# 12:45:10 PM, 35, 34, 5
# %V → ISO week number
# %W → Week number (Monday first)
# %w → Weekday number (Sunday=0)

print(time.strftime("%r, %u, %U"))
# Example output:
# 12:45:10 PM, 5, 34
# %u → ISO weekday (Monday=1)
# %U → Week number (Sunday first)

# -----------------------------
# strptime() → parse string into struct_time
# -----------------------------
print(time.strptime("Tue, 03 Aug 2021 10:45:08","%a, %d %b %Y %H:%M:%S"))
# Example output:
# time.struct_time(tm_year=2021, tm_mon=8, tm_mday=3, tm_hour=10, tm_min=45, tm_sec=8,
#                  tm_wday=1, tm_yday=215, tm_isdst=-1)