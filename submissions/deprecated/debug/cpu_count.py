import os

print(len(os.sched_getaffinity(0)))
print(os.cpu_count())