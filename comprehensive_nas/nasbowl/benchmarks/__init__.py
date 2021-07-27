from .abstract_benchmark import *
# This check the amount of physical RAM installed, as somehow the process crashes if the system memory is small.
# mem_gigabytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3)

# from .nas201 import NAS201
# from nasbowl.benchmarks.nas_1.nasbench101 import NASBench101
