from .nas_201_api import NASBench201API
from .nas_301_api import NASBench301API

APIMapping = {
    "nasbench201": NASBench201API,
    "nasbench301": NASBench301API,
}
