import sys, os
sys.path.append(os.pardir)

from common.functions import softmax, cross_entrophy_error
from common.graident import numerical_gradient