from decimal import Decimal
import numpy as np


def format_string(s):
    s = np.reshape(s, -1).tolist()
    x = [('%.2E' % Decimal(x)) for x in s]
    return ', '.join(x)
