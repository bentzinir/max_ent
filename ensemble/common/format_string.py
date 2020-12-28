def format_string(s):
    from decimal import Decimal
    x = [('%.2E' % Decimal(x)) for x in s]
    return ', '.join(x)
