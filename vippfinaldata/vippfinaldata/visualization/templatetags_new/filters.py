from django import template

register = template.Library()

@register.filter
def floatformat_or_na(value, arg=2):
    """Format a float or return N/A if value is None"""
    if value is None:
        return "N/A"
    try:
        return float(value).__format__('.{}f'.format(int(arg)))
    except (ValueError, TypeError):
        return "N/A"

@register.filter(name='replace')
def replace(value, arg):
    """
    Replace all instances of the first argument with the second argument.
    Example usage: 
    - {{ "hello_world"|replace:"_":" " }}  # Simple replacement
    - {{ "hello_world"|replace:"_,!" }}    # Replace with another character
    """
    if value is None:
        return ''
    
    if ',' in arg:
        # Two arguments provided: replace arg[0] with arg[1]
        args = arg.split(',')
        return value.replace(args[0], args[1])
    else:
        # One argument provided: replace with space by default
        return value.replace(arg, ' ')

@register.filter(name='cut')
def cut(value, arg):
    """Removes all values of arg from the given string"""
    if value is None:
        return ''
    return value.replace(arg, '')

@register.filter(name='slice')
def slice_filter(value, arg):
    """
    Returns a slice of the string.
    Uses the same syntax as Python's list slicing.
    """
    if value is None:
        return ''
    try:
        bits = []
        for x in arg.split(':'):
            if x: 
                bits.append(int(x))
            else:
                bits.append(None)
        return value[slice(*bits)]
    except (ValueError, TypeError):
        return value 