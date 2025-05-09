from django import template

register = template.Library()

@register.filter(name='replace')
def replace(value, arg):
    """Replace all instances of arg with a space"""
    if value is None:
        return ''
    return value.replace(arg, ' ')

@register.filter(name='cut')
def cut(value, arg):
    """Removes all values of arg from the given string"""
    if value is None:
        return ''
    return value.replace(arg, '')

@register.filter(name='slice')
def slice_filter(value, arg):
    """Returns a slice of the string"""
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

content = '''from django import template

register = template.Library()

@register.filter(name='replace')
def replace(value, arg):
    """Replace all instances of arg with a space"""
    if value is None:
        return ''
    return value.replace(arg, ' ')

@register.filter(name='cut')
def cut(value, arg):
    """Removes all values of arg from the given string"""
    if value is None:
        return ''
    return value.replace(arg, '')

@register.filter(name='slice')
def slice_filter(value, arg):
    """Returns a slice of the string"""
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
'''

with open('visualization/templatetags/filters.py', 'w', encoding='utf-8') as f:
    f.write(content)
    
print("Created filters.py") 