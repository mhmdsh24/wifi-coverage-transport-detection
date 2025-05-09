content = '''from django import template

register = template.Library()

@register.simple_tag
def hello_world():
    return "Hello from a simple tag!"
'''

with open('visualization/templatetags/simple_tags.py', 'w', encoding='utf-8') as f:
    f.write(content)
    
print("Created simple_tags.py") 