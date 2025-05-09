from django import template
from django.template.defaultfilters import floatformat
import json
import os

register = template.Library()

@register.filter
def as_percentage(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.1f}%"
    except (ValueError, TypeError):
        return value

@register.filter
def format_rssi(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.1f} dBm"
    except (ValueError, TypeError):
        return value

@register.filter
def format_dict_value(dictionary, key):
    """Get a value from a dictionary by key"""
    if not dictionary or key is None:
        return ""
    return dictionary.get(key, "")

@register.filter
def as_json(value):
    """Format a value as JSON for javascript use"""
    return json.dumps(value)

@register.filter
def get_item(dictionary, key):
    """Get a value from a dictionary by key"""
    if not dictionary or key is None:
        return None
    return dictionary.get(key)

@register.filter
def dictkey(dictionary, key):
    """Get a value from a dictionary by key"""
    if not dictionary or key is None:
        return None
    return dictionary.get(key)

@register.filter
def filename(path):
    """Extract just the filename from a path"""
    if not path:
        return ""
    return os.path.basename(path)

@register.filter
def percentage(value, total):
    """Calculate percentage of value out of total"""
    if not value or not total or total == 0:
        return 0
    try:
        return round((float(value) / float(total)) * 100, 1)
    except (ValueError, TypeError):
        return 0

@register.filter
def multiply(value, arg):
    """Multiply the value by the argument"""
    if not value:
        return 0
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0 