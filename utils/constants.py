import os
import json
from string import ascii_uppercase as ABC

# CATEGORIES as defined by OSPAR
_path_categories = os.path.join(os.getcwd(), 'utils/categories.json')
f = open(_path_categories)
CATEGORIES = json.load(f)