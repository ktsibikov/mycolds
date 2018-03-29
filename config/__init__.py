import os

from config.load_configuration import LoadConfiguration

if not 'CONFIG' in globals():
    global CONFIG
    CONFIG = LoadConfiguration()