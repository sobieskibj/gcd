from ..templates import *

import logging
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

def make_config(config_name):
    if config_name == 'chexpert224_autoenc':
        log.info(f'Using config: {config_name}')
        return chexpert224_autoenc()
    else:
        raise NotImplementedError('Invalid config name or option not implemented yet')