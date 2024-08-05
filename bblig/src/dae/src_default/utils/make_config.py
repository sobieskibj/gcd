from ..templates import *

import logging
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

def make_config(config_name):
    if config_name == 'ffhq256_autoenc':
        log.info(f'Using config: {config_name}')
        return ffhq256_autoenc()
    elif config_name == 'celeba128_autoenc':
        log.info(f'Using config: {config_name}')
        return celeba128_autoenc()
    else:
        raise NotImplementedError('Invalid config name or option not implemented yet')