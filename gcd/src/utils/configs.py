import os
from pathlib import Path

import logging
log = logging.getLogger(__name__)

def extract_output_dir_path_from_config(config):
    date = '/'.join(list(config._metadata.resolver_cache['now'].values()))
    if config._content['mode'] == 'MULTIRUN':
        top_dir = 'MULTIRUN'
        output_dir_partial = Path.cwd() / top_dir / date
        # Search for subdirs and choose the last one created
        sub_dirs = [p for p in output_dir_partial.iterdir() if p.is_dir()]
        sub_dirs.sort(key = lambda x: os.path.getmtime(x))
        output_dir = output_dir_partial / sub_dirs[-1]
    else:
        output_dir = Path.cwd() / 'outputs' / date
    log.info(f'Output directory: {output_dir}')
    return output_dir