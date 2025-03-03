import functools
import os
import re
import time
import logging
from concurrent.futures import ProcessPoolExecutor
import utils

# Configure logging 🪓
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# the data - it's in utils, otherwise it's overwhelming to have on one page
dems = utils.get_dems()
reps = utils.get_repubs()
last_names = utils.get_congress_118_lastnames()
state_names = utils.get_state_names()
extra_stop_phrases = utils.get_stop_phrases()
freedom_caucus = utils.get_freedom_caucus()
root_dir = utils.get_data_directory()

# Precompile regex to save time
congress_last_names, state_regex, extra_phrases_regex = utils.precompile_regex_patterns(
    last_names, state_names, extra_stop_phrases
)

freedom_caucus_lower = {name.lower() for name in freedom_caucus}


def standardize_speaker_names(speaker_info):
    """ Standardizes speaker names to match official records. """
    all_official_names = dems + reps
    clean_name = re.sub(r'[\s()\d]+', '', speaker_info.lower())

    base_name_mapping = {re.sub(r'[A-Z]{2}$', '', name).lower(): name for name in all_official_names}
    name_mapping = {name.lower(): name for name in all_official_names}

    for name in all_official_names:
        clean_variation = re.sub(r'[^a-z]', '', name.lower().replace('-', '').replace('_', ''))
        name_mapping[clean_variation] = name

    if clean_name in name_mapping:
        return name_mapping[clean_name]

    base_name = re.sub(r'[^a-z]', '', clean_name)
    if base_name in name_mapping:
        return name_mapping[base_name]

    logger.warning(f"Could not find standardized name for {speaker_info}, using {clean_name}")
    return clean_name


def process_single_file(file_path, multilabel=True):
    """ Process a single file and return processed content & metadata. """
    basename = os.path.basename(file_path)
    match = re.match(r'^(.*?)-([DRdr])\.txt$', basename)

    if not match:
        return None

    speaker_info, party = match.groups()
    party = party.upper()

    if party == 'I':
        return None

    speaker_matching = standardize_speaker_names(speaker_info)
    label = 'democrat' if party == 'D' else 'republican'

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None

    content = state_regex.sub('', content, count=1)
    content = extra_phrases_regex.sub('', content)

    if len(content.split()) < 8:
        return None

    if party == 'R' and multilabel and speaker_matching.lower() in freedom_caucus_lower:
        label = 'freedom caucus'

    return {'text': content, 'label': label, 'speaker': speaker_matching}


def load_data_parallel(multilabel=True, num_workers=None):
    """ Load and process files in parallel. """
    file_paths = [os.path.join(root, f) for root, _, files in os.walk(root_dir) for f in files if f.endswith('.txt')]
    process_func = functools.partial(process_single_file, multilabel=multilabel)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_func, file_paths))

    results = [r for r in results if r is not None]
    return [r['text'] for r in results], [r['label'] for r in results], [r['speaker'] for r in results]


def fix_file_names(name, party, dry_run=True):
    """
    Fixes misnamed files by checking their contents and renaming them accordingly.

    :param root_dir: Root directory to search for files.
    :param name: The MC's name.
    :param party: The
    :param dry_run: If True, only prints what would be renamed.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if name.lower() in filename.lower():
                file_path = os.path.join(dirpath, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        file.read()
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
                    return

                number_suffix = re.search(r"\d+", filename)
                number_suffix = number_suffix.group() if number_suffix else ""

                new_filename = f"{name}{number_suffix}_{party}.txt"
                new_path = os.path.join(dirpath, new_filename)

                try:
                    if dry_run:
                        logger.info(f"Would rename: {file_path} -> {new_path}")
                    else:
                        os.rename(file_path, new_path)
                        logger.info(f"Renamed: {file_path} -> {new_path}")
                except Exception as e:
                    logger.error(f"Unexpected error renaming {file_path}: {e}")







