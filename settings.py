import os
import configparser

# Load config.ini if it exists
config = configparser.ConfigParser()
CONFIG_FILE = 'config.ini'
if os.path.exists(CONFIG_FILE):
    config.read(CONFIG_FILE)

# Default Defaults
def get_setting(key, default, cast_type=str):
    val = config.get('DEFAULT', key, fallback=None)
    if val is None:
        return default
    try:
        return cast_type(val)
    except (ValueError, TypeError):
        return default

def get_list_setting(key, default):
    val = config.get('DEFAULT', key, fallback=None)
    if val is None:
        return default
    return [x.strip() for x in val.split(',')]

# File Paths
INPUT_FILE = get_setting('INPUT_FILE', 'fqdns.txt')
OUTPUT_FILE = get_setting('OUTPUT_FILE', 'fqdn_analysis.json')
PROGRESS_FILE = get_setting('PROGRESS_FILE', 'fqdn_analysis_progress.txt')
MODEL_DIR = get_setting('MODEL_DIR', 'models')

# Analysis Settings
MAX_WORKERS = get_setting('MAX_WORKERS', 32, int)
TIMEOUT = get_setting('TIMEOUT', 1.0, float)
WHOIS_TIMEOUT = get_setting('WHOIS_TIMEOUT', 2.0, float)
ANALYSIS_TIMEOUT = get_setting('ANALYSIS_TIMEOUT', 30, int) # New in P2, consolidated from predict.py

# Feature Extraction Settings
UNKNOWN_VALUE = get_setting('UNKNOWN_VALUE', 2, int)
MAX_REDIRECTS = get_setting('MAX_REDIRECTS', 5, int)
MAX_DOMAIN_LENGTH = get_setting('MAX_DOMAIN_LENGTH', 63, int)
GOOD_DOMAIN_LENGTH = get_setting('GOOD_DOMAIN_LENGTH', 20, int)
MAX_HYPHENS = get_setting('MAX_HYPHENS', 1, int)
MAX_DIGITS = get_setting('MAX_DIGITS', 4, int)
MAX_SUBDOMAINS = get_setting('MAX_SUBDOMAINS', 3, int)
MAX_FQDN_LENGTH = get_setting('MAX_FQDN_LENGTH', 253, int)

# Cache Settings
NEGATIVE_CACHE_TTL = get_setting('NEGATIVE_CACHE_TTL', 30, int)
POSITIVE_CACHE_TTL = get_setting('POSITIVE_CACHE_TTL', 3600, int)

# Lists
RISKY_TLDS = get_list_setting('RISKY_TLDS', ['.xyz', '.top', '.loan', '.online', '.club', '.click', '.icu', '.cn'])
SHORTENER_DOMAINS = get_list_setting('SHORTENER_DOMAINS', ['bit.ly', 't.co', 'tinyurl.com', 'ow.ly', 'is.gd', 'buff.ly', 'adf.ly'])
KEYWORDS = get_list_setting('KEYWORDS', ['login', 'signin', 'account', 'verify', 'secure', 'update', 'bank', 'payment', 'free', 'download', 'admin', 'password', 'credential'])
DNS_RESOLVERS = get_list_setting('DNS_RESOLVERS', ['1.1.1.1', '1.0.0.1', '8.8.8.8', '8.8.4.4', '9.9.9.9'])

# API Settings
MAX_BATCH_SIZE = get_setting('MAX_BATCH_SIZE', 100, int)
MAX_RETRIES = get_setting('MAX_RETRIES', 3, int)
RETRY_DELAY = get_setting('RETRY_DELAY', 1, int)
