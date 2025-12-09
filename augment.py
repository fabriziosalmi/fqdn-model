import requests
import socket
import dns.resolver
import time
import ssl
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import concurrent.futures
import os
import signal
import sys
import argparse
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich.console import Console
import datetime
import tldextract
from functools import lru_cache, wraps
import logging
import requests
import socket
import dns.resolver
import time
import ssl
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import concurrent.futures
import os
import signal
import sys
import argparse
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich.console import Console
import datetime
import tldextract
from functools import lru_cache, wraps
import logging
from multiprocessing import Value
import json
import settings as conf

try:
    import whois
    WHOIS_ENABLED = True
except ImportError:
    WHOIS_ENABLED = False

# Use settings from settings.py
INPUT_FILE = conf.INPUT_FILE
OUTPUT_FILE = conf.OUTPUT_FILE
PROGRESS_FILE = conf.PROGRESS_FILE
MAX_WORKERS = conf.MAX_WORKERS
TIMEOUT = conf.TIMEOUT
UNKNOWN_VALUE = conf.UNKNOWN_VALUE
RISKY_TLDS = conf.RISKY_TLDS
MAX_REDIRECTS = conf.MAX_REDIRECTS
NEGATIVE_CACHE_TTL = conf.NEGATIVE_CACHE_TTL
POSITIVE_CACHE_TTL = conf.POSITIVE_CACHE_TTL
SHORTENER_DOMAINS = conf.SHORTENER_DOMAINS
MAX_DOMAIN_LENGTH = conf.MAX_DOMAIN_LENGTH
GOOD_DOMAIN_LENGTH = conf.GOOD_DOMAIN_LENGTH
MAX_HYPHENS = conf.MAX_HYPHENS
MAX_DIGITS = conf.MAX_DIGITS
MAX_SUBDOMAINS = conf.MAX_SUBDOMAINS
KEYWORDS = conf.KEYWORDS
DNS_RESOLVERS = conf.DNS_RESOLVERS
WHOIS_TIMEOUT = conf.WHOIS_TIMEOUT

processed_fqdns = set()
shutdown_event = False
console = Console()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def signal_handler(sig, frame):
    global shutdown_event
    console.print("\n[red]Received interrupt signal. Shutting down...[/red]")
    shutdown_event = True
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return set(f.read().splitlines())
        except Exception:
            logger.error("Error loading progress")
            return set()
    return set()

def save_progress(fqdn):
    try:
        with open(PROGRESS_FILE, 'a') as f:
            f.write(fqdn + '\n')
    except Exception:
        logger.error("Error saving progress")

def is_valid_ip(ip_address):
    try:
        socket.inet_pton(socket.AF_INET, ip_address)
        return True
    except socket.error:
        try:
            socket.inet_pton(socket.AF_INET6, ip_address)
            return True
        except socket.error:
            return False

def negative_cache(ttl):
    def decorator(func):
        cache = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                if time.time() - cache[key][0] < ttl:
                    return cache[key][1]
            result = func(*args, **kwargs)
            cache[key] = (time.time(), result)
            return result
        return wrapper
    return decorator

@lru_cache(maxsize=1024)
@negative_cache(ttl=NEGATIVE_CACHE_TTL)
def resolve_dns(fqdn):
    results = {
        'DNS_A_Record': UNKNOWN_VALUE,
        'DNS_AAAA_Record': UNKNOWN_VALUE,
        'DNS_MX_Record': UNKNOWN_VALUE,
        'DNS_TXT_Record': UNKNOWN_VALUE,
        'DNS_CNAME_Record': UNKNOWN_VALUE,
        'DNS_CNAME_Resolution': UNKNOWN_VALUE,
    }

    if is_valid_ip(fqdn):
        results['DNS_A_Record'] = 0 if ":" not in fqdn else UNKNOWN_VALUE
        results['DNS_AAAA_Record'] = 0 if ":" in fqdn else UNKNOWN_VALUE
        return results

    for record_type in ['A', 'AAAA', 'MX', 'TXT', 'CNAME']:
        resolved = False
        for resolver_ip in DNS_RESOLVERS:
            resolver = dns.resolver.Resolver(configure=False)
            resolver.nameservers = [resolver_ip]
            resolver.timeout = TIMEOUT
            resolver.lifetime = TIMEOUT
            try:
                record_present, _ = resolve_single_record(resolver, fqdn, record_type)
                results[record_type] = record_present
                if record_type == 'CNAME':
                    results['DNS_CNAME_Resolution'] = record_present
                resolved = True
                break
            except Exception:
                pass

        if not resolved:
            results[record_type] = UNKNOWN_VALUE
    return results

def resolve_single_record(resolver, fqdn, record_type, resolved_cnames=None):
    if resolved_cnames is None:
        resolved_cnames = set()

    if fqdn in resolved_cnames:
        logger.warning(f"Circular CNAME detected for {fqdn}.")
        return 1, ""

    try:
        answers = resolver.resolve(fqdn, record_type)
        if record_type == 'CNAME':
            for rdata in answers:
                target = str(rdata.target).rstrip('.')
                if target == fqdn:
                    logger.warning(f"CNAME points to itself: {fqdn}")
                    return 1, target
                resolved_cnames.add(fqdn)
                try:
                    resolve_single_record(resolver, target, 'A', resolved_cnames)
                    return 0, target
                except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers,
                        dns.exception.Timeout, dns.resolver.LifetimeTimeout):
                    return 1, target
        return 0, ""
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers):
        return 2, ""
    except (dns.exception.Timeout, dns.resolver.LifetimeTimeout):
        logger.debug(f"DNS resolution timeout ({record_type}) for {fqdn}")
        return 2, ""
    except Exception:
        logger.error(f"DNS resolution error ({record_type}) for {fqdn}")
        return 2, ""

def get_certificate_info(hostname):
    try:
        with socket.create_connection((hostname, 443), timeout=TIMEOUT) as sock:
            context = ssl.create_default_context()
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                expiry_date_str = cert.get('notAfter', '')
                expiry_date = datetime.datetime.strptime(expiry_date_str, '%b %d %H:%M:%S %Y %Z')
                return 0 if datetime.datetime.now() < expiry_date else 1
    except ssl.SSLCertVerificationError:
        logger.debug(f"SSL certificate verification error for {hostname}")
        return 2
    except Exception:
        logger.debug(f"Certificate retrieval error for {hostname}")
        return 2

def get_whois_info(domain):
    if not WHOIS_ENABLED:
        return {'WHOIS_Info_Available': 2, 'WHOIS_Age': UNKNOWN_VALUE}
    try:
        w = whois.whois(domain)
        if w.status is None:
            return {'WHOIS_Info_Available': 2, 'WHOIS_Age': UNKNOWN_VALUE}

        creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        expiration_date = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
        updated_date = w.updated_date[0] if isinstance(w.updated_date, list) else w.updated_date

        creation_date = creation_date if isinstance(creation_date, datetime.datetime) else None
        expiration_date = expiration_date if isinstance(expiration_date, datetime.datetime) else None
        updated_date = updated_date if isinstance(updated_date, datetime.datetime) else None

        whois_age = (datetime.datetime.now() - creation_date).days / 365.25 if creation_date else None

        # Convert dates to epoch timestamps or set to UNKNOWN_VALUE if None
        creation_timestamp = int(creation_date.timestamp()) if creation_date else UNKNOWN_VALUE
        expiration_timestamp = int(expiration_date.timestamp()) if expiration_date else UNKNOWN_VALUE
        updated_timestamp = int(updated_date.timestamp()) if updated_date else UNKNOWN_VALUE
        whois_age_value = int(whois_age) if whois_age is not None else UNKNOWN_VALUE

        return {
            'WHOIS_Creation_Date': creation_timestamp,
            'WHOIS_Expiration_Date': expiration_timestamp,
            'WHOIS_Updated_Date': updated_timestamp,
            'WHOIS_Age': whois_age_value,
            'WHOIS_Info_Available': 0
        }
    except whois.parser.PywhoisError:
        logger.debug(f"WHOIS parsing error for {domain}")
        return {'WHOIS_Info_Available': 2, 'WHOIS_Age': UNKNOWN_VALUE}
    except Exception:
        logger.warning(f"WHOIS lookup failed for {domain}")
        return {'WHOIS_Info_Available': 2, 'WHOIS_Age': UNKNOWN_VALUE}

def detect_keywords(fqdn, title, body_text):
    text = f"{fqdn} {title} {body_text}".lower()
    return 1 if any(keyword in text for keyword in KEYWORDS) else 0

def is_url_shortener(fqdn):
    domain = tldextract.extract(fqdn).domain
    return 1 if domain + '.' + tldextract.extract(fqdn).suffix in SHORTENER_DOMAINS else 0

def count_subdomains(fqdn):
    ext = tldextract.extract(fqdn)
    if not ext.subdomain:
        return 0
    # Normalize subdomain count by not counting 'www' as a separate subdomain
    subdomains = ext.subdomain.split('.')
    if subdomains and subdomains[0] == 'www':
        subdomains = subdomains[1:]
    subdomain_count = len(subdomains)
    return 0 if subdomain_count <= MAX_SUBDOMAINS else 1

def analyze_fqdn(fqdn, default_is_bad_numeric, whois_enabled):
    if fqdn in processed_fqdns:
        return None
        
    # Normalize domain by removing 'www.' prefix for consistent classification
    # Store original FQDN for display purposes
    original_fqdn = fqdn
    
    # Always normalize the domain for feature extraction and classification
    ext = tldextract.extract(fqdn)
    # Create a normalized version without www for consistent classification
    normalized_domain = f"{ext.domain}.{ext.suffix}"
    
    # Use the normalized domain for all feature extraction and classification
    # but keep track of the original for display purposes
    fqdn = normalized_domain
    
    results = {
        'FQDN': original_fqdn,  # Keep original FQDN for display
        'Overall_Score': default_is_bad_numeric,
        'Has_WWW': 1 if ext.subdomain.startswith('www') else 0,  # Track www prefix but don't use for classification
        'DNS_A_Record': UNKNOWN_VALUE,
        'DNS_AAAA_Record': UNKNOWN_VALUE,
        'DNS_MX_Record': UNKNOWN_VALUE,
        'DNS_TXT_Record': UNKNOWN_VALUE,
        'DNS_CNAME_Record': UNKNOWN_VALUE,
        'DNS_CNAME_Resolution': UNKNOWN_VALUE,
        'Certificate_Valid': UNKNOWN_VALUE,
        'Status_Code_OK': UNKNOWN_VALUE,
        'Final_Protocol_HTTPS': UNKNOWN_VALUE,
        'HTTP_to_HTTPS_Redirect': UNKNOWN_VALUE,
        'High_Redirects': UNKNOWN_VALUE,
        'HSTS_Present': UNKNOWN_VALUE,
        'Has_Suspicious_Keywords': UNKNOWN_VALUE,
        'SSL_Verification_Failed': UNKNOWN_VALUE,
        'Is_Risky_TLD': UNKNOWN_VALUE,
        'Domain_Length': UNKNOWN_VALUE,
        'Num_Hyphens': UNKNOWN_VALUE,
        'Num_Digits': UNKNOWN_VALUE,
        'Contains_IP_Address': UNKNOWN_VALUE,
        'URL_Shortener': UNKNOWN_VALUE,
        'Subdomain_Count': UNKNOWN_VALUE,
        'Title_Length': UNKNOWN_VALUE,
        'Body_Length': UNKNOWN_VALUE,
    }
    if whois_enabled:
        results.update({
            'WHOIS_Creation_Date': UNKNOWN_VALUE,
            'WHOIS_Expiration_Date': UNKNOWN_VALUE,
            'WHOIS_Updated_Date': UNKNOWN_VALUE,
            'WHOIS_Age': UNKNOWN_VALUE,
            'WHOIS_Info_Available': UNKNOWN_VALUE
        })

    dns_checks = resolve_dns(fqdn)
    results.update(dns_checks)

    results['Is_Risky_TLD'] = 1 if "." + tldextract.extract(fqdn).suffix in RISKY_TLDS else 0
    results['Domain_Length'] = 0 if len(fqdn) <= GOOD_DOMAIN_LENGTH else 1 if len(fqdn) <= MAX_DOMAIN_LENGTH else 1
    results['Num_Hyphens'] = 0 if fqdn.count('-') <= MAX_HYPHENS else 1
    results['Num_Digits'] = 0 if sum(c.isdigit() for c in fqdn) <= MAX_DIGITS else 1
    results['Contains_IP_Address'] = 1 if any(part.isdigit() and int(part) <= 255 for part in fqdn.split(".")) else 0
    results['URL_Shortener'] = is_url_shortener(fqdn)
    results['Subdomain_Count'] = count_subdomains(fqdn)

    try:
        with requests.Session() as session:
            session.max_redirects = MAX_REDIRECTS
            response = session.get(f"http://{fqdn}", timeout=TIMEOUT, allow_redirects=True, verify=True)
            final_url = response.url
            parsed_url = urlparse(final_url)

            results['Status_Code_OK'] = 0 if 200 <= response.status_code < 300 else 1
            results['Final_Protocol_HTTPS'] = 0 if parsed_url.scheme == 'https' else 1
            results['High_Redirects'] = 1 if len(response.history) > 3 else 0
            if response.history:
                initial_url = response.history[0].url
                if initial_url.startswith('http://') and final_url.startswith('https://'):
                    results['HTTP_to_HTTPS_Redirect'] = 0
                elif initial_url.startswith('https://') and final_url.startswith('http://'):
                    results['HTTP_to_HTTPS_Redirect'] = 1
                else:
                    results['HTTP_to_HTTPS_Redirect'] = 2
            else:
                results['HTTP_to_HTTPS_Redirect'] = 2

            results['HSTS_Present'] = 0 if 'strict-transport-security' in response.headers else 2

            if parsed_url.scheme == 'https':
                results['Certificate_Valid'] = get_certificate_info(parsed_url.hostname)
            else:
                results['Certificate_Valid'] = 2

            if 200 <= response.status_code < 300 and 'html' in response.headers.get('Content-Type', '').lower():
                try:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.title.string.strip() if soup.title and soup.title.string else ""
                    body_text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
                    results['Has_Suspicious_Keywords'] = detect_keywords(fqdn, title, body_text)
                    results['Title_Length'] = 0 if title else 2 # 0 if title exists, 2 if doesn't
                    results['Body_Length'] = 0 if body_text else 2  # 0 if body exists, 2 if it doesn't.
                except Exception:
                    logger.error(f"Error parsing HTML for {fqdn}")
                    results['Has_Suspicious_Keywords'] = UNKNOWN_VALUE
                    results['Title_Length'] = UNKNOWN_VALUE
                    results['Body_Length'] = UNKNOWN_VALUE

    except requests.exceptions.SSLError:
        logger.debug(f"SSL verification failed for {fqdn}")
        results['SSL_Verification_Failed'] = 1
        results['Certificate_Valid'] = 2
    except requests.exceptions.RequestException:
        logger.debug(f"Request failed for {fqdn}")
        results['Status_Code_OK'] = 2
        results['Final_Protocol_HTTPS'] = 2
        results['HTTP_to_HTTPS_Redirect'] = 2
        results['High_Redirects'] = 2
        results['HSTS_Present'] = 2
        results['Certificate_Valid'] = 2

    if whois_enabled:
        whois_info = get_whois_info(fqdn)
        results.update(whois_info)
        # WHOIS_Age is already set to UNKNOWN_VALUE if unavailable

    results['Overall_Score'] = calculate_overall_score(results, default_is_bad_numeric)
    return results

def calculate_overall_score(results, default_is_bad_numeric):
    if results['SSL_Verification_Failed'] == 1:
        return 1
    if results['Certificate_Valid'] == 1:
        return 1
    if results['Has_Suspicious_Keywords'] == 1:
        return 1

    bad_indicators = [
        'Is_Risky_TLD', 'Status_Code_OK', 'High_Redirects', 'Domain_Length',
        'Num_Hyphens', 'Num_Digits', 'Contains_IP_Address', 'URL_Shortener',
        'DNS_CNAME_Resolution'
    ]
    # Has_WWW is intentionally excluded from bad_indicators to ensure it doesn't affect classification
    if any(results[indicator] == 1 for indicator in bad_indicators):
        return 1

    unknown_indicators = [
        'DNS_A_Record', 'DNS_AAAA_Record', 'DNS_MX_Record', 'DNS_TXT_Record',
        'DNS_CNAME_Resolution',
        'Certificate_Valid', 'Status_Code_OK', 'Final_Protocol_HTTPS',
        'HTTP_to_HTTPS_Redirect', 'High_Redirects', 'HSTS_Present',
        'Has_Suspicious_Keywords', 'SSL_Verification_Failed', 'Subdomain_Count',
        'WHOIS_Info_Available', 'WHOIS_Age'
    ]
    if any(results[indicator] == 2 for indicator in unknown_indicators):
        return 2

    return 0

def process_batch(fqdns, default_is_bad_numeric, progress, task, whois_enabled, bytes_read, non_unknown_count):
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_fqdn, fqdn, default_is_bad_numeric, whois_enabled): fqdn for fqdn in fqdns}
        for future in concurrent.futures.as_completed(futures):
            if shutdown_event:
                return
            fqdn = futures[future]
            try:
                result = future.result()
                if result:
                    yield result
                    save_progress(fqdn)
                    processed_fqdns.add(fqdn)
                    bytes_read.value += len((fqdn + '\n').encode('utf-8'))
                    if result['Overall_Score'] != UNKNOWN_VALUE:
                        non_unknown_count.value += 1
            except Exception:
                logger.exception(f"Error processing {fqdn}")
            finally:
                progress.update(task, advance=1)

def main():
    global processed_fqdns, MAX_WORKERS, TIMEOUT, INPUT_FILE, OUTPUT_FILE, PROGRESS_FILE, KEYWORDS, RISKY_TLDS, DNS_RESOLVERS, WHOIS_TIMEOUT, POSITIVE_CACHE_TTL, NEGATIVE_CACHE_TTL, UNKNOWN_VALUE

    parser = argparse.ArgumentParser(description="Analyze FQDNs for security risks.")
    parser.add_argument("-i", "--input", help="Input file with FQDNs", default=INPUT_FILE)
    parser.add_argument("-o", "--output", help="Output JSON file", default=OUTPUT_FILE)
    parser.add_argument("-p", "--progress", help="Progress file", default=PROGRESS_FILE)
    parser.add_argument("-w", "--workers", type=int, help="Worker threads", default=MAX_WORKERS)
    parser.add_argument("-t", "--timeout", type=float, help="Request timeout", default=TIMEOUT)
    parser.add_argument("-k", "--keywords", help="Suspicious keywords", default=",".join(KEYWORDS))
    parser.add_argument("--risky-tlds", help="Risky TLDs", default=",".join(RISKY_TLDS))
    parser.add_argument("--dns-resolvers", help="DNS resolvers (comma-separated)", default=None)
    parser.add_argument("--is_bad", type=str, choices=["Yes", "No", "Unknown"],
                        help="Default Is_Bad (0=No, 1=Yes, 2=Unknown)", default="Unknown")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--whois", action="store_true", help="Enable WHOIS lookups (requires python-whois)")
    parser.add_argument("--whois-timeout", type=float, help="Timeout for WHOIS lookups", default=WHOIS_TIMEOUT)
    parser.add_argument("--positive-cache-ttl", type=int, help="Positive cache TTL in seconds", default=POSITIVE_CACHE_TTL)
    parser.add_argument("--negative-cache-ttl", type=int, help="Negative cache TTL in seconds", default=NEGATIVE_CACHE_TTL)
    parser.add_argument("--config", help="Path to configuration file", default="config.ini")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    if os.path.exists(args.config):
        config.read(args.config)

    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    PROGRESS_FILE = args.progress
    MAX_WORKERS = args.workers
    TIMEOUT = args.timeout
    KEYWORDS = args.keywords.split(",")
    RISKY_TLDS = args.risky_tlds.split(",")
    WHOIS_TIMEOUT = args.whois_timeout
    default_is_bad = args.is_bad
    POSITIVE_CACHE_TTL = args.positive_cache_ttl
    NEGATIVE_CACHE_TTL = args.negative_cache_ttl
    default_is_bad_numeric = {"Yes": 1, "No": 0, "Unknown": 2}.get(default_is_bad, 2)
    whois_enabled = args.whois and WHOIS_ENABLED

    if args.dns_resolvers:
        DNS_RESOLVERS = args.dns_resolvers.split(",")
        if not all(is_valid_ip(r) for r in DNS_RESOLVERS):
            console.print(f"[red]Invalid DNS resolver IP[/red]")
            sys.exit(1)
    elif config.has_option('DEFAULT', 'DNS_RESOLVERS'):
        DNS_RESOLVERS = config.get('DEFAULT', 'DNS_RESOLVERS').split(',')

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    processed_fqdns = load_progress()
    all_fqdns = []

    try:
        with open(INPUT_FILE, 'r') as infile:
            all_fqdns = [line.strip() for line in infile if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        console.print(f"[red]Error: Input file '{INPUT_FILE}' not found.[/red]")
        sys.exit(1)
    except Exception:
        logger.error("Error reading input file")
        sys.exit(1)

    total_file_size = os.path.getsize(INPUT_FILE)
    total_file_size_mb = total_file_size / (1024 * 1024)

    json_data = []

    fqdns_to_process = [fqdn for fqdn in all_fqdns if fqdn not in processed_fqdns]
    total_to_process = len(fqdns_to_process)
    bytes_read = Value('l', 0)
    non_unknown_count = Value('i', 0)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("Processed: {task.percentage:.1f}%"),
        TextColumn("MB: {task.fields[mb_processed]:.2f}/{task.fields[total_mb]:.2f}"),
        TextColumn("Non-Unknown: {task.fields[non_unknown_percent]:.1f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:

        task = progress.add_task("[cyan]Processing FQDNs[/cyan]", total=total_to_process,
                                 mb_processed=0, total_mb=total_file_size_mb, non_unknown_percent=0)

        for results in process_batch(fqdns_to_process, default_is_bad_numeric, progress, task, whois_enabled, bytes_read, non_unknown_count):
            if shutdown_event:
                break

            json_result = {}
            for key, value in results.items():
                # Special handling for WHOIS date fields: convert to epoch or keep as UNKNOWN_VALUE
                if key in ('WHOIS_Creation_Date', 'WHOIS_Expiration_Date', 'WHOIS_Updated_Date'):
                     json_result[key] = value
                # Keep FQDN as is
                elif key == 'FQDN':
                    json_result[key] = value
                # Ensure all other values are integers (0, 1, or 2)
                else:
                    try:
                        json_result[key] = int(value)  # Convert to integer
                    except (ValueError, TypeError):
                        json_result[key] = UNKNOWN_VALUE  # Default to UNKNOWN_VALUE on conversion failure

            json_data.append(json_result)


            mb_processed = bytes_read.value / (1024 * 1024)
            non_unknown_percent = (non_unknown_count.value / total_to_process) * 100 if total_to_process > 0 else 0.0
            progress.update(task, mb_processed=mb_processed, non_unknown_percent=non_unknown_percent, advance=1)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        json.dump(json_data, outfile, indent=4)


    if not shutdown_event:
        console.print("[green]Processing complete.[/green]")
        if os.path.exists(PROGRESS_FILE):
            try:
                os.remove(PROGRESS_FILE)
            except Exception:
                logger.error("Error removing progress file")
    else:
        console.print("[yellow]Processing interrupted. Progress saved.[/yellow]")

if __name__ == "__main__":
    main()