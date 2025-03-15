import requests
import csv
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
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
import datetime
import whois
import tldextract
from ipwhois import IPWhois
from ipwhois.exceptions import ASNRegistryError
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Configuration (Defaults) ---
INPUT_FILE = 'fqdns.txt'
OUTPUT_FILE = 'fqdn_analysis.csv'
PROGRESS_FILE = 'fqdn_analysis_progress.txt'
MAX_WORKERS = 10
TIMEOUT = 0.2
KEYWORDS = [
    "login", "signin", "account", "verify", "secure", "update", "bank",
    "payment", "free", "download", "admin", "password", "credential"
]
RETRY_DELAY = 0.5
MAX_RETRIES = 1
UNKNOWN_VALUE = 2  # Use 2 for all N/A, NULL, Unknown, Timeout cases
RISKY_TLDS = ['.xyz', '.top', '.loan', '.online', '.club', '.click', '.icu', '.cn']

# --- Global Variables ---
processed_fqdns = set()
shutdown_event = False
console = Console()


# --- Signal Handler ---
def signal_handler(sig, frame):
    global shutdown_event
    console.print("\n[red]Received interrupt signal. Shutting down gracefully...[/red]")
    shutdown_event = True
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# --- Helper Functions ---

def load_progress():
    """Loads progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return set(f.read().splitlines())
    return set()

def save_progress(fqdn):
    """Saves progress to file."""
    with open(PROGRESS_FILE, 'a') as f:
        f.write(fqdn + '\n')

def resolve_dns(fqdn):
    """Resolves DNS records, handling timeouts."""
    results = {}
    for record_type in ['A', 'AAAA', 'MX', 'TXT']:
        try:
            resolver = dns.resolver.Resolver()
            resolver.timeout = TIMEOUT
            resolver.lifetime = TIMEOUT
            answers = resolver.resolve(fqdn, record_type)
            # Convert to 1 if records exist, otherwise leave as initialized
            results[record_type] = 1 if answers else 0
            results[f'{record_type}_Records'] = ','.join([str(rdata) for rdata in answers]) if answers else "" # Keep the records for later
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers, dns.exception.Timeout, dns.resolver.LifetimeTimeout):
            results[record_type] = UNKNOWN_VALUE # DNS resolution failed
            results[f'{record_type}_Records'] = ""
        except Exception as e:
            results[record_type] = UNKNOWN_VALUE
            results[f'{record_type}_Records'] = ""
            console.print(f"[yellow]DNS resolution error for {fqdn} ({record_type}): {e}[/yellow]")
    return results

def get_domain_age(fqdn):
    """Gets domain age (in days), returns 0/1/2."""
    try:
        w = whois.whois(fqdn)
        if isinstance(w.creation_date, list):
            creation_date = w.creation_date[0]
        else:
            creation_date = w.creation_date

        if creation_date:
            age = (datetime.datetime.now() - creation_date).days
            return 1 if age < 30 else 0  # 1 if "young" (bad), 0 if older
        else:
            return UNKNOWN_VALUE  # Couldn't determine age
    except Exception:
        return UNKNOWN_VALUE  # WHOIS failed


def get_certificate_expiry(cert):
    """Checks if certificate is expired, returns 0/1/2."""
    try:
        expiry_date_str = cert.get('notAfter', '')
        expiry_date = datetime.datetime.strptime(expiry_date_str, '%b %d %H:%M:%S %Y %Z')
        return 0 if datetime.datetime.now() < expiry_date else 1  # 0 if valid, 1 if expired

    except Exception:
        return UNKNOWN_VALUE  # Could not determine


def analyze_ip(ip_address):
    """Analyzes IP, returns dict with 0/1/2 values."""
    try:
        obj = IPWhois(ip_address)
        results = obj.lookup_rdap(depth=1)
        #  Return 0/1/2. We don't have a good/bad definition here, so use presence/absence of data.
        return {
            'ASN': 0 if results.get('asn') else UNKNOWN_VALUE,
            'ASN_Country_Code': 0 if results.get('asn_country_code') else UNKNOWN_VALUE,
            'ASN_Description': 0 if results.get('asn_description') else UNKNOWN_VALUE,
        }
    except (ASNRegistryError, ValueError, Exception):
        return {
            'ASN': UNKNOWN_VALUE,
            'ASN_Country_Code': UNKNOWN_VALUE,
            'ASN_Description': UNKNOWN_VALUE,
        }

def detect_keywords(fqdn, title):
    """Detects keywords, returns 0/1."""
    for keyword in KEYWORDS:
        if keyword in fqdn.lower() or (title and keyword in title.lower()):
            return 1  # Bad: keyword found
    return 0  # Good: no keywords

def determine_overall_status(results):
    """Determines overall status and Is_Bad using a scoring system."""
    score = 0

    if results.get('Certificate_Valid') == 1:  # Certificate is invalid
        score += 5
    if results.get('Has_Suspicious_Keywords') == 1:
        score += 5
    if results.get('Status_Code_OK') == 0:
        score += 4
    if results.get('Final_Protocol_HTTPS') == 0:
        score += 4
    if results.get('High_Redirects') == 1:
        score += 4
    if results.get('Domain_Age') == 1:  # Domain is young
        score += 4
    if results.get('TLD') == 1: # TLD is risky
        score += 3
    if results.get('HSTS') == 0:
        score += 3
    content_type = results.get('Content_Type')
    if content_type == 1:
      score += 4

    if score >= 10:
        status = "Suspicious"
        is_bad = 1  # Bad
    elif score >= 5:
        status = "Warning"
        is_bad = UNKNOWN_VALUE  # Unknown
    elif score > 0:
        status = "Caution"
        is_bad = UNKNOWN_VALUE  # Unknown
    else:
        status = "Good"
        is_bad = 0  # Good

    return status, is_bad


def analyze_fqdn(fqdn, default_is_bad):
    """Analyzes a single FQDN, all outputs 0/1/2."""

    if fqdn in processed_fqdns:
        return None

    results = {'FQDN': fqdn, 'Is_Bad': default_is_bad}
    dns_results = resolve_dns(fqdn)
    results.update(dns_results) # Includes both presence (0/1/2) and record strings.

    results['Has_A_Record'] = dns_results.get('A', UNKNOWN_VALUE)
    results['Has_AAAA_Record'] = dns_results.get('AAAA', UNKNOWN_VALUE)
    results['Has_MX_Record'] = dns_results.get('MX', UNKNOWN_VALUE)
    results['Has_TXT_Record'] = dns_results.get('TXT', UNKNOWN_VALUE)
    results['A_Records'] = dns_results.get('A_Records', "") # Keep for later
    results['AAAA_Records'] = dns_results.get('AAAA_Records', "")
    results['MX_Records'] = dns_results.get('MX_Records', "")
    results['TXT_Records'] = dns_results.get('TXT_Records', "")
    results['Final_URL_Known'] = UNKNOWN_VALUE  # Default
    results['Domain_Age'] = get_domain_age(fqdn) # Returns 0/1/2
    results['TLD'] = UNKNOWN_VALUE # Initialize
    extracted_tld = tldextract.extract(fqdn).suffix
    if extracted_tld:
      results['TLD'] = 1 if "." + extracted_tld in RISKY_TLDS else 0 # 1 if risky, 0 if not, 2 from init.
    results['Content_Type'] = UNKNOWN_VALUE
    results['Certificate_Issuer'] = UNKNOWN_VALUE # Initialize.

    ip_analysis_results = {}
    if results['Has_A_Record'] == 1: # Only analyze if A record exists
        for ip_address in results['A_Records'].split(','):
            ip_analysis_results[ip_address] = analyze_ip(ip_address)
    results['IP_Analysis'] = ip_analysis_results


    for attempt in range(MAX_RETRIES):
        try:
            session = requests.Session()
            session.max_redirects = 5
            url = f"http://{fqdn}"
            response = session.get(url, timeout=TIMEOUT, allow_redirects=True, verify=False)

            final_url = response.url
            parsed_url = urlparse(final_url)
            results['Final_URL_Known'] = 1  # We know the final URL now
            results['Final_Protocol_HTTPS'] = 1 if parsed_url.scheme == 'https' else 0
            results['Status_Code_OK'] = 1 if 200 <= response.status_code < 300 else 0  # 1 if good, 0 bad
            results['HTTP_to_HTTPS_Redirect'] = 1 if parsed_url.scheme == 'https' and response.history else 0
            results['Redirects'] = len(response.history) # Keep raw number for viz
            results['High_Redirects'] = 1 if len(response.history) > 3 else 0
            content_type = response.headers.get('Content-Type', '').lower()
            results['Content_Type'] = 1 if content_type != "" and 'html' not in content_type and 'text' not in content_type and 'json' not in content_type else 0 # 1 if suspicious


            if parsed_url.scheme == 'https':
                try:
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    with socket.create_connection((parsed_url.hostname, 443), timeout=TIMEOUT) as sock:
                        with ctx.wrap_socket(sock, server_hostname=parsed_url.hostname) as ssock:
                            cert = ssock.getpeercert()
                            results['Certificate_Valid'] = 1  # Assume valid, then check
                            results['Certificate_Expiry'] = get_certificate_expiry(cert)  # 0/1/2
                            #Issuer
                            for field in cert.get('issuer', []):
                                for key, value in field:
                                    if key == 'commonName':
                                        # Check if we extracted the common name
                                        results['Certificate_Issuer'] = 0 if value else UNKNOWN_VALUE
                                        break
                                else: continue
                                break
                            else: results['Certificate_Issuer'] = UNKNOWN_VALUE # Could not find Common Name
                    results['HSTS'] = 1 if 'strict-transport-security' in response.headers else 0

                    try:
                        with socket.create_connection((parsed_url.hostname, 443), timeout=TIMEOUT) as sock:
                            ctx = ssl.create_default_context()
                            with ctx.wrap_socket(sock, server_hostname=parsed_url.hostname) as ssock:
                                pass
                    except (ssl.SSLError, socket.timeout, OSError, ValueError):
                        results['Certificate_Valid'] = 1  # Now we know it's invalid

                except (socket.timeout, OSError, ValueError):
                    results.update({
                        'Certificate_Issuer': UNKNOWN_VALUE,
                        'Certificate_Valid': 1,
                        'Certificate_Expiry': UNKNOWN_VALUE,
                        'HSTS': UNKNOWN_VALUE
                    })
            else:
                results.update({
                    'Certificate_Issuer': UNKNOWN_VALUE,
                    'Certificate_Valid': 1,
                    'Certificate_Expiry': UNKNOWN_VALUE,
                    'HSTS': UNKNOWN_VALUE
                })

            if 200 <= response.status_code < 300:
                try:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.title.string.strip() if soup.title else "" # Check if there is a title
                    results['Has_Suspicious_Keywords'] = detect_keywords(fqdn, title) # returns 0/1

                except Exception:
                    results['Has_Suspicious_Keywords'] = UNKNOWN_VALUE
            else:
                results['Has_Suspicious_Keywords'] = UNKNOWN_VALUE  # Couldn't get content

            break  # Success, exit retry loop

        except requests.exceptions.RequestException as e:
            results.update({
                'Status_Code_OK': UNKNOWN_VALUE,
                'Final_URL_Known': UNKNOWN_VALUE,
                'Final_Protocol_HTTPS': UNKNOWN_VALUE,
                'HTTP_to_HTTPS_Redirect': UNKNOWN_VALUE,
                'Redirects': UNKNOWN_VALUE,
                'High_Redirects': UNKNOWN_VALUE,
                'HSTS': UNKNOWN_VALUE,
                'Certificate_Valid': UNKNOWN_VALUE,
                'Certificate_Issuer': UNKNOWN_VALUE,
                'Certificate_Expiry': UNKNOWN_VALUE,
                'Content_Type': UNKNOWN_VALUE,
                'Has_Suspicious_Keywords' : UNKNOWN_VALUE
            })
            if attempt < MAX_RETRIES - 1:
                console.print(f"[yellow]Request failed for {fqdn}: {e}. Retrying in {RETRY_DELAY} seconds...[/yellow]")
                time.sleep(RETRY_DELAY)
            else:
                console.print(f"[red]Max retries reached for {fqdn}.[/red]")

    # Final Is_Bad determination, respecting default if inconclusive
    status, is_bad = determine_overall_status(results)
    results['Overall_Status'] = status  # Keep string for viz

    # IMPORTANT: If analysis was inconclusive, use the *default* Is_Bad
    if results['Overall_Status'] == "Caution" or results['Overall_Status'] == "Warning":
          results['Is_Bad'] = default_is_bad if default_is_bad!= UNKNOWN_VALUE else UNKNOWN_VALUE
    else: # If Good or Suspicious
          results['Is_Bad'] = is_bad # 0 or 1

    return results


def process_batch(fqdns, default_is_bad):
    """Processes FQDNs concurrently."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
            transient=False
        ) as progress:
            task = progress.add_task("[cyan]Processing FQDNs[/cyan]", total=len(fqdns))
            futures = {executor.submit(analyze_fqdn, fqdn, default_is_bad): fqdn for fqdn in fqdns}

            for future in concurrent.futures.as_completed(futures):
                if shutdown_event:
                    break
                fqdn = futures[future]
                try:
                    result = future.result()
                    if result:
                        yield result
                        save_progress(fqdn)
                        processed_fqdns.add(fqdn)
                except Exception as e:
                    console.print(f"[red]Error processing {fqdn}: {e}[/red]")
                finally:
                    progress.update(task, advance=1)

def generate_visualizations(df):
    """Generates visualizations (handles numeric data)."""

    def create_pie_chart(df, column, title, value_map=None):
        if column in df.columns and not df[column].isnull().all():
            counts = df[column].value_counts().reset_index()
            counts.columns = [column, 'Count']
            # If a value map is provided, try to apply it
            if value_map:
              try:
                counts[column] = counts[column].map(value_map)
              except: # in case of mixed data types
                pass
            fig = px.pie(counts, values='Count', names=column, title=title,
                         color_discrete_sequence=px.colors.qualitative.Set1)
            return fig
        else:
            fig = go.Figure()
            fig.update_layout(title_text=f"{title} (Data Not Available)")
            return fig


    def create_bar_chart(df, column, title, color=None, value_map=None, angle=0, numeric=False):
      if column in df.columns and not df[column].isnull().all():
          if numeric:
            # If it's a numeric column, convert to numeric and handle errors.
            df[column] = pd.to_numeric(df[column], errors='coerce')
            counts = df[column].value_counts().nlargest(10).reset_index()
          else: # not numeric
            counts = df[column].value_counts().nlargest(10).reset_index()
          counts.columns = [column, 'Count']
          if value_map:
            try:
              counts[column] = counts[column].map(value_map)
            except:
              pass
          fig = px.bar(counts, x=column, y='Count', title=title,
                      color=color, color_discrete_sequence=px.colors.qualitative.Pastel1)
          fig.update_layout(xaxis_tickangle=angle)
          return fig
      else: # Data not available
          fig = go.Figure()
          fig.update_layout(title_text=f"{title} (Data Not Available)")
          return fig


    protocol_map = {0: 'HTTP', 1: 'HTTPS', 2: 'Unknown'}
    keyword_map = {0: 'No', 1: 'Yes', 2: 'Unknown'}
    is_bad_map = {0: 'Good', 1: 'Bad', 2: 'Unknown'}

    fig_status = create_pie_chart(df, 'Overall_Status', 'Distribution of Overall Status')
    fig_protocol = create_bar_chart(df, 'Final_Protocol_HTTPS', 'HTTP vs HTTPS', color='Final_Protocol_HTTPS', value_map=protocol_map)
    fig_redirects = create_bar_chart(df, 'Redirects', "Distribution of Redirects", numeric=True)
    fig_issuer = create_bar_chart(df, 'Certificate_Issuer', 'Top 10 Certificate Issuers', color='Certificate_Issuer', angle=-45) # Don't apply value map as values can be many
    fig_keywords = create_pie_chart(df, 'Has_Suspicious_Keywords', 'Suspicious Keywords Found', value_map=keyword_map)
    fig_is_bad = create_pie_chart(df, 'Is_Bad', 'Is Bad Distribution', value_map=is_bad_map)
    fig_domain_age =  create_bar_chart(df, 'Domain_Age', "Domain Age Distribution", numeric=True)
    fig_tld = create_bar_chart(df, 'TLD', 'Top 10 TLDs', color='TLD', angle=-45)

    # --- Combined Dashboard ---
    fig = make_subplots(
      rows=4, cols=2,
      specs=[[{"type": "pie"}, {"type": "bar"}],
              [{"type": "bar"}, {"type": "bar"}],
              [{"type": "pie"}, {"type": "pie"}],
              [{"type": "bar"}, {"type": "bar"}]],
      subplot_titles=("Overall Status", "HTTP vs HTTPS", "Redirects",
                      "Top 10 Certificate Issuers", "Suspicious Keywords", "Is Bad",
                      "Domain Age", "Top 10 TLDs"),
      vertical_spacing=0.1,
      horizontal_spacing=0.1
    )

    if fig_status and 'data' in fig_status and len(fig_status['data']) > 0:
        fig.add_trace(fig_status['data'][0], row=1, col=1)
    if fig_protocol and 'data' in fig_protocol and len(fig_protocol['data']) > 0:
        fig.add_trace(fig_protocol['data'][0], row=1, col=2)
    if fig_redirects and 'data' in fig_redirects and len(fig_redirects['data']) > 0:
        fig.add_trace(fig_redirects['data'][0], row=2, col=1)
    if fig_issuer and 'data' in fig_issuer and len(fig_issuer['data'])>0:
        fig.add_trace(fig_issuer['data'][0], row=2, col=2)
    if fig_keywords and 'data' in fig_keywords and len(fig_keywords['data']) > 0:
        fig.add_trace(fig_keywords['data'][0], row=3, col=1)
    if fig_is_bad and 'data' in fig_is_bad and len(fig_is_bad['data']) > 0:
        fig.add_trace(fig_is_bad['data'][0], row=3, col=2)
    if fig_domain_age and 'data' in fig_domain_age and len(fig_domain_age['data']) > 0:
        fig.add_trace(fig_domain_age['data'][0], row=4, col=1)
    if fig_tld and 'data' in fig_tld and len(fig_tld['data'])>0:
        fig.add_trace(fig_tld['data'][0], row=4, col=2)

    fig.update_layout(title_text="FQDN Analysis Dashboard", title_x=0.5, height=1200)
    fig.write_html("fqdn_analysis_dashboard.html")



def main():
    global processed_fqdns, MAX_WORKERS, TIMEOUT, INPUT_FILE, OUTPUT_FILE, PROGRESS_FILE, KEYWORDS, RISKY_TLDS

    parser = argparse.ArgumentParser(description="Analyze FQDNs (no external APIs).")
    parser.add_argument("-i", "--input", help="Input file with FQDNs", required=True)
    parser.add_argument("-o", "--output", help="Output CSV file", default=OUTPUT_FILE)
    parser.add_argument("-p", "--progress", help="Progress file", default=PROGRESS_FILE)
    parser.add_argument("-w", "--workers", type=int, help="Number of workers", default=MAX_WORKERS)
    parser.add_argument("-t", "--timeout", type=int, help="Timeout (seconds)", default=TIMEOUT)
    parser.add_argument("-k", "--keywords", help="Suspicious keywords", default=",".join(KEYWORDS))
    parser.add_argument("--risky-tlds", help="Risky TLDs", default=",".join(RISKY_TLDS))
    parser.add_argument("--is_bad", type=int, choices=[0, 1, 2], help="Force Is_Bad label (0=good, 1=bad, 2=unknown)")
    parser.add_argument("--no-visualizations", action="store_true", help="Disable visualizations")

    args = parser.parse_args()

    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    PROGRESS_FILE = args.progress
    MAX_WORKERS = args.workers
    TIMEOUT = args.timeout
    KEYWORDS = args.keywords.split(",")
    RISKY_TLDS = args.risky_tlds.split(",")
    default_is_bad = args.is_bad if args.is_bad is not None else UNKNOWN_VALUE

    processed_fqdns = load_progress()
    all_fqdns = []

    with open(INPUT_FILE, 'r') as infile:
        for line in infile:
            all_fqdns.append(line.strip())

    csv_header = [
      'FQDN', 'Is_Bad', 'Has_A_Record', 'Has_AAAA_Record', 'Has_MX_Record',
      'Has_TXT_Record', 'A_Records', 'AAAA_Records', 'MX_Records', 'TXT_Records',
      'Final_Protocol_HTTPS', 'Status_Code_OK', 'HTTP_to_HTTPS_Redirect',
      'High_Redirects', 'Redirects', 'HSTS', 'Certificate_Valid',
      'Has_Suspicious_Keywords', 'Final_URL_Known', 'Certificate_Issuer',
      'Certificate_Expiry', 'Domain_Age', 'TLD', 'Content_Type', 'IP_Analysis',
      'Overall_Status'
    ]

    file_exists = os.path.exists(OUTPUT_FILE)
    if file_exists:
        try:
            with open(OUTPUT_FILE, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_header = next(reader)
                header_matches = existing_header == csv_header
        except StopIteration:
            header_matches = False
    else:
        header_matches = False

    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        if not file_exists or not header_matches:
            writer.writerow(csv_header)

        for results in process_batch(all_fqdns, default_is_bad):
            if shutdown_event:
                break
            try:
                ip_analysis_str = ""
                for ip, analysis in results.get('IP_Analysis', {}).items():
                    #  Convert IP analysis to 0/2
                    asn = analysis.get('ASN', UNKNOWN_VALUE)
                    country = analysis.get('ASN_Country_Code', UNKNOWN_VALUE)
                    desc = analysis.get('ASN_Description', UNKNOWN_VALUE)
                    ip_analysis_str += f"{ip}: (ASN={asn}, Country={country}, Description={desc}); "


                writer.writerow([
                    results['FQDN'], results['Is_Bad'], results['Has_A_Record'],
                    results['Has_AAAA_Record'], results['Has_MX_Record'],
                    results['Has_TXT_Record'], results['A_Records'], results['AAAA_Records'],
                    results['MX_Records'], results['TXT_Records'],
                    results['Final_Protocol_HTTPS'], results['Status_Code_OK'],
                    results['HTTP_to_HTTPS_Redirect'], results['High_Redirects'],
                    results['Redirects'], results['HSTS'], results['Certificate_Valid'],
                    results['Has_Suspicious_Keywords'], results['Final_URL_Known'],
                    results['Certificate_Issuer'], results['Certificate_Expiry'],
                    results['Domain_Age'], results['TLD'], results['Content_Type'],
                    ip_analysis_str, results['Overall_Status']
                ])
            except Exception as e:
                console.print(f"[red]Error writing row: {e}[/red]")

    if shutdown_event:
        pass
    else:
        console.print("[green]Processing complete.[/green]")
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)

        if not args.no_visualizations:
            try:
                df = pd.read_csv(OUTPUT_FILE)
                # Convert relevant columns to numeric where appropriate.
                for col in ['Is_Bad','Has_A_Record','Has_AAAA_Record','Has_MX_Record','Has_TXT_Record','Final_Protocol_HTTPS','Status_Code_OK',
                            'HTTP_to_HTTPS_Redirect','High_Redirects','HSTS','Certificate_Valid','Has_Suspicious_Keywords',
                            'Final_URL_Known', 'Domain_Age', 'Redirects']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Ensure object columns are strings, which handles mixed types and N/A well.
                for col in ['Overall_Status', 'Certificate_Issuer', 'TLD', 'Content_Type', 'A_Records', 'AAAA_Records', 'MX_Records', 'TXT_Records','IP_Analysis']:
                    if col in df.columns:
                        df[col] = df[col].astype(str)

                generate_visualizations(df)
                console.print("[green]Visualizations generated: fqdn_analysis_dashboard.html[/green]")
            except Exception as e:
                console.print(f"[red]Error generating visualizations: {e}[/red]")

if __name__ == "__main__":
    main()