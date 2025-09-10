import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.robotparser as robotparser
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

from src.data.chunker import chunk_text
from src.data.embeddings import embed_texts
from src.data.store import get_collection
from support_deflect_bot.utils.settings import ALLOW_HOSTS, CRAWL_CACHE_PATH, TRUSTED_DOMAINS
from support_deflect_bot.utils.settings import USER_AGENT as UA

# Cache file for freshness (ETag/Last-Modified + content_hash)
CACHE_PATH = CRAWL_CACHE_PATH
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

# ---------------------- Utilities ----------------------


def _load_cache() -> Dict[str, Dict]:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Dict]):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, CACHE_PATH)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()


def _normalize_url(href: str, base: Optional[str] = None) -> Optional[str]:
    if not href:
        return None
    absu = urljoin(base, href) if base else href
    u = urlparse(absu)
    if u.scheme not in ("http", "https"):
        return None
    # drop fragment + query for stability (adjust if you need query-specific pages)
    u = u._replace(fragment="", query="")
    return u.geturl()


def _same_host(a: str, b: str) -> bool:
    return urlparse(a).netloc == urlparse(b).netloc


def _robots_ok(url: str) -> bool:
    # Check if domain is trusted (Option B: bypass robots.txt for trusted domains)
    parsed_url = urlparse(url)
    host = parsed_url.netloc
    if host in TRUSTED_DOMAINS:
        logging.info(f"Bypassing robots.txt check for trusted domain: {host}")
        return True
    
    # Option D: Fix robotparser remote fetching by using our own fetch_html
    host_url = f"{parsed_url.scheme}://{host}"
    robots_url = urljoin(host_url, "/robots.txt")
    
    try:
        # Use our own fetch_html function which has proper headers and error handling
        resp = fetch_html(robots_url, timeout=10)
        resp.raise_for_status()
        
        # Parse robots.txt content locally using robotparser
        import tempfile
        import os
        
        # Create a temporary file with robots.txt content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(resp.text)
            temp_path = f.name
        
        try:
            rp = robotparser.RobotFileParser()
            rp.set_url('file://' + temp_path)
            rp.read()
        finally:
            os.unlink(temp_path)
        
        return rp.can_fetch(UA, url)
    except (urllib.error.URLError, urllib.error.HTTPError, requests.RequestException) as e:
        # If robots.txt fails to load, be conservative and allow
        logging.info(f"Failed to fetch robots.txt from {robots_url}: {e}. Allowing crawl.")
        return True
    except Exception as e:
        logging.warning(f"Unexpected error checking robots.txt for {url}: {e}")
        return True


# ---------------------- HTML processing ----------------------


def _drop_noise(soup: BeautifulSoup):
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "aside", "footer"]):
        tag.decompose()
    noisy = soup.select(".sidebar, .toc, .sphinxsidebar, .related, .breadcrumbs")
    for t in noisy:
        t.decompose()


def html_to_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    _drop_noise(soup)

    lines: List[str] = []
    main = soup.find("main") or soup.find("div", {"role": "main"}) or soup.body or soup

    def append(s: str):
        s = (s or "").strip()
        if s:
            lines.append(s)

    for el in main.descendants:
        if isinstance(el, Tag):
            name = el.name.lower()
            if name in {"h1", "h2", "h3", "h4"}:
                append(f"# {el.get_text(' ', strip=True)}")
            elif name in {"p", "li"}:
                append(el.get_text(" ", strip=True))
            elif name in {"pre"}:
                txt = el.get_text("\n", strip=True)
                if txt:
                    append("```")
                    append(txt)
                    append("```")
            elif name == "code":
                txt = el.get_text("", strip=True)
                if txt:
                    append(f"`{txt}`")

    text = "\n\n".join(lines)
    return title, text


def extract_links(html: str, base_url: str) -> Set[str]:
    soup = BeautifulSoup(html, "lxml")
    links: Set[str] = set()
    for a in soup.find_all("a", href=True):
        url = _normalize_url(a["href"], base=base_url)
        if not url:
            continue
        host = urlparse(url).netloc
        if host in ALLOW_HOSTS:
            links.add(url)
    return links


# ---------------------- Fetching + Indexing ----------------------


def fetch_html(
    url: str,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    timeout: int = 20,
):
    headers = {"User-Agent": UA}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    resp = requests.get(url, headers=headers, timeout=timeout)
    return resp  # may be 200 or 304


def _index_single(url: str, html: str, title: str, text: str) -> int:
    """Delete existing docs for URL and add new chunks. Returns #chunks."""
    coll = get_collection()
    try:
        coll.delete(where={"path": url})
    except Exception:
        pass

    chunks = chunk_text(text, chunk_size=900, overlap=150)
    if not chunks:
        return 0

    vecs = embed_texts(chunks)
    ids = [f"{url}#{i}" for i in range(len(chunks))]

    host = urlparse(url).netloc
    metas = [
        {"path": url, "title": title, "chunk_id": i, "host": host}
        for i in range(len(chunks))
    ]

    coll.add(documents=chunks, embeddings=vecs, metadatas=metas, ids=ids)
    return len(chunks)


# index_urls(...)
def index_urls(urls: List[str], force: bool = False) -> Dict[str, Dict[str, int]]:
    cache = _load_cache()
    out = {}
    for url in urls:
        stats = {
            "fetched": 0,
            "chunks": 0,
            "replaced": 0,
            "errors": 0,
            "skipped_304": 0,
            "skipped_samehash": 0,
        }
        try:
            entry = cache.get(url, {})
            # ↓ conditional fetch only if NOT force
            resp = fetch_html(
                url,
                etag=None if force else entry.get("etag"),
                last_modified=None if force else entry.get("last_modified"),
            )
            if not force and resp.status_code == 304:
                stats["skipped_304"] = 1
                out[url] = stats
                continue

            resp.raise_for_status()
            html = resp.text
            title, text = html_to_text(html)
            content_hash = _sha256(text)

            # ↓ skip same-hash only if NOT force
            if not force and entry.get("content_hash") == content_hash:
                stats["skipped_samehash"] = 1
                out[url] = stats
                continue

            n = _index_single(url, html, title, text)
            stats["fetched"] = 1
            stats["chunks"] = n
            stats["replaced"] = 1

            cache[url] = {
                "etag": resp.headers.get("ETag") or resp.headers.get("Etag"),
                "last_modified": resp.headers.get("Last-Modified"),
                "content_hash": content_hash,
                "updated_at": int(time.time()),
            }
        except (requests.RequestException, ConnectionError, TimeoutError) as e:
            logging.error(f"Failed to fetch/index {url}: {e}")
            stats["errors"] += 1
        except Exception as e:
            logging.error(f"Unexpected error processing {url}: {e}")
            stats["errors"] += 1
        out[url] = stats
        time.sleep(0.3)
    _save_cache(cache)
    return out


def crawl_urls(
    seeds: List[str],
    depth: int = 1,
    max_pages: int = 40,
    same_domain: bool = True,
    force: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    BFS crawl limited by depth and max_pages. Respects ALLOW_HOSTS and robots.txt.
    Applies freshness cache (ETag/Last-Modified + content hash).
    """
    cache = _load_cache()
    visited: Set[str] = set()
    out: Dict[str, Dict[str, int]] = {}
    q = deque()

    # seed queue
    for s in seeds:
        ns = _normalize_url(s)
        if ns:
            q.append((ns, 0))

    while q and len(out) < max_pages:
        url, d = q.popleft()
        if url in visited:
            continue
        visited.add(url)

        host = urlparse(url).netloc
        if host not in ALLOW_HOSTS:
            continue
        if not _robots_ok(url):
            out[url] = {
                "fetched": 0,
                "chunks": 0,
                "replaced": 0,
                "errors": 0,
                "skipped_304": 0,
                "skipped_samehash": 0,
                "robots_blocked": 1,
            }
            continue

        stats = {
            "fetched": 0,
            "chunks": 0,
            "replaced": 0,
            "errors": 0,
            "skipped_304": 0,
            "skipped_samehash": 0,
        }
        try:
            entry = cache.get(url, {})
            # ↓ conditional fetch only if NOT force
            resp = fetch_html(
                url,
                etag=None if force else entry.get("etag"),
                last_modified=None if force else entry.get("last_modified"),
            )
            if not force and resp.status_code == 304:
                stats["skipped_304"] = 1
                out[url] = stats
                continue

            resp.raise_for_status()
            html = resp.text
            title, text = html_to_text(html)
            content_hash = _sha256(text)

            # ↓ skip same-hash only if NOT force
            if not force and entry.get("content_hash") == content_hash:
                stats["skipped_samehash"] = 1
                out[url] = stats
                # Still can extract links to continue crawl
            else:
                n = _index_single(url, html, title, text)
                stats["fetched"] = 1
                stats["chunks"] = n
                stats["replaced"] = 1
                cache[url] = {
                    "etag": resp.headers.get("ETag") or resp.headers.get("Etag"),
                    "last_modified": resp.headers.get("Last-Modified"),
                    "content_hash": content_hash,
                    "updated_at": int(time.time()),
                }

            out[url] = stats

            # Follow links if within depth
            if d < depth:
                for link in extract_links(html, url):
                    if same_domain and not _same_host(url, link):
                        continue
                    if link not in visited:
                        q.append((link, d + 1))

        except (requests.RequestException, ConnectionError, TimeoutError) as e:
            logging.error(f"Failed to crawl {url}: {e}")
            stats["errors"] += 1
            out[url] = stats
        except Exception as e:
            logging.error(f"Unexpected error crawling {url}: {e}")
            stats["errors"] += 1
            out[url] = stats

        time.sleep(0.3)  # polite crawl delay

    _save_cache(cache)
    return out
