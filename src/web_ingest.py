import hashlib
import json
import os
import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Set
from urllib.parse import urlparse, urljoin
import urllib.robotparser as robotparser

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from src.chunker import chunk_text
from src.embeddings import embed_texts
from src.store import get_collection

UA = "SupportDeflectBot/0.1 (+https://example.local; contact: you@example.com)"  # be polite

# Cache file for freshness (ETag/Last-Modified + content_hash)
CACHE_PATH = "./data/crawl_cache.json"
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

# Allowlist of hostnames we intend to crawl
ALLOW_HOSTS = {
    "docs.python.org",
    "packaging.python.org",
    "pip.pypa.io",
    "virtualenv.pypa.io",
}

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
    # Basic robots.txt check per host
    host = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(host, "/robots.txt"))
    try:
        rp.read()
        return rp.can_fetch(UA, url)
    except Exception:
        # If robots fails to load, be conservative and allow
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

def fetch_html(url: str, etag: Optional[str] = None, last_modified: Optional[str] = None, timeout: int = 20):
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
    metas = [{"path": url, "title": title, "chunk_id": i} for i in range(len(chunks))]
    coll.add(documents=chunks, embeddings=vecs, metadatas=metas, ids=ids)
    return len(chunks)

def index_urls(urls: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Original one-shot indexer (no link follow). Kept for convenience.
    """
    cache = _load_cache()
    out: Dict[str, Dict[str, int]] = {}
    for url in urls:
        stats = {"fetched": 0, "chunks": 0, "replaced": 0, "errors": 0, "skipped_304": 0, "skipped_samehash": 0}
        try:
            # Conditional fetch
            entry = cache.get(url, {})
            resp = fetch_html(url, etag=entry.get("etag"), last_modified=entry.get("last_modified"))
            if resp.status_code == 304:
                stats["skipped_304"] = 1
                out[url] = stats
                continue

            resp.raise_for_status()
            html = resp.text
            title, text = html_to_text(html)
            content_hash = _sha256(text)

            if entry.get("content_hash") == content_hash:
                stats["skipped_samehash"] = 1
                out[url] = stats
                continue

            n = _index_single(url, html, title, text)
            stats["fetched"] = 1
            stats["chunks"] = n
            stats["replaced"] = 1

            # Update cache
            cache[url] = {
                "etag": resp.headers.get("ETag") or resp.headers.get("Etag"),
                "last_modified": resp.headers.get("Last-Modified"),
                "content_hash": content_hash,
                "updated_at": int(time.time()),
            }
        except Exception:
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
            out[url] = {"fetched": 0, "chunks": 0, "replaced": 0, "errors": 0, "skipped_304": 0, "skipped_samehash": 0, "robots_blocked": 1}
            continue

        stats = {"fetched": 0, "chunks": 0, "replaced": 0, "errors": 0, "skipped_304": 0, "skipped_samehash": 0}
        try:
            entry = cache.get(url, {})
            resp = fetch_html(url, etag=entry.get("etag"), last_modified=entry.get("last_modified"))
            if resp.status_code == 304:
                stats["skipped_304"] = 1
                out[url] = stats
                continue

            resp.raise_for_status()
            html = resp.text
            title, text = html_to_text(html)
            content_hash = _sha256(text)

            if entry.get("content_hash") == content_hash:
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

        except Exception:
            stats["errors"] += 1
            out[url] = stats

        time.sleep(0.3)  # polite crawl delay

    _save_cache(cache)
    return out
