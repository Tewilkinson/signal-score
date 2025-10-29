# app.py — SignalScore Content (SEO Suite)

# CSR auto-render (requests-html → Pyppeteer fallback) + PSI-first UX scoring + OpenAI keyword relevance

import re
import os
import json
import math
import html
import time
import urllib.parse
from collections import Counter
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup, Comment
import tldextract
from dateutil import parser as dateparser
import streamlit as st
import pandas as pd

# Charts
import plotly.express as px

# Optional JS rendering (fast path)
try:
    from requests_html import HTMLSession
    HAS_REQ_HTML = True
except Exception:
    HAS_REQ_HTML = False

# Deep JS rendering fallback (bulletproof)
PYPP_OK = False
try:
    from pyppeteer import launch
    PYPP_OK = True
except Exception:
    PYPP_OK = False

# Streamlit often runs an active asyncio loop; allow nested awaits when needed
try:
    import nest_asyncio  # type: ignore
    nest_asyncio.apply()
except Exception:
    pass

# Optional OpenAI embeddings
OPENAI_OK = False
_openai_client = None
EMBED_MODEL = "text-embedding-3-small"

try:
    from openai import OpenAI
    if st.secrets.get("OPENAI_API_KEY"):
        _openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        OPENAI_OK = True
    elif os.getenv("OPENAI_API_KEY"):
        _openai_client = OpenAI()
        OPENAI_OK = True
except Exception:
    try:
        import openai as _openai_legacy
        if st.secrets.get("OPENAI_API_KEY"):
            _openai_legacy.api_key = st.secrets["OPENAI_API_KEY"]
            _openai_client = _openai_legacy
            OPENAI_OK = True
        elif os.getenv("OPENAI_API_KEY"):
            _openai_client = _openai_legacy
            OPENAI_OK = True
    except Exception:
        OPENAI_OK = False
        _openai_client = None

# -----------------------------
# Config / Constants
# -----------------------------
st.set_page_config(page_title="SignalScore Content", page_icon="📶", layout="wide")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
ABS_TIMEOUT = 25

DATE_META_KEYS = [
    ("meta", {"property": "article:published_time"}),
    ("meta", {"name": "pubdate"}),
    ("meta", {"name": "publish_date"}),
    ("meta", {"itemprop": "datePublished"}),
    ("time", {"itemprop": "datePublished"}),
]
MODIFIED_META_KEYS = [
    ("meta", {"property": "article:modified_time"}),
    ("meta", {"name": "lastmod"}),
    ("meta", {"itemprop": "dateModified"}),
    ("time", {"itemprop": "dateModified"}),
]
AUTHOR_META_KEYS = [
    ("meta", {"name": "author"}),
    ("a", {"rel": "author"}),
    ("span", {"class": re.compile(r"author", re.I)}),
    ("div", {"class": re.compile(r"author", re.I)}),
]

NAV_LIKE = re.compile(
    r"(nav|menu|breadcrumb|footer|toc|table-of-contents|sidebar|aside|"
    r"pagination|pager|next-prev|share|social|subscribe|cookie|banner|"
    r"newsletter|promo|advert|ad-)",
    re.I
)

YEAR_IN_URL = re.compile(r"/(19|20)\d{2}(/|-)", re.I)
DATE_IN_URL = re.compile(r"/(19|20)\d{2}/(0?[1-9]|1[0-2])/", re.I)

GENERIC_ANCHORS = {
    "click here","read more","learn more","here","this","link","more",
    "see more","details","visit","check this","go","find out more"
}

STOPWORDS = set("""
a an and the for with that this from your you are was were have has not but can will our into over under
more most make made being been them they their there here very just about when where which also than then onto those these such
to of in on by as at be it is or if so do did done we i me my us our yours theirs its it's
""".split())

# -----------------------------
# Helpers
# -----------------------------
def absolute_url(href: str, base: str) -> str:
    try:
        return urllib.parse.urljoin(base, href)
    except Exception:
        return href

def clean_text(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def get_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return ".".join([p for p in [ext.domain, ext.suffix] if p])

def is_internal(href: str, base_domain: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(href)
        if not parsed.netloc:
            return True
        ext = tldextract.extract(parsed.netloc)
        dom = ".".join([p for p in [ext.domain, ext.suffix] if p])
        return dom == base_domain
    except Exception:
        return False

def try_parse_date(value: str):
    try:
        dt = dateparser.parse(value)
        if dt and not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def months_since(dt: datetime) -> float:
    if not dt:
        return math.inf
    now = datetime.now(timezone.utc)
    delta_days = (now - dt).days
    return delta_days / 30.44

def estimate_syllables(word: str) -> int:
    w = re.sub(r"[^a-zà-öø-ÿ]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouyà-äæè-ëì-ïò-öø-ü]+", w)
    count = max(1, len(groups))
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def flesch_reading_ease(text: str) -> float:
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not words or not sentences:
        return 0.0
    word_count = len(words)
    sent_count = max(1, len(sentences))
    syllables = sum(estimate_syllables(w) for w in words)
    fre = 206.835 - 1.015 * (word_count / sent_count) - 84.6 * (syllables / word_count)
    return max(0.0, min(100.0, fre))

def top_terms(text: str, n=20):
    tokens = [t.lower() for t in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]{3,}", text)]
    tokens = [t for t in tokens if t not in STOPWORDS]
    cnt = Counter(tokens)
    return cnt.most_common(n)

def token_set_from_url_path(url: str):
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    parts = re.split(r"[\/\-\_\.\+]+", path)
    parts = [p for p in parts if p and not re.match(r"^\d+$", p) and p not in STOPWORDS]
    return set(parts)

def keyword_stuffing_risk(text: str) -> float:
    tokens = [t.lower() for t in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]{3,}", text)]
    total = len(tokens)
    if total < 120:
        return 0.0
    cnt = Counter(tokens)
    _, freq = cnt.most_common(1)[0]
    ratio = freq / total
    if ratio <= 0.03: return 0.0
    if ratio >= 0.10: return 1.0
    return (ratio - 0.03) / (0.10 - 0.03)

# -------- CSR Rendering: requests-html (fast path) --------
def render_with_requests_html(url: str, timeout=ABS_TIMEOUT, scrolldown=6) -> str:
    if not HAS_REQ_HTML:
        raise RuntimeError("requests-html not installed")
    s = HTMLSession()
    r = s.get(url, headers=HEADERS, timeout=timeout)
    r.html.render(
        timeout=timeout * 1000,
        sleep=1.5,
        scrolldown=scrolldown,
        reload=False,
        keep_page=False
    )
    return r.html.html

# -------- CSR Rendering: Pyppeteer (deep fallback) --------
async def _pyppeteer_render(url: str, timeout=40, scroll_steps=8, wait_selectors=None) -> str:
    if wait_selectors is None:
        wait_selectors = [
            ".aem-Grid", ".article", "article", "main",
            "[data-cmp-hook-richtext='text']", ".content", ".post-content", ".entry-content",
            "[role='main']", ".prose", ".markdown-body"
        ]
    browser = await launch(
        headless=True,
        args=["--no-sandbox","--disable-dev-shm-usage","--disable-gpu","--single-process"]
    )
    try:
        page = await browser.newPage()
        await page.setUserAgent(HEADERS["User-Agent"])
        # Enrich headers to avoid geo/cookie walls and bot flags
        await page.setExtraHTTPHeaders({
            "Accept": HEADERS.get("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
            "Accept-Language": "en-US,en;q=0.9",
            "Upgrade-Insecure-Requests": "1",
            "DNT": "1",
        })
        await page.setViewport({"width": 1366, "height": 900, "deviceScaleFactor": 1})
        await page.goto(url, {"waitUntil": "networkidle2", "timeout": timeout*1000})

        # Gentle scroll to trigger lazy loads
        for _ in range(scroll_steps):
            await page.evaluate("window.scrollBy(0, document.body.scrollHeight / 6);")
            await page.waitForTimeout(600)

        # Try to accept cookie/consent dialogs commonly seen (e.g., Snowflake)
        try:
            # Click buttons with common accept text
            accept_btns = [
                "button:has-text('Accept All')",
                "button:has-text('Accept all')",
                "button:has-text('I Accept')",
                "button:has-text('Agree')",
                "#onetrust-accept-btn-handler",
                "button#truste-consent-button",
            ]
            for sel in accept_btns:
                try:
                    await page.click(sel, {"delay": 50})
                    await page.waitForTimeout(500)
                    break
                except Exception:
                    continue
        except Exception:
            pass

        # Wait for any known content selector
        found = False
        for sel in wait_selectors:
            try:
                await page.waitForSelector(sel, {"timeout": 4000, "visible": True})
                found = True
                break
            except Exception:
                continue

        # One extra idle wait to let late hydration finish
        await page.waitForTimeout(1200)

        content = await page.content()
        return content
    finally:
        try:
            await browser.close()
        except Exception:
            pass

def render_with_pyppeteer(url: str, timeout=40, wait_selectors=None) -> str:
    if not PYPP_OK:
        raise RuntimeError("pyppeteer not installed")
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # With nest_asyncio applied above, we can safely run until complete
            return loop.run_until_complete(_pyppeteer_render(url, timeout=timeout, scroll_steps=10, wait_selectors=wait_selectors))
        else:
            return loop.run_until_complete(_pyppeteer_render(url, timeout=timeout, scroll_steps=10, wait_selectors=wait_selectors))
    except RuntimeError:
        # No current event loop; use asyncio.run
        return asyncio.run(_pyppeteer_render(url, timeout=timeout, scroll_steps=10, wait_selectors=wait_selectors))

def fetch_html(url: str, use_js: bool, wait_selectors: list[str] | None = None):
    """
    Returns (html_text, fetch_ms, html_bytes, did_js, renderer)
    Auto-upgrades to JS render if first pass looks like CSR shell (tiny body / no content).
    Tries requests-html first; if still looks empty and pyppeteer is available, uses pyppeteer.
    """
    import time as _t
    did_js = False
    renderer = "plain"

    def _measure(text, t0):
        ms = int((_t.time() - t0) * 1000)
        size = len((text or "").encode("utf-8", errors="ignore"))
        return ms, size

    # Pass 1: plain GET
    t0 = _t.time()
    r = requests.get(url, headers=HEADERS, timeout=ABS_TIMEOUT)
    r.raise_for_status()
    html_text = r.text
    fetch_ms, html_bytes = _measure(html_text, t0)
    soup_probe = BeautifulSoup(html_text, "html.parser")
    body_text = (soup_probe.body.get_text(" ", strip=True) if soup_probe.body else "")
    wordish = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", body_text))
    has_mainish_tag = bool(soup_probe.find(["article","main"]))
    looks_csr_shell = (html_bytes < 60_000) or (wordish < 160) or (not has_mainish_tag)

    def _looks_empty(htm):
        sp = BeautifulSoup(htm or "", "html.parser")
        bt = (sp.body.get_text(" ", strip=True) if sp.body else "")
        wd = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", bt))
        return wd < 120

    # Pass 2: requests-html render
    if use_js or looks_csr_shell:
        try:
            did_js = True
            renderer = "requests-html"
            t1 = _t.time()
            html_text = render_with_requests_html(url, timeout=max(ABS_TIMEOUT, 35), scrolldown=8)
            fetch_ms, html_bytes = _measure(html_text, t1)
        except Exception:
            did_js = False
            renderer = "plain"

    # Pass 3: pyppeteer fallback if still empty
    if (not html_text) or _looks_empty(html_text):
        if PYPP_OK:
            try:
                did_js = True
                renderer = "pyppeteer"
                t2 = _t.time()
                html_text = render_with_pyppeteer(url, timeout=55, wait_selectors=wait_selectors)
                fetch_ms, html_bytes = _measure(html_text, t2)
            except Exception:
                # keep whatever we have
                pass

    return html_text, fetch_ms, html_bytes, did_js, renderer

def extract_ldjson(soup: BeautifulSoup):
    blobs = []
    for tag in soup.find_all("script", type="application/ld+json"):
        text = (tag.string or tag.get_text() or "").strip()
        if not text:
            continue
        try:
            blobs.append(json.loads(text))
        except Exception:
            try:
                text = re.sub(r",\s*}", "}", text)
                text = re.sub(r",\s*]", "]", text)
                blobs.append(json.loads(text))
            except Exception:
                continue
    return blobs

def extract_main_content(soup: BeautifulSoup):
    candidates = soup.find_all(["article", "main"])
    if candidates:
        best = max(candidates, key=lambda c: len(c.get_text(" ", strip=True)))
        return best
    body = soup.body or soup
    # strip obvious nav/footer/aside
    for tag in body.find_all(["nav","footer","aside"]):
        tag.extract()
    for tag in body.find_all(True, class_=NAV_LIKE):
        tag.extract()
    for c in body.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    return body

def is_in_navigation_context(tag) -> bool:
    for parent in tag.parents:
        name = getattr(parent, "name", None)
        if not name:
            continue
        if name in {"article", "main"} or (parent.get("role") == "main") or (parent.get("itemprop") == "articleBody"):
            return False
        if name in {"nav", "footer", "aside"}:
            return True
        if name == "header":
            gp = getattr(parent, "parent", None)
            if gp is not None and getattr(gp, "name", None) == "body":
                return True
        cls = " ".join(parent.get("class") or [])
        pid = parent.get("id") or ""
        if NAV_LIKE.search(cls) or NAV_LIKE.search(pid):
            return True
    return False

def visible_anchor_text(a):
    txt = clean_text(a.get_text(" ", strip=True))
    if txt:
        return txt
    txt = clean_text(a.get("title") or a.get("aria-label") or "")
    if txt:
        return txt
    img = a.find("img")
    if img and img.get("alt"):
        return clean_text(img.get("alt"))
    return ""

def extract_text_blocks(node):
    texts = []
    for tag in node.find_all(["p","li","h1","h2","h3","h4","h5","h6","blockquote","figcaption","pre","td","th"]):
        t = clean_text(tag.get_text(" ", strip=True))
        if t:
            texts.append(t)
    return texts

# ------------- PageSpeed Insights (CWV) -------------
def fetch_pagespeed(url: str, api_key: str | None):
    base = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    params = {"url": url, "strategy": "mobile", "category": "PERFORMANCE"}
    if api_key:
        params["key"] = api_key.strip()
    try:
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {"ok": False}

    out = {"ok": True}
    try:
        lh = data.get("lighthouseResult", {})
        cats = lh.get("categories", {})
        out["perf_score"] = cats.get("performance", {}).get("score", None)
        audits = lh.get("audits", {})
        def val(id_):
            v = audits.get(id_, {}).get("numericValue")
            return float(v) if v is not None else None
        out["lab"] = {
            "FCP_ms": val("first-contentful-paint"),
            "LCP_ms": val("largest-contentful-paint"),
            "CLS": audits.get("cumulative-layout-shift", {}).get("numericValue"),
            "SI_ms": val("speed-index"),
            "TTI_ms": val("interactive"),
            "TBT_ms": val("total-blocking-time"),
        }
    except Exception:
        pass

    try:
        le = data.get("loadingExperience", {})
        metrics = le.get("metrics", {})
        def p75(id_):
            m = metrics.get(id_, {})
            p = m.get("percentile")
            return float(p) if p is not None else None
        out["field"] = {
            "LCP_ms_p75": p75("LARGEST_CONTENTFUL_PAINT_MS"),
            "CLS_p75": metrics.get("CUMULATIVE_LAYOUT_SHIFT_SCORE", {}).get("percentile", None),
            "INP_ms_p75": p75("INTERACTION_TO_NEXT_PAINT"),
        }
    except Exception:
        pass
    return out

def normalize_ms(value, good, poor):
    if value is None:
        return None
    if value <= good:
        return 1.0
    if value >= poor:
        return 0.0
    return max(0.0, min(1.0, (poor - value) / (poor - good)))

# ------------- OpenAI Embeddings & Relevance -------------
def cosine_sim(u, v):
    import math as _m
    if not u or not v:
        return 0.0
    s = sum(a*b for a,b in zip(u,v))
    nu = _m.sqrt(sum(a*a for a in u))
    nv = _m.sqrt(sum(b*b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return s/(nu*nv)

def embed_texts(texts: list[str]) -> list[list[float]] | None:
    if not OPENAI_OK or _openai_client is None or not texts:
        return None
    try:
        if "OpenAI" in str(type(_openai_client)):
            resp = _openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        else:
            resp = _openai_client.Embedding.create(model=EMBED_MODEL, input=texts)
            return [d["embedding"] for d in resp["data"]]
    except Exception:
        return None

def compute_keyword_relevance(full_text: str, h1: str, h2s: list[str], keywords_raw: str):
    raw = keywords_raw or ""
    if not raw.strip():
        return pd.DataFrame(columns=["Keyword","Matches","Density","LexicalScore","SemanticSim","Relevance","Recommendation"]), None
    parts = [p.strip() for p in re.split(r"[\n,]+", raw) if p.strip()]
    seen=set(); keywords=[]
    for p in parts:
        if p.lower() not in seen:
            keywords.append(p)
            seen.add(p.lower())

    text = (h1 or "") + " " + " ".join(h2s or []) + " " + (full_text or "")
    text_low = text.lower()
    total_words = max(1, len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text_low)))

    rows = []
    semantic_sims = {}
    page_for_embed = text[:8000] if len(text) > 8000 else text
    page_vec = None
    if OPENAI_OK:
        vecs = embed_texts([page_for_embed] + keywords)
        if vecs and len(vecs) == (1 + len(keywords)):
            page_vec = vecs[0]
            for i, kw in enumerate(keywords, start=1):
                semantic_sims[kw] = cosine_sim(page_vec, vecs[i])

    for kw in keywords:
        kw_low = kw.lower()
        matches = len(re.findall(r"\b" + re.escape(kw_low) + r"\b", text_low))
        density = matches / total_words
        lexical = 0.0
        if matches > 0:
            lexical = min(1.0, 0.6 + min(0.4, math.log1p(matches)/3.0))
        sem = semantic_sims.get(kw) if semantic_sims else None
        rel = (0.5*min(1.0, max(0.0, sem)) + 0.5*lexical) if sem is not None else lexical
        if rel >= 0.8: rec = "Strong coverage."
        elif rel >= 0.5: rec = "Deepen coverage with examples, entities, and subheadings."
        else: rec = "Add a focused section + internal links + supporting media."
        rows.append([kw, matches, round(density,6), round(lexical,2), (round(sem,3) if sem is not None else None), round(rel,2), rec])

    df = pd.DataFrame(rows, columns=["Keyword","Matches","Density","LexicalScore","SemanticSim","Relevance","Recommendation"])
    agg = float(df["Relevance"].mean()) if not df.empty else None
    return df, agg

# -----------------------------
# Extraction & Analysis
# -----------------------------
def analyze_url(url: str, use_js=False, exclude_toc=True, require_nonempty_anchor=True, want_psi=False, psi_key=None, keywords_raw:str="", wait_selector_input: str | None = None):
    wait_selectors = None
    if wait_selector_input and wait_selector_input.strip():
        # Support comma/line separated selectors
        wait_selectors = [s.strip() for s in re.split(r"[\n,]+", wait_selector_input) if s.strip()]
    else:
        # Add a Snowflake-friendly default hint
        wait_selectors = [
            ".markdown-body",
            "article",
            "main article",
            "main [class*='content']",
            "main [data-component]",
        ]
    html_text, fetch_ms, html_bytes, did_js, renderer = fetch_html(url, use_js, wait_selectors=wait_selectors)
    soup = BeautifulSoup(html_text, "html.parser")

    # If content still looks empty, try <noscript> fallback
    def _noscript_fallback(soup_):
        nos = soup_.find_all("noscript")
        texts = []
        for n in nos:
            inner = BeautifulSoup(n.decode_contents() or "", "html.parser")
            texts.append(inner.get_text(" ", strip=True))
        return " ".join(t for t in texts if t)

    probe_text = (soup.body.get_text(" ", strip=True) if soup.body else "")
    if len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", probe_text)) < 80:
        ns_text = _noscript_fallback(soup)
        if ns_text and len(ns_text.split()) > 50:
            body = soup.body or soup
            body.append(BeautifulSoup(f"<div data-noscript-fallback='1'>{ns_text}</div>", "html.parser"))

    base_domain = get_domain(url)

    # Meta basics
    title = clean_text(soup.title.get_text()) if soup.title else ""
    md = soup.find("meta", attrs={"name": "description"})
    meta_desc = clean_text(md["content"]) if (md and md.get("content")) else ""

    # Canonical
    canonical = None; canonical_offsite = False
    link_canon = soup.find("link", rel=lambda x: x and "canonical" in x.lower())
    if link_canon and link_canon.get("href"):
        canonical = absolute_url(link_canon["href"], url)
        canonical_offsite = get_domain(canonical) != base_domain

    # Robots noindex
    robots_noindex = False
    mr = soup.find("meta", attrs={"name": re.compile(r"robots", re.I)})
    if mr and mr.get("content") and re.search(r"noindex", mr["content"], re.I):
        robots_noindex = True

    # Viewport
    viewport_ok = bool(soup.find("meta", attrs={"name": re.compile(r"viewport", re.I)}))

    # Headings
    h1_tag = soup.find("h1"); h1 = clean_text(h1_tag.get_text()) if h1_tag else ""
    h2s = [clean_text(h.get_text()) for h in soup.find_all("h2")][:40]

    # Main content & text
    main = extract_main_content(soup)
    text_blocks = extract_text_blocks(main)
    full_text = clean_text(" ".join(text_blocks))
    word_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", full_text))
    reading_ease = flesch_reading_ease(full_text)

    # ----- Links: in-content (skip nav/footer/aside & nav-like) -----
    internal_links, external_links, internal_anchor_texts = [], [], []
    def collect_links(root):
        _i, _e, _t = [], [], []
        for a in root.find_all("a", href=True):
            if is_in_navigation_context(a):
                continue
            href = absolute_url(a["href"], url)
            txt = visible_anchor_text(a).lower()
            if exclude_toc:
                if href.endswith("#") or urllib.parse.urlparse(href).fragment:
                    continue
                chain = []
                for p in a.parents:
                    chain.extend(p.get("class") or [])
                    pid = p.get("id")
                    if pid: chain.append(pid)
                if re.search(r"(toc|table-of-contents|jump\-links|page\-contents)", " ".join(chain), re.I):
                    continue
            if require_nonempty_anchor and not txt:
                continue
            if is_internal(href, base_domain):
                _i.append(href); _t.append(txt)
            else:
                _e.append(href)
        return _i, _e, _t
    internal_links, external_links, internal_anchor_texts = collect_links(main)
    if not internal_links and not external_links:
        internal_links, external_links, internal_anchor_texts = collect_links(soup)

    # ----- Internal anchor text quality -----
    descriptive = 0
    for txt in internal_anchor_texts:
        words = [w for w in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+", txt) if w not in STOPWORDS]
        if not txt or txt in GENERIC_ANCHORS:
            continue
        descriptive += 1 if len(words) >= 3 else (0.6 if words else 0)
    internal_anchor_quality = min(1.0, descriptive / len(internal_anchor_texts)) if internal_anchor_texts else 0.0

    # ----- Internal semantic relatedness -----
    page_terms = [t for t,_ in top_terms(full_text, n=40)]
    page_term_set = set(page_terms)
    sims = []
    for href in internal_links:
        slug_terms = token_set_from_url_path(href)
        if not slug_terms:
            continue
        overlap = page_term_set.intersection(slug_terms)
        union = page_term_set.union(slug_terms)
        jacc = len(overlap) / max(1, len(union))
        bonus = 0.15 if overlap else 0.0
        sims.append(min(1.0, jacc*1.5 + bonus))
    internal_semantic_score = sum(sims)/len(sims) if sims else 0.0

    # ----- Lead / Inverted Pyramid score -----
    first_words = " ".join(full_text.split()[:150])
    rest_words = " ".join(full_text.split()[150:])
    lead_terms = set(t for t,_ in top_terms(first_words, n=20))
    all_terms = set(t for t,_ in top_terms(full_text, n=40))
    coverage = (len(lead_terms.intersection(all_terms)) / len(all_terms)) if all_terms else 0.0
    lead_cnt = sum(1 for t in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]{3,}", first_words.lower()) if t in all_terms)
    rest_cnt = sum(1 for t in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]{3,}", rest_words.lower()) if t in all_terms) or 1
    density_ratio = (lead_cnt / max(1, len(first_words.split()))) / (rest_cnt / max(1, len(rest_words.split())))
    lead_score = max(0.0, min(1.0, 0.6*coverage + 0.4*min(1.5, density_ratio)))

    # ----- Image + video effort -----
    images = main.find_all("img")
    img_count = len(images)
    with_alt = sum(1 for im in images if (im.get("alt") and clean_text(im.get("alt"))))
    img_alt_pct = (with_alt / img_count) if img_count else 0.0
    videos = main.find_all(["video","iframe"])
    video_like = 0
    for v in videos:
        src = (v.get("src") or "").lower()
        if any(host in src for host in ["youtube","vimeo","wistia","loom"]):
            video_like += 1

    # ----- Scripts (for heuristic UX fallback) -----
    script_srcs = [s.get("src") for s in soup.find_all("script", src=True)]
    third_party_scripts = [u for u in script_srcs if u and not is_internal(absolute_url(u, url), get_domain(url))]
    inline_scripts = len([s for s in soup.find_all("script") if not s.get("src")])

    # ----- Author detection -----
    author_present = False
    author_names = set()
    for sel, attrs in AUTHOR_META_KEYS:
        for tag in soup.find_all(sel, attrs=attrs):
            if tag.name == "meta" and tag.get("content"):
                author_present = True
                author_names.add(clean_text(tag["content"]))
            elif tag.name == "a" and tag.get("title"):
                author_present = True
                author_names.add(clean_text(tag.get("title") or ""))
            else:
                txt = clean_text(tag.get_text(" ", strip=True))
                if txt and len(txt.split()) <= 10 and not re.search(r"by\s*$", txt, re.I):
                    author_present = True
                    author_names.add(txt)

    # ----- Dates -----
    def find_first_date(keys):
        for sel, attrs in keys:
            tag = soup.find(sel, attrs=attrs)
            if tag:
                if tag.name == "meta" and tag.get("content"):
                    dt = try_parse_date(tag["content"])
                    if dt: return dt
                else:
                    txt = clean_text(tag.get_text())
                    dt = try_parse_date(txt)
                    if dt: return dt
        return None
    published_dt = find_first_date(DATE_META_KEYS)
    modified_dt = find_first_date(MODIFIED_META_KEYS)

    # ----- JSON-LD / schema -----
    ld = extract_ldjson(soup)
    ld_types = set(); org_schema=False; person_schema=False
    def walk_ld(obj):
        nonlocal org_schema, person_schema
        if isinstance(obj, dict):
            t = obj.get("@type")
            if isinstance(t, list):
                for x in t: ld_types.add(str(x))
            elif isinstance(t, str):
                ld_types.add(t)
            tl = str(obj.get("@type","")).lower()
            if "organization" in tl: org_schema = True
            if "person" in tl: person_schema = True
            for v in obj.values(): walk_ld(v)
        elif isinstance(obj, list):
            for v in obj: walk_ld(v)
    for blob in ld: walk_ld(blob)
    jsonld_present = bool(ld)

    # ----- URL patterns -----
    url_has_year = bool(re.search(YEAR_IN_URL, url))
    url_has_date = bool(re.search(DATE_IN_URL, url))

    # ----- Heuristic originality / AI-ish proxies -----
    tokens = [t.lower() for t in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]{3,}", full_text)]
    total = len(tokens)
    cnt = Counter(tokens)
    uniq = len(cnt)
    ttr = (uniq / total) if total > 0 else 0.0
    if total >= 2:
        bigrams = list(zip(tokens, tokens[1:]))
        bigram_total = len(bigrams)
        bigram_div = (len(set(bigrams)) / bigram_total) if bigram_total > 0 else 0.0
    else:
        bigrams = []
        bigram_div = 0.0
    top_ratio = (cnt.most_common(1)[0][1] / total) if (total > 0 and cnt) else 0.0
    sents = [s.strip() for s in re.split(r"[.!?]+", full_text) if s.strip()]
    if sents:
        sent_lengths = [len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", s)) for s in sents]
        try:
            import statistics as stats
            mean_len = max(1, stats.mean(sent_lengths))
            burst = (stats.stdev(sent_lengths) / mean_len) if len(sent_lengths) >= 2 else 0.0
        except Exception:
            burst = 0.0
    else:
        burst = 0.0
    ai_phrases = [
        "as an ai language model","in conclusion","delve into","moreover","furthermore",
        "this comprehensive guide","utilize","it is important to note that"
    ]
    ai_hits = sum(1 for p in ai_phrases if p in full_text.lower())
    diversity = (
        0.35 * min(1.0, ttr * 3.0) +
        0.25 * min(1.0, bigram_div * 4.0) +
        0.25 * min(1.0, burst) +
        0.15 * (1.0 - max(0.0, (top_ratio - 0.03) / 0.12))
    )
    ai_penalty = min(0.25, ai_hits * 0.07)
    originality_score = max(0.0, min(1.0, diversity - ai_penalty))

    # ----- Content Effort score -----
    norm_len = min(1.0, word_count/1800)
    norm_img = min(1.0, img_count/6)
    norm_vid = min(1.0, video_like/2)
    norm_schema = 1.0 if jsonld_present else 0.0
    norm_author = 1.0 if author_present else 0.0
    content_effort_score = 0.35*norm_len + 0.2*norm_img + 0.15*norm_vid + 0.15*norm_schema + 0.15*norm_author

    # ----- Stuffing risk -----
    stuffing_risk = keyword_stuffing_risk(full_text)

    # ----- Freshness -----
    months_pub = months_since(published_dt)
    months_mod = months_since(modified_dt)

    # ----- PageSpeed / CWV -----
    secrets_key = st.secrets.get("PAGESPEED_API_KEY")
    attempt_psi = bool(secrets_key) or want_psi
    psi = fetch_pagespeed(url, secrets_key or (psi_key if attempt_psi else None)) if attempt_psi else {"ok": False}

    lab_perf = psi.get("perf_score", None) if psi.get("ok") else None
    lab_LCP = psi.get("lab",{}).get("LCP_ms") if psi.get("ok") else None
    lab_CLS = psi.get("lab",{}).get("CLS") if psi.get("ok") else None
    field_LCP = psi.get("field",{}).get("LCP_ms_p75") if psi.get("ok") else None
    field_CLS = psi.get("field",{}).get("CLS_p75") if psi.get("ok") else None
    field_INP = psi.get("field",{}).get("INP_ms_p75") if psi.get("ok") else None

    heuristic_perf = None
    if not psi.get("ok"):
        size_score = 1.0 if html_bytes <= 200_000 else 0.7 if html_bytes <= 500_000 else 0.4 if html_bytes <= 1_000_000 else 0.2
        media_pen = min(1.0, (img_count + video_like*3) / 20)
        media_score = 1.0 - 0.6*media_pen
        scripts_pen = min(1.0, (len(third_party_scripts)/10) + (inline_scripts/40))
        scripts_score = 1.0 - 0.7*scripts_pen
        fetch_score = 1.0 if fetch_ms <= 800 else 0.7 if fetch_ms <= 1600 else 0.4 if fetch_ms <= 3000 else 0.2
        heuristic_perf = max(0.0, min(1.0, 0.35*size_score + 0.25*media_score + 0.25*scripts_score + 0.15*fetch_score))

    ux = {
        "psi_ok": psi.get("ok", False),
        "perf_score": lab_perf if lab_perf is not None else heuristic_perf,
        "LCP_ms": field_LCP if field_LCP is not None else lab_LCP,
        "CLS": field_CLS if field_CLS is not None else lab_CLS,
        "INP_ms": field_INP,
        "FCP_ms": psi.get("lab",{}).get("FCP_ms") if psi.get("ok") else None,
        "TTI_ms": psi.get("lab",{}).get("TTI_ms") if psi.get("ok") else None,
        "TBT_ms": psi.get("lab",{}).get("TBT_ms") if psi.get("ok") else None,
        "SI_ms": psi.get("lab",{}).get("SI_ms") if psi.get("ok") else None,
        "fetch_time_ms": fetch_ms,
        "html_bytes": html_bytes,
        "third_party_scripts": len(third_party_scripts),
        "inline_scripts": inline_scripts,
        "did_js": did_js,
        "renderer": renderer,
    }

    # ----- Keyword Relevance -----
    kw_df, kw_agg = compute_keyword_relevance(full_text, h1, h2s, keywords_raw)

    return {
        "title": title, "meta_desc": meta_desc, "h1": h1, "h2s": h2s,
        "word_count": word_count, "reading_ease": reading_ease,
        "p_internal_links": internal_links, "p_external_links": external_links,
        "internal_anchor_texts": internal_anchor_texts,
        "internal_anchor_quality": internal_anchor_quality,
        "internal_semantic_score": internal_semantic_score,
        "lead_score": lead_score,
        "img_count": img_count, "img_alt_pct": img_alt_pct, "video_like": video_like,
        "author_present": author_present, "author_names": list(author_names),
        "published_dt": published_dt, "modified_dt": modified_dt,
        "months_since_pub": months_pub, "months_since_mod": months_mod,
        "url_has_year": url_has_year, "url_has_date": url_has_date,
        "viewport_ok": viewport_ok, "canonical": canonical, "canonical_offsite": canonical_offsite,
        "robots_noindex": robots_noindex, "jsonld_present": jsonld_present,
        "ld_types": sorted(ld_types), "org_schema": org_schema, "person_schema": person_schema,
        "stuffing_risk": stuffing_risk,
        "originality_score": originality_score,
        "content_effort_score": content_effort_score,
        "ux": ux,
        "kw_df": kw_df,
        "kw_agg": kw_agg,
        # Expose content for rendering
        "full_text": full_text,
        "text_blocks": text_blocks,
    }

# -----------------------------
# Scoring Model (0..100) + category breakdown
# -----------------------------
def score_page(s: dict):
    weights = {
        # On-page (incl. lead score) — 25
        "title_match": 6,
        "h_structure": 5,
        "content_depth": 8,
        "readability": 4,
        "url_evergreen": 2,
        "lead_priority": 0,

        # Links/media — shown but weight here 0
        "internal_links": 0,
        "external_links": 0,
        "image_alts": 0,
        "internal_anchor_quality": 0,
        "internal_semantic": 0,

        # E-E-A-T — 25
        "author": 6,
        "org_schema": 4,
        "person_schema": 3,
        "jsonld": 4,
        "about_contact": 4,
        "citations": 4,

        # UX / Speed — 20
        "ux_perf": 9,
        "ux_lcp": 4,
        "ux_cls": 3,
        "ux_inp": 4,

        # Originality / Effort — 20
        "no_stuffing": 5,
        "originality": 7,
        "content_effort": 8,

        # Freshness / Canonical — 0
        "fresh_pub": 0,
        "fresh_mod": 0,
        "canonical_ok": 0,
        "viewport": 0,
        "noindex": 0,

        # Keyword Relevance — 10
        "kw_relevance": 10,
    }

    # Compute individual signals
    recs = []
    pts = 0.0

    # --- On-page signals ---
    title = s.get("title",""); h1 = s.get("h1",""); h2s = s.get("h2s") or []
    title_ok = 0.0
    if 25 <= len(title) <= 70: title_ok += 0.6
    if h1 and title and (h1.lower() in title.lower() or title.lower() in h1.lower()): title_ok += 0.4
    if title_ok < 1.0: recs.append("Tighten <title> to 25–70 chars and align strongly with H1 / search intent.")

    h_ok = 1.0 if (h1 and len(h2s) >= 2) else 0.5 if (h1 or len(h2s)>=1) else 0.0
    if h_ok < 1.0: recs.append("Use a clear heading hierarchy: one H1 and 2–6 H2s covering subtopics.")

    wc = s.get("word_count",0)
    if wc <= 300:
        depth = 0.2; recs.append("Add more substance: target ~800–1800 words of high-value content.")
    elif wc <= 800:
        depth = 0.6; recs.append("Deepen topical coverage with examples, data, FAQs.")
    elif wc <= 3000:
        depth = 1.0
    else:
        depth = 0.9

    fre = s.get("reading_ease",0)
    if fre < 30:
        rscore=0.2; recs.append("Improve readability (shorter sentences, subheadings, bullets).")
    elif fre < 40: rscore=0.5
    elif fre <= 80: rscore=1.0
    elif fre <= 90: rscore=0.8
    else: rscore=0.6

    evergreen = 0.2 if (s.get("url_has_year") or s.get("url_has_date")) else 1.0
    if evergreen < 1.0: recs.append("Prefer evergreen URLs (avoid embedding years/dates in path).")

    # Links/media (kept for recommendations)
    intern = len(s.get("p_internal_links",[]))
    extern = len(s.get("p_external_links",[]))
    if intern == 0:
        recs.append("Add contextual internal links inside content (aim 3–15).")
    elif intern < 3:
        recs.append("Add a few more internal links in the main content.")
    elif intern > 20:
        recs.append("Trim excessive internal links to avoid dilution.")

    if extern == 0:
        recs.append("Cite reputable external sources (1–10) inside the content.")
    elif extern > 10:
        recs.append("Trim excessive external links; focus on the strongest citations.")

    alt_pct = s.get("img_alt_pct",0.0)
    if alt_pct < 0.6: recs.append("Improve image alt coverage (aim >60%).")

    iaq = s.get("internal_anchor_quality",0.0)
    if iaq < 0.7: recs.append("Use descriptive internal anchor text (avoid ‘click here’; use 3+ meaningful words).")

    isem = s.get("internal_semantic_score",0.0)
    if isem < 0.6: recs.append("Link to semantically related internal URLs (align slugs with key entities/topics).")

    # --- E-E-A-T ---
    ap = 1.0 if s.get("author_present") else 0.0
    if ap==0.0: recs.append("Add clear author attribution with credentials and an author page.")
    org = 1.0 if s.get("org_schema") else 0.0
    if org==0.0: recs.append("Add Organization schema (name, logo, sameAs).")
    per = 1.0 if s.get("person_schema") else 0.0
    if per==0.0: recs.append("Add Person schema for the author.")
    jsonld = 1.0 if s.get("jsonld_present") else 0.0
    if jsonld==0.0: recs.append("Add Article/BlogPosting JSON-LD with author & dates.")
    about_contact = any(any(x in u.lower() for x in ["/about","/contact","/impressum","/company"]) for u in s.get("p_internal_links",[]))
    ac = 1.0 if about_contact else 0.0
    if ac==0.0: recs.append("Link prominently to About/Contact to improve trust.")
    citations = 1.0 if extern>0 else 0.0

    # --- UX / Speed ---
    ux = s.get("ux", {})
    perf = ux.get("perf_score")
    if perf is None:
        perf = 0.5
        recs.append("Could not fetch PageSpeed; enable PSI with an API key for CWV-backed scoring.")
    lcp = ux.get("LCP_ms")
    lcp_score = normalize_ms(lcp, 2500, 4000) if lcp is not None else 0.6
    if lcp is not None and lcp_score < 0.8:
        recs.append("Improve LCP: optimize hero image, reduce render-blocking JS/CSS, inline critical CSS, lazy-load below-the-fold.")
    cls = ux.get("CLS")
    if cls is None:
        cls_score = 0.6
    else:
        if cls <= 0.1: cls_score = 1.0
        elif cls >= 0.25: cls_score = 0.0
        else: cls_score = (0.25 - cls) / (0.25 - 0.1)
        if cls_score < 0.8:
            recs.append("Reduce CLS: reserve space for media/embeds, set width/height, avoid DOM injection above content.")
    inp = ux.get("INP_ms")
    inp_score = normalize_ms(inp, 200, 500) if inp is not None else 0.6
    if inp is not None and inp_score < 0.8:
        recs.append("Improve responsiveness (INP): split long tasks, code-split bundles, hydrate interactives lazily.")

    # --- Originality / Effort ---
    stuffing = 1.0 - min(1.0, s.get("stuffing_risk",0.0))
    if s.get("stuffing_risk",0.0) >= 0.5: recs.append("Reduce repeated keyword usage; vary phrasing and add unique value.")
    orig = s.get("originality_score",0.0)
    if orig < 0.6: recs.append("Increase originality: add unique data, examples, quotes; avoid stock phrasing.")
    effort = s.get("content_effort_score",0.0)
    if effort < 0.6: recs.append("Add images (with alts), embed video, include schema & clear authorship.")

    # --- Freshness / Canonical (recommendations only) ---
    mp = s.get("months_since_pub", math.inf)
    if mp > 24: recs.append("Refresh old content or add a clear ‘last updated’ if materially revised.")
    mm = s.get("months_since_mod", math.inf)
    if not math.isinf(mm) and mm > 12: recs.append("Make a meaningful update and surface the modified date.")
    if not s.get("canonical"): recs.append("Declare a canonical URL.")
    if s.get("canonical_offsite"): recs.append("Canonical points off-site — verify this is intended.")
    if not s.get("viewport_ok"): recs.append("Add responsive meta viewport.")
    if s.get("robots_noindex"): recs.append("Remove noindex if you want this page to rank.")

    # --- Keyword Relevance ---
    kw_agg = s.get("kw_agg")
    kw_points = max(0.0, min(1.0, float(kw_agg))) if kw_agg is not None else 0.0
    if kw_points < 0.7:
        recs.append("Strengthen topical relevance: add sections that explicitly target your priority keywords and entities.")

    # ----- Award points by category -----
    # On-page (25)
    onpage_points = (
        title_ok*6 + h_ok*5 + depth*8 + rscore*4 + evergreen*2
    )

    # E-E-A-T (25)
    eeat_points = (
        ap*6 + org*4 + per*3 + jsonld*4 + ac*4 + citations*4
    )

    # UX / Speed (20)
    ux_points = (
        (perf or 0.0)*9 + (lcp_score or 0.0)*4 + (cls_score or 0.0)*3 + (inp_score or 0.0)*4
    )

    # Originality / Effort (20)
    oe_points = (
        stuffing*5 + orig*7 + effort*8
    )

    # Keyword Relevance (10)
    kw_cat_points = kw_points*10

    total_points = onpage_points + eeat_points + ux_points + oe_points + kw_cat_points
    score_100 = round(total_points, 1)

    # Build category breakdown DF
    cat_df = pd.DataFrame([
        ["On-Page", round(onpage_points,2), 25],
        ["E-E-A-T", round(eeat_points,2), 25],
        ["UX / Speed", round(ux_points,2), 20],
        ["Originality / Effort", round(oe_points,2), 20],
        ["Keyword Relevance", round(kw_cat_points,2), 10],
    ], columns=["Category","Points","Max"])

    # Flat breakdown table for transparency
    detail_rows = [
        ("Title / Intent", round(title_ok*6,2), 6),
        ("Heading Structure", round(h_ok*5,2), 5),
        ("Content Depth", round(depth*8,2), 8),
        ("Readability", round(rscore*4,2), 4),
        ("Evergreen URL", round(evergreen*2,2), 2),

        ("Author Attribution", round(ap*6,2), 6),
        ("Org Schema", round(org*4,2), 4),
        ("Person Schema", round(per*3,2), 3),
        ("JSON-LD Present", round(jsonld*4,2), 4),
        ("About/Contact Links", round(ac*4,2), 4),
        ("Citations Present", round(citations*4,2), 4),

        ("UX: Performance", round((perf or 0.0)*9,2), 9),
        ("UX: LCP", round((lcp_score or 0.0)*4,2), 4),
        ("UX: CLS", round((cls_score or 0.0)*3,2), 3),
        ("UX: INP", round((inp_score or 0.0)*4,2), 4),

        ("No Stuffing", round(stuffing*5,2), 5),
        ("Originality", round(orig*7,2), 7),
        ("Content Effort", round(effort*8,2), 8),

        ("Keyword Relevance (avg)", round(kw_points*10,2), 10),
    ]
    breakdown = pd.DataFrame(detail_rows, columns=["Signal","Points Awarded","Max Points"])

    # Dedup & cap recs
    seen=set(); uniq=[]
    for r in recs:
        if r not in seen:
            uniq.append(r); seen.add(r)
    return score_100, cat_df, breakdown, uniq[:18]

# -----------------------------
# UI
# -----------------------------
st.title("📶 SignalScore Content")
st.caption("Heuristic content quality, E-E-A-T, UX/Speed (PSI-first), and on-page scoring (0–100) with transparent signals + keyword relevance.")

with st.sidebar:
    st.header("Analyze a Page")
    url = st.text_input("URL", placeholder="https://example.com/article")

    st.subheader("Content Relevance")
    keywords_input = st.text_area("Target Keywords (one per line or comma-separated)", height=120,
                                  placeholder="e.g.\niphone 15 battery life\na15 vs a16 speed\nbest iphone tips")

    js = st.checkbox("Enable JavaScript rendering (manual boost)", value=False,
                     help="The app auto-detects CSR pages and renders JS anyway. Enable to force JS render early.")
    wait_selector = st.text_input("Wait for selector (optional)",
                                  help="CSS selector(s), comma or line separated, to wait for before snapshot. Example: main article, .markdown-body")

    st.markdown("---")
    st.subheader("Core Web Vitals")
    want_psi_switch = st.checkbox("Fetch Google PageSpeed (mobile)", value=False,
                           help="Optional. If off, we still try PSI automatically when PAGESPEED_API_KEY is set in secrets.")
    psi_key_manual = st.text_input("PageSpeed API Key (optional)", type="password",
                            help="Leave blank to use PAGESPEED_API_KEY from Streamlit secrets (recommended).")
    ua = st.text_input("Custom User-Agent (optional)", value=HEADERS["User-Agent"])
    run = st.button("Run Analysis", type="primary")
    st.markdown("---")
    st.caption("In-content links across the page (div/section/li, etc.), excluding global nav/footer/aside & nav-like wrappers. Auto-renders JS for CSR shells.")

if run:
    if not url:
        st.error("Please enter a URL."); st.stop()
    with st.spinner("Fetching & analyzing…"):
        try:
            if ua and ua.strip():
                HEADERS["User-Agent"] = ua.strip()
            signals = analyze_url(
                url,
                use_js=js,
                want_psi=want_psi_switch,
                psi_key=(psi_key_manual or None),
                keywords_raw=keywords_input or "",
                wait_selector_input=wait_selector or None
            )
            score, cat_df, breakdown, recs = score_page(signals)
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}"); st.stop()
        except Exception as e:
            st.exception(e); st.stop()

    # GREEN SCORE CARD
    st.success(f"SignalScore Content: **{score}/100**")

    # CATEGORY BAR CHART (under card)
    chart_df = cat_df.copy()
    chart_df["Pct"] = (chart_df["Points"] / chart_df["Max"]) * 100
    fig = px.bar(chart_df, x="Category", y="Pct", range_y=[0,100],
                 text=chart_df["Pct"].round(1).astype(str) + "%",
                 labels={"Pct":"% of Category Max"})
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)

    # KPI Row
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Words", f"{signals['word_count']:,}")
    with c2: st.metric("Readability (FRE)", f"{signals['reading_ease']:.0f}")
    with c3: st.metric("Internal Links (content)", str(len(signals["p_internal_links"])))
    with c4: st.metric("External Links (content)", str(len(signals["p_external_links"])))
    with c5:
        perf = signals["ux"].get("perf_score")
        st.metric("UX Perf (0–1)", f"{perf:.2f}" if perf is not None else "—")

    # Tabs: Category Breakdown, Content, Signals, Keyword Relevance, Recommendations, Debug
    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "Category Breakdown","Content","Signals","Keyword Relevance","Recommendations","Debug"
    ])

    with tab1:
        st.subheader("Category Breakdown")
        st.dataframe(cat_df, use_container_width=True, hide_index=True)
        st.caption("Points vs Max per category. The overall score sums all category points (max 100).")

        st.markdown("### Signal Breakdown (with points and max)")
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Main Content")
        ft = signals.get("full_text") or ""
        if ft.strip():
            preview = ft[:5000] + ("..." if len(ft) > 5000 else "")
            st.text(preview)
        else:
            st.info("No main content text detected after rendering. Try enabling JavaScript rendering or check if the page is heavily client-rendered.")

    with tab3:
        st.subheader("Extracted Signals")
        colA,colB = st.columns([2,1])
        with colA:
            st.markdown(f"**Title:** {signals['title'] or '—'}")
            st.markdown(f"**Meta Description:** {signals['meta_desc'] or '—'}")
            st.markdown(f"**H1:** {signals['h1'] or '—'}")
            if signals["h2s"]:
                st.markdown("**H2s:**")
                st.write(pd.DataFrame({"H2": signals["h2s"]}))
            st.markdown(f"**Author Present:** {signals['author_present']}")
            if signals["author_names"]:
                st.markdown("**Author Names:** " + ", ".join(signals["author_names"]))
            pub = signals["published_dt"].isoformat() if signals["published_dt"] else "—"
            mod = signals["modified_dt"].isoformat() if signals["modified_dt"] else "—"
            st.markdown(f"**Published:** {pub}")
            st.markdown(f"**Modified:** {mod}")
            st.markdown(f"**Internal Anchor Quality:** {signals['internal_anchor_quality']:.2f}")
            st.markdown(f"**Internal Semantic Score:** {signals['internal_semantic_score']:.2f}")
            st.markdown(f"**Lead (Inverted Pyramid) Score:** {signals['lead_score']:.2f}")
            st.markdown(f"**Originality Score (heuristic):** {signals['originality_score']:.2f}")
            st.markdown(f"**Content Effort Score:** {signals['content_effort_score']:.2f}")
        with colB:
            st.markdown("**Technical / Meta & UX**")
            ux = signals["ux"]
            st.write(pd.DataFrame({
                "Key":[
                    "Viewport Meta","Canonical Present","Canonical Off-site",
                    "Robots Noindex","JSON-LD Present","Org Schema","Person Schema",
                    "URL Has Year","URL Has Date","Images (count)","Images w/ Alt %","Videos (embeds)",
                    "Perf Score (PSI/heuristic)","LCP (ms)","CLS","INP (ms)",
                    "Fetch Time (ms)","HTML Size (bytes)","3P Scripts","Inline Scripts",
                    "JS Rendered?","Renderer"
                ],
                "Value":[
                    signals["viewport_ok"], bool(signals["canonical"]), signals["canonical_offsite"],
                    signals["robots_noindex"], signals["jsonld_present"], signals["org_schema"], signals["person_schema"],
                    signals["url_has_year"], signals["url_has_date"], signals["img_count"], f"{int(round(signals['img_alt_pct']*100))}%", signals["video_like"],
                    (f"{ux.get('perf_score'):.2f}" if ux.get("perf_score") is not None else "—"),
                    ux.get("LCP_ms"), ux.get("CLS"), ux.get("INP_ms"),
                    ux.get("fetch_time_ms"), ux.get("html_bytes"), ux.get("third_party_scripts"), ux.get("inline_scripts"),
                    ux.get("did_js"), ux.get("renderer"),
                ],
            }))
            if signals["ld_types"]:
                st.markdown("**Detected schema.org @type:** " + ", ".join(signals["ld_types"]))

    with tab4:
        st.subheader("Keyword Relevance (per input)")
        kw_df = signals.get("kw_df")
        if kw_df is not None and not kw_df.empty:
            st.dataframe(kw_df, use_container_width=True)
            if signals.get('kw_agg') is not None:
                st.caption(f"Average Relevance: **{signals.get('kw_agg'):.2f}** (feeds the 10-point Keyword Relevance block).")
        else:
            st.info("Enter target keywords in the sidebar to compute per-keyword relevance.")

    with tab5:
        st.subheader("Top Recommendations")
        if recs:
            for r in recs: st.markdown(f"- {r}")
        else:
            st.info("Nice! No immediate gaps detected based on current heuristics.")
        # Keyword-specific recommendations for weak coverage
        kw_df = signals.get("kw_df")
        if kw_df is not None and not kw_df.empty:
            weak = kw_df[(kw_df["Relevance"].fillna(0) < 0.7)]
            if not weak.empty:
                st.markdown("**Keyword-specific recommendations:**")
                for _, row in weak.iterrows():
                    st.markdown(f"- {row['Keyword']}: {row['Recommendation']}")
        st.caption("CWV shown when available from PageSpeed Insights. Otherwise, heuristic UX score is used. Keyword relevance blends OpenAI semantic similarity (if API key present) with lexical coverage.")

    with tab6:
        st.subheader("Debug / Raw")
        st.json({
            "url": url,
            "internal_links_sample": signals["p_internal_links"][:30],
            "external_links_sample": signals["p_external_links"][:30],
            "internal_anchor_texts_sample": signals["internal_anchor_texts"][:30],
            "ux": signals["ux"],
            "stuffing_risk": round(signals["stuffing_risk"],3),
            "kw_df_preview": signals.get("kw_df").head(10).to_dict(orient="records") if (signals.get("kw_df") is not None and not signals.get("kw_df").empty) else [],
        })
else:
    st.info("Enter a URL + optional target keywords in the sidebar and click **Run Analysis**.")
