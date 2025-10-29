# app.py — SignalScore Content (SEO Suite)
import re
import json
import math
import html
import urllib.parse
from collections import Counter
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup, Comment
import tldextract
from dateutil import parser as dateparser
import streamlit as st
import pandas as pd

# Optional JS rendering (experimental)
try:
    from requests_html import HTMLSession
    HAS_REQ_HTML = True
except Exception:
    HAS_REQ_HTML = False

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

NAV_LIKE = re.compile(r"(nav|menu|breadcrumb|footer|subscribe|cookie|banner|sidebar|aside|comment|share)", re.I)
YEAR_IN_URL = re.compile(r"/(19|20)\d{2}(/|-)", re.I)
DATE_IN_URL = re.compile(r"/(19|20)\d{2}/(0?[1-9]|1[0-2])/", re.I)

GENERIC_ANCHORS = {
    "click here","read more","learn more","here","this","link","more","see more","details","visit","check this"
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
    groups = re.findall(r"[aeiouyà-äæè-ëì-ïò-öøù-ü]+", w)
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

def token_set_from_text(text: str):
    toks = {t.lower() for t in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]{3,}", text) if t.lower() not in STOPWORDS}
    return toks

def token_set_from_url_path(url: str):
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    # split by / - _ . and remove stopwords, numbers
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

def render_with_js(url: str, timeout=ABS_TIMEOUT) -> str:
    if not HAS_REQ_HTML:
        raise RuntimeError("requests-html not installed")
    s = HTMLSession()
    r = s.get(url, headers=HEADERS, timeout=timeout)
    # Render JS (pyppeteer). This may fail on some hosts; we catch upstream.
    r.html.render(timeout=timeout*1000, sleep=1)
    return r.html.html

def fetch_html(url: str, use_js: bool) -> str:
    if use_js:
        try:
            return render_with_js(url)
        except Exception:
            # Fallback to non-JS fetch
            pass
    r = requests.get(url, headers=HEADERS, timeout=ABS_TIMEOUT)
    r.raise_for_status()
    return r.text

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
    # Prefer <article>, then <main>, fallback to body stripped
    candidates = soup.find_all(["article", "main"])
    if candidates:
        # choose the longest
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

def extract_text_blocks(node):
    texts = []
    for tag in node.find_all(["p","li","h1","h2","h3","blockquote"]):
        t = clean_text(tag.get_text(" ", strip=True))
        if t:
            texts.append(t)
    return texts

# -----------------------------
# Extraction & Analysis
# -----------------------------
def analyze_url(url: str, use_js=False):
    html_text = fetch_html(url, use_js)
    soup = BeautifulSoup(html_text, "html.parser")

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

    # ----- Links: ONLY inside <p> tags in main -----
    internal_links, external_links, internal_anchor_texts = [], [], []
    p_tags = main.find_all("p")
    for p in p_tags:
        for a in p.find_all("a", href=True):
            href = absolute_url(a["href"], url)
            txt = clean_text(a.get_text(" ", strip=True)).lower()
            if is_internal(href, base_domain):
                internal_links.append(href)
                internal_anchor_texts.append(txt)
            else:
                external_links.append(href)

    # ----- Internal anchor text quality score -----
    # descriptive if >= 3 words OR contains non-generic tokens
    descriptive = 0
    for txt in internal_anchor_texts:
        words = [w for w in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+", txt) if w not in STOPWORDS]
        if not txt or txt in GENERIC_ANCHORS:
            continue
        if len(words) >= 3:
            descriptive += 1
        elif words:
            descriptive += 0.6
    internal_anchor_quality = 0.0
    if internal_anchor_texts:
        internal_anchor_quality = min(1.0, descriptive / len(internal_anchor_texts))

    # ----- Internal semantic relatedness (URL slug context vs page terms) -----
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
        # Also reward if any top term appears in slug
        bonus = 0.15 if overlap else 0.0
        sims.append(min(1.0, jacc*1.5 + bonus))
    internal_semantic_score = sum(sims)/len(sims) if sims else 0.0

    # ----- Lead / Inverted Pyramid score -----
    first_words = " ".join(full_text.split()[:150])
    rest_words = " ".join(full_text.split()[150:])
    lead_terms = set(t for t,_ in top_terms(first_words, n=20))
    all_terms = set(t for t,_ in top_terms(full_text, n=40))
    # coverage of top overall terms inside the lead
    if all_terms:
        coverage = len(lead_terms.intersection(all_terms)) / len(all_terms)
    else:
        coverage = 0.0
    # density: compare lead top-term count vs rest
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
                author_names.add(clean_text(tag["title"]))
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

    # ----- Heuristic originality / AI-ish proxies (free) -----
    # Metrics: type-token ratio, bigram diversity, repetition (top-term dominance), sentence length variance, AI-ish phrases
    tokens = [t.lower() for t in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]{3,}", full_text)]
    uniq = len(set(tokens)); total = len(tokens) or 1
    ttr = uniq / total  # higher → more diverse
    bigrams = list(zip(tokens, tokens[1:])) if len(tokens) > 1 else []
    bigram_div = len(set(bigrams))/max(1,len(bigrams))
    cnt = Counter(tokens)
    top_ratio = cnt.most_common(1)[0][1]/total if total>0 else 0.0
    # sentence burstiness
    sents = [s.strip() for s in re.split(r"[.!?]+", full_text) if s.strip()]
    sent_lengths = [len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", s)) for s in sents] or [0]
    import statistics as stats
    try:
        burst = (stats.stdev(sent_lengths)/max(1, stats.mean(sent_lengths)))
    except Exception:
        burst = 0.0
    ai_phrases = [
        "as an ai language model","in conclusion","delve into","moreover","furthermore",
        "this comprehensive guide","utilize","it is important to note that"
    ]
    ai_hits = sum(1 for p in ai_phrases if p in full_text.lower())
    # Compose originality 0..1
    diversity = 0.35*min(1.0, ttr*3) + 0.25*min(1.0, bigram_div*4) + 0.25*min(1.0, burst) + 0.15*(1.0-max(0.0,(top_ratio-0.03)/0.12))
    ai_penalty = min(0.25, ai_hits*0.07)
    originality_score = max(0.0, min(1.0, diversity - ai_penalty))

    # ----- Content Effort score (videos, images, length, schema, author) -----
    # Normalize with soft caps
    norm_len = min(1.0, word_count/1800)          # 1.0 at ~1800 words
    norm_img = min(1.0, img_count/6)              # 1.0 at 6 images
    norm_vid = min(1.0, video_like/2)             # 1.0 at 2 videos
    norm_schema = 1.0 if jsonld_present else 0.0
    norm_author = 1.0 if author_present else 0.0
    content_effort_score = 0.35*norm_len + 0.2*norm_img + 0.15*norm_vid + 0.15*norm_schema + 0.15*norm_author

    # ----- Stuffing risk -----
    stuffing_risk = keyword_stuffing_risk(full_text)

    # ----- Freshness -----
    months_pub = months_since(published_dt)
    months_mod = months_since(modified_dt)

    # Summarize
    return {
        "title": title, "meta_desc": meta_desc, "h1": h1, "h2s": h2s,
        "word_count": word_count, "reading_ease": reading_ease,
        "p_internal_links": internal_links, "p_external_links": external_links,
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
        "ld_types": sorted(ld_types), "org

