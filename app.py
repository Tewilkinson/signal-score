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

# Expanded nav-like detector (names + classes/ids)
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

def render_with_js(url: str, timeout=ABS_TIMEOUT) -> str:
    if not HAS_REQ_HTML:
        raise RuntimeError("requests-html not installed")
    s = HTMLSession()
    r = s.get(url, headers=HEADERS, timeout=timeout)
    # Render JS (pyppeteer). May fail; fallback handled upstream.
    r.html.render(timeout=timeout*1000, sleep=1)
    return r.html.html

def fetch_html(url: str, use_js: bool) -> str:
    if use_js:
        try:
            return render_with_js(url)
        except Exception:
            pass  # fallback
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
        best = max(candidates, key=lambda c: len(c.get_text(" ", strip=True)))
        return best
    body = soup.body or soup
    # strip obvious nav/footer/aside (leave header; treat global header separately later)
    for tag in body.find_all(["nav","footer","aside"]):
        tag.extract()
    for tag in body.find_all(True, class_=NAV_LIKE):
        tag.extract()
    for c in body.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    return body

def is_in_navigation_context(tag) -> bool:
    """
    Skip links inside true navigation/utility regions.
    - Treat <nav>, <footer>, <aside> as navigation.
    - Treat <header> as navigation ONLY if it's a direct child of <body> (global site header).
    - Also skip ancestors whose class/id matches NAV_LIKE.
    - If we hit <article> or role='main', we assume we're in content and stop checking.
    """
    for parent in tag.parents:
        name = getattr(parent, "name", None)
        if not name:
            continue

        # Stop early if we've reached the content root
        if name in {"article", "main"} or (parent.get("role") == "main") or (parent.get("itemprop") == "articleBody"):
            return False

        if name in {"nav", "footer", "aside"}:
            return True
        if name == "header":
            # Only treat as nav if global (direct child of body)
            gp = getattr(parent, "parent", None)
            if gp is not None and getattr(gp, "name", None) == "body":
                return True

        cls = " ".join(parent.get("class") or [])
        pid = parent.get("id") or ""
        if NAV_LIKE.search(cls) or NAV_LIKE.search(pid):
            return True

    return False

def visible_anchor_text(a):
    """Prefer visible text; fallback to title/aria-label or <img alt>."""
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
    for tag in node.find_all(["p","li","h1","h2","h3","blockquote"]):
        t = clean_text(tag.get_text(" ", strip=True))
        if t:
            texts.append(t)
    return texts

# -----------------------------
# Extraction & Analysis
# -----------------------------
def analyze_url(url: str, use_js=False, exclude_toc=True, require_nonempty_anchor=False):
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

    # ----- Link collection (content area) -----
    def collect_links(root):
        internal_links, external_links, internal_anchor_texts = [], [], []
        for a in root.find_all("a", href=True):
            if is_in_navigation_context(a):
                continue  # ignore nav/footer/etc.
            href = absolute_url(a["href"], url)
            txt = visible_anchor_text(a).lower()

            # Optional TOC exclusion
            if exclude_toc:
                if href.endswith("#") or urllib.parse.urlparse(href).fragment:
                    continue
                # skip ToC-like ancestors
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
                internal_links.append(href)
                internal_anchor_texts.append(txt)
            else:
                external_links.append(href)
        return internal_links, external_links, internal_anchor_texts

    # First pass: within main/article/body (our extracted main)
    internal_links, external_links, internal_anchor_texts = collect_links(main)

    # Fallback pass: if nothing found (some CMS markups are odd), sweep whole soup
    if not internal_links and not external_links:
        all_internal, all_external, all_texts = collect_links(soup)
        internal_links, external_links, internal_anchor_texts = all_internal, all_external, all_texts

    # ----- Internal anchor text quality -----
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
    tokens = [t.lower() for t in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]{3,}", full_text)]
    uniq = len(set(tokens)); total = len(tokens) or 1
    ttr = uniq / total
    bigrams = list(zip(tokens, tokens[1:])) if len(tokens) > 1 else []
    bigram_div = len(set(bigrams))/max(1,len(bigrams))
    cnt = Counter(tokens)
    top_ratio = cnt.most_common(1)[0][1]/total if total>0 else 0.0
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
    diversity = 0.35*min(1.0, ttr*3) + 0.25*min(1.0, bigram_div*4) + 0.25*min(1.0, burst) + 0.15*(1.0-max(0.0,(top_ratio-0.03)/0.12))
    ai_penalty = min(0.25, ai_hits*0.07)
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

    # Summarize
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
    }

# -----------------------------
# Scoring Model (0..100)
# -----------------------------
def score_page(s: dict):
    weights = {
        # On-page (incl. lead score) — 35
        "title_match": 6,
        "h_structure": 5,
        "content_depth": 8,
        "readability": 4,
        "url_evergreen": 3,
        "lead_priority": 9,

        # Links in content — 20
        "internal_links": 5,
        "external_links": 4,
        "image_alts": 4,
        "internal_anchor_quality": 3,
        "internal_semantic": 4,

        # E-E-A-T proxies — 20
        "author": 5,
        "org_schema": 3,
        "person_schema": 2,
        "jsonld": 3,
        "about_contact": 2,
        "citations": 5,

        # Freshness / canonical — 12
        "fresh_pub": 4,
        "fresh_mod": 4,
        "canonical_ok": 4,

        # Quality & safety — 13
        "viewport": 3,
        "noindex": 3,
        "stuffing_penalty": 3,
        "originality": 2,
        "content_effort": 2,
    }
    max_points = sum(weights.values())
    pts = 0.0
    recs = []

    title = s.get("title",""); h1 = s.get("h1","")
    title_ok = 0.0
    if 25 <= len(title) <= 70: title_ok += 0.6
    if h1 and title and (h1.lower() in title.lower() or title.lower() in h1.lower()): title_ok += 0.4
    pts += title_ok * weights["title_match"]
    if title_ok < 1.0: recs.append("Tighten <title> to 25–70 chars and align strongly with H1 / search intent.")

    h2s = s.get("h2s") or []
    h_ok = 1.0 if (h1 and len(h2s) >= 2) else 0.5 if (h1 or len(h2s)>=1) else 0.0
    pts += h_ok * weights["h_structure"]
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
    pts += depth * weights["content_depth"]

    fre = s.get("reading_ease",0)
    if fre < 30:
        rscore=0.2; recs.append("Improve readability (shorter sentences, subheadings, bullets).")
    elif fre < 40: rscore=0.5
    elif fre <= 80: rscore=1.0
    elif fre <= 90: rscore=0.8
    else: rscore=0.6
    pts += rscore * weights["readability"]

    evergreen = 0.2 if (s.get("url_has_year") or s.get("url_has_date")) else 1.0
    if evergreen < 1.0: recs.append("Prefer evergreen URLs (avoid embedding years/dates in path).")
    pts += evergreen * weights["url_evergreen"]

    # Lead priority
    lead = s.get("lead_score",0.0)
    pts += lead * weights["lead_priority"]
    if lead < 0.6: recs.append("Open with the most important info in the first ~150 words (entities, numbers, answers).")

    # Links/media (content-only)
    intern = len(s.get("p_internal_links",[]))
    extern = len(s.get("p_external_links",[]))
    if intern == 0:
        il=0.0; recs.append("Add contextual internal links inside content (aim 3–15).")
    elif intern < 3:
        il=0.6; recs.append("Add a few more internal links in the main content.")
    elif intern <= 20:
        il=1.0
    else: il=0.9
    pts += il * weights["internal_links"]

    if extern == 0:
        el=0.0; recs.append("Cite reputable external sources (1–10) inside the content.")
    elif extern <= 10:
        el=1.0
    else: el=0.8
    pts += el * weights["external_links"]

    alt_pct = s.get("img_alt_pct",0.0)
    ia = 1.0 if alt_pct>=0.9 else 0.8 if alt_pct>=0.6 else 0.5 if alt_pct>0 else 0.0
    if alt_pct < 0.6: recs.append("Improve image alt coverage (aim >60%).")
    pts += ia * weights["image_alts"]

    iaq = s.get("internal_anchor_quality",0.0)
    pts += iaq * weights["internal_anchor_quality"]
    if iaq < 0.7: recs.append("Use descriptive internal anchor text (avoid ‘click here’; use 3+ meaningful words).")

    isem = s.get("internal_semantic_score",0.0)
    pts += isem * weights["internal_semantic"]
    if isem < 0.6: recs.append("Link to semantically related internal URLs (align slugs with page key entities/topics).")

    # E-E-A-T proxies
    ap = 1.0 if s.get("author_present") else 0.0
    if ap==0.0: recs.append("Add clear author attribution with credentials and an author page.")
    pts += ap * weights["author"]

    org = 1.0 if s.get("org_schema") else 0.0
    per = 1.0 if s.get("person_schema") else 0.0
    jsonld = 1.0 if s.get("jsonld_present") else 0.0
    if org==0.0: recs.append("Add Organization schema (name, logo, sameAs).")
    if per==0.0: recs.append("Add Person schema for the author.")
    if jsonld==0.0: recs.append("Add Article/BlogPosting JSON-LD with author & dates.")
    pts += org * weights["org_schema"]
    pts += per * weights["person_schema"]
    pts += jsonld * weights["jsonld"]

    about_contact = any(any(x in u.lower() for x in ["/about","/contact","/impressum","/company"]) for u in s.get("p_internal_links",[]))
    ac = 1.0 if about_contact else 0.0
    if ac==0.0: recs.append("Link prominently to About/Contact to improve trust.")
    pts += ac * weights["about_contact"]

    citations = 1.0 if extern>0 else 0.0
    pts += citations * weights["citations"]

    # Freshness & canonical
    mp = s.get("months_since_pub", math.inf)
    if mp <= 6: fp=1.0
    elif mp <= 18: fp=0.8
    elif mp <= 24: fp=0.6
    else:
        fp=0.2; recs.append("Refresh old content or add a clear ‘last updated’ if materially revised.")
    pts += fp * weights["fresh_pub"]

    mm = s.get("months_since_mod", math.inf)
    if mm <= 3: fm=1.0
    elif mm <= 12: fm=0.8
    elif math.isfinite(mm): fm=0.4
    else:
        fm=0.2; recs.append("Make a meaningful update and surface the modified date.")
    pts += fm * weights["fresh_mod"]

    canonical_ok = 1.0 if (s.get("canonical") and not s.get("canonical_offsite")) else 0.6 if s.get("canonical") else 0.8
    if not s.get("canonical"): recs.append("Declare a canonical URL.")
    if s.get("canonical_offsite"): recs.append("Canonical points off-site — verify this is intended.")
    pts += canonical_ok * weights["canonical_ok"]

    # Quality/safety
    viewport = 1.0 if s.get("viewport_ok") else 0.6
    if viewport<1.0: recs.append("Add responsive meta viewport.")
    pts += viewport * weights["viewport"]

    noindex = 1.0 if not s.get("robots_noindex") else 0.0
    if noindex==0.0: recs.append("Remove noindex if you want this page to rank.")
    pts += noindex * weights["noindex"]

    stuffing = 1.0 - min(1.0, s.get("stuffing_risk",0.0))
    if s.get("stuffing_risk",0.0) >= 0.5: recs.append("Reduce repeated keyword usage; vary phrasing and add unique value.")
    pts += stuffing * weights["stuffing_penalty"]

    orig = s.get("originality_score",0.0)
    pts += orig * weights["originality"]
    if orig < 0.6: recs.append("Increase originality: add unique data, examples, quotes; avoid stock phrasing.")

    effort = s.get("content_effort_score",0.0)
    pts += effort * weights["content_effort"]
    if effort < 0.6: recs.append("Increase content effort: add images (with alts), embed video, include schema & clear authorship.")

    score_100 = round(pts / max_points * 100, 1)

    breakdown = pd.DataFrame([
        ("Title / Intent", round(title_ok*weights["title_match"],2), weights["title_match"]),
        ("Heading Structure", round(h_ok*weights["h_structure"],2), weights["h_structure"]),
        ("Content Depth", round(depth*weights["content_depth"],2), weights["content_depth"]),
        ("Readability", round(rscore*weights["readability"],2), weights["readability"]),
        ("Evergreen URL", round(evergreen*weights["url_evergreen"],2), weights["url_evergreen"]),
        ("Lead Priority", round(lead*weights["lead_priority"],2), weights["lead_priority"]),
        ("Internal Links (content)", round(il*weights["internal_links"],2), weights["internal_links"]),
        ("External Citations (content)", round(el*weights["external_links"],2), weights["external_links"]),
        ("Image Alt Coverage", round(ia*weights["image_alts"],2), weights["image_alts"]),
        ("Internal Anchor Quality", round(iaq*weights["internal_anchor_quality"],2), weights["internal_anchor_quality"]),
        ("Internal Semantic Relatedness", round(isem*weights["internal_semantic"],2), weights["internal_semantic"]),
        ("Author Attribution", round(ap*weights["author"],2), weights["author"]),
        ("Org Schema", round(org*weights["org_schema"],2), weights["org_schema"]),
        ("Person Schema", round(per*weights["person_schema"],2), weights["person_schema"]),
        ("JSON-LD Present", round(jsonld*weights["jsonld"],2), weights["jsonld"]),
        ("About/Contact Links", round(ac*weights["about_contact"],2), weights["about_contact"]),
        ("Citations Present", round(citations*weights["citations"],2), weights["citations"]),
        ("Freshness (Publish)", round(fp*weights["fresh_pub"],2), weights["fresh_pub"]),
        ("Freshness (Modify)", round(fm*weights["fresh_mod"],2), weights["fresh_mod"]),
        ("Canonical OK", round(canonical_ok*weights["canonical_ok"],2), weights["canonical_ok"]),
        ("Viewport / Mobile", round(viewport*weights["viewport"],2), weights["viewport"]),
        ("Indexable", round(noindex*weights["noindex"],2), weights["noindex"]),
        ("No Stuffing", round(stuffing*weights["stuffing_penalty"],2), weights["stuffing_penalty"]),
        ("Originality", round(orig*weights["originality"],2), weights["originality"]),
        ("Content Effort", round(effort*weights["content_effort"],2), weights["content_effort"]),
    ], columns=["Signal","Points Awarded","Max Points"])

    # dedupe recs, cap at 12
    seen=set(); uniq=[]
    for r in recs:
        if r not in seen:
            uniq.append(r); seen.add(r)
    return score_100, breakdown, uniq[:12]

# -----------------------------
# UI
# -----------------------------
st.title("📶 SignalScore Content")
st.caption("Heuristic content quality, E-E-A-T, and on-page scoring (0–100) with transparent signals.")

with st.sidebar:
    st.header("Analyze a Page")
    url = st.text_input("URL", placeholder="https://example.com/article")
    js = st.checkbox("Enable JavaScript rendering (experimental)", value=False,
                     help="Uses requests-html / Pyppeteer. May be slower or unavailable on some hosts. Falls back to basic fetch.")
    exclude_toc = st.checkbox("Exclude Table of Contents links", value=True,
                              help="Skips #fragment links and links inside toc/table-of-contents wrappers.")
    require_nonempty_anchor = st.checkbox("Require non-empty anchor text", value=False,
                              help="Ignore links with no visible/title/aria/img-alt text.")
    ua = st.text_input("Custom User-Agent (optional)", value=HEADERS["User-Agent"])
    run = st.button("Run Analysis", type="primary")
    st.markdown("---")
    st.caption("Counts in-content links across the page (div/section/li, etc.), excluding global nav/footer/aside & nav-like wrappers. Falls back to full-page sweep if needed.")

if run:
    if not url:
        st.error("Please enter a URL."); st.stop()
    with st.spinner("Fetching & analyzing…"):
        try:
            if ua and ua.strip():
                HEADERS["User-Agent"] = ua.strip()
            signals = analyze_url(url, use_js=js, exclude_toc=exclude_toc, require_nonempty_anchor=require_nonempty_anchor)
            score, breakdown, recs = score_page(signals)
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}"); st.stop()
        except Exception as e:
            st.exception(e); st.stop()

    st.success(f"SignalScore Content: **{score}/100**")

    # KPI Row
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Words", f"{signals['word_count']:,}")
    with c2: st.metric("Readability (FRE)", f"{signals['reading_ease']:.0f}")
    with c3: st.metric("Internal Links (content)", str(len(signals["p_internal_links"])))
    with c4: st.metric("External Links (content)", str(len(signals["p_external_links"])))
    with c5: st.metric("Lead Score", f"{signals['lead_score']:.2f}")

    tab1,tab2,tab3,tab4 = st.tabs(["Score Breakdown","Signals","Recommendations","Debug"])

    with tab1:
        st.subheader("Score Breakdown")
        st.dataframe(breakdown, use_container_width=True, hide_index=True)
        st.caption("Points Awarded vs Max Points (weights).")

    with tab2:
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
            st.markdown("**Technical / Meta**")
            st.write(pd.DataFrame({
                "Key":[
                    "Viewport Meta","Canonical Present","Canonical Off-site",
                    "Robots Noindex","JSON-LD Present","Org Schema","Person Schema",
                    "URL Has Year","URL Has Date","Images (count)","Images w/ Alt %","Videos (embeds)"
                ],
                "Value":[
                    signals["viewport_ok"], bool(signals["canonical"]), signals["canonical_offsite"],
                    signals["robots_noindex"], signals["jsonld_present"], signals["org_schema"], signals["person_schema"],
                    signals["url_has_year"], signals["url_has_date"], signals["img_count"], f"{int(round(signals['img_alt_pct']*100))}%", signals["video_like"]
                ],
            }))
            if signals["ld_types"]:
                st.markdown("**Detected schema.org @type:** " + ", ".join(signals["ld_types"]))

    with tab3:
        st.subheader("Top Recommendations")
        if recs:
            for r in recs: st.markdown(f"- {r}")
        else:
            st.info("Nice! No immediate gaps detected based on current heuristics.")
        st.caption("Originality & AI-likeness are heuristic only (no external databases used).")

    with tab4:
        st.subheader("Debug / Raw")
        st.json({
            "url": url,
            "internal_links_sample": signals["p_internal_links"][:30],
            "external_links_sample": signals["p_external_links"][:30],
            "internal_anchor_texts_sample": signals["internal_anchor_texts"][:30],
            "lead_score": round(signals["lead_score"],3),
            "internal_anchor_quality": round(signals["internal_anchor_quality"],3),
            "internal_semantic_score": round(signals["internal_semantic_score"],3),
            "originality_score": round(signals["originality_score"],3),
            "content_effort_score": round(signals["content_effort_score"],3),
            "stuffing_risk": round(signals["stuffing_risk"],3),
        })
else:
    st.title("")
    st.info("Enter a URL in the sidebar and click **Run Analysis**.")
