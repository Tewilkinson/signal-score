# app.py
import re
import json
import math
import time
import html
import tldextract
import urllib.parse
import requests
from bs4 import BeautifulSoup, Comment
from collections import Counter
from datetime import datetime, timezone
from dateutil import parser as dateparser

import streamlit as st
import pandas as pd

# -----------------------------
# Utilities
# -----------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

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

ABS_TIMEOUT = 20


def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=ABS_TIMEOUT)
    r.raise_for_status()
    return r.text


def absolute_url(href: str, base: str) -> str:
    try:
        return urllib.parse.urljoin(base, href)
    except Exception:
        return href


def clean_text(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_ldjson(soup: BeautifulSoup):
    blobs = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            # Some sites have multiple JSON objects or invalid trailing commas
            text = tag.string or tag.get_text() or ""
            text = text.strip()
            if not text:
                continue
            # Handle arrays or multiple objects
            parsed = json.loads(text)
            blobs.append(parsed)
        except Exception:
            # try to recover simple JSON issues
            try:
                text = re.sub(r",\s*}", "}", text)
                text = re.sub(r",\s*]", "]", text)
                parsed = json.loads(text)
                blobs.append(parsed)
            except Exception:
                continue
    return blobs


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


def flesch_reading_ease(text: str) -> float:
    # Simple Flesch Reading Ease (approx)
    # FRE = 206.835 − 1.015*(words/sentences) − 84.6*(syllables/words)
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    if not words or not sentences:
        return 0.0
    word_count = len(words)
    sent_count = max(1, len(sentences))
    syllables = sum(estimate_syllables(w) for w in words)
    fre = 206.835 - 1.015 * (word_count / sent_count) - 84.6 * (syllables / word_count)
    return max(0.0, min(100.0, fre))


def estimate_syllables(word: str) -> int:
    w = word.lower()
    w = re.sub(r"[^a-zà-öø-ÿ]", "", w)
    if not w:
        return 0
    # very rough heuristic
    groups = re.findall(r"[aeiouyà-äæè-ëì-ïò-öøù-ü]+", w)
    count = max(1, len(groups))
    # silent 'e'
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def extract_main_content(soup: BeautifulSoup):
    # Prefer <article>, then <main>, fallback to body
    candidates = soup.find_all(["article", "main"])
    if not candidates:
        # fallback: body without nav/footer/aside/comments
        body = soup.body or soup
        removable = []
        for tag in body.find_all(["nav", "footer", "aside"]):
            removable.append(tag)
        for tag in body.find_all(True, class_=NAV_LIKE):
            removable.append(tag)
        for tag in body.find_all(string=lambda t: isinstance(t, Comment)):
            removable.append(tag)
        for r in set(removable):
            r.extract()
        return body

    # choose longest text
    best = None
    best_len = 0
    for c in candidates:
        txt = c.get_text(separator=" ", strip=True)
        l = len(txt)
        if l > best_len:
            best_len = l
            best = c
    return best or (soup.body or soup)


def extract_text_blocks(node):
    # Get only paragraphs and headings to estimate article text
    texts = []
    for tag in node.find_all(["p", "li", "h1", "h2", "h3", "blockquote"]):
        t = clean_text(tag.get_text(" ", strip=True))
        if t:
            texts.append(t)
    return texts


def top_terms(text: str, n=10):
    tokens = [t.lower() for t in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]{3,}", text)]
    stop = set("""
        the and for with that this from your you are was were have has not but can will our
        into over under more most make made being been them they their there here very just
        about when where which also than then into onto those these such suchlike upon
        """.split())
    tokens = [t for t in tokens if t not in stop]
    cnt = Counter(tokens)
    return cnt.most_common(n)


def keyword_stuffing_score(text: str) -> float:
    # crude: if top term frequency dominates too much of the text, penalize
    tokens = [t.lower() for t in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]{3,}", text)]
    total = len(tokens)
    if total < 100:
        return 0.0
    cnt = Counter(tokens)
    top_term, freq = cnt.most_common(1)[0]
    ratio = freq / total  # e.g. > 0.05 might be suspicious for long docs
    # Return 0..1 risk score
    if ratio <= 0.03:
        return 0.0
    if ratio >= 0.10:
        return 1.0
    return (ratio - 0.03) / (0.10 - 0.03)


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


# -----------------------------
# Scoring Model (0..100)
# -----------------------------
def score_page(signals: dict) -> tuple[float, pd.DataFrame, list[str]]:
    """
    signals contains:
      - title, meta_desc, h1, h2s, word_count, reading_ease
      - internal_links, external_links, img_count, img_alt_pct
      - author_present, author_names, org_schema, person_schema
      - published_dt, modified_dt, months_since_pub, months_since_mod
      - url_has_year, url_has_date, viewport_ok, canonical, robots_noindex
      - ld_types (schema.org types), jsonld_present
    """
    weights = {
        # On-page relevance & structure (30)
        "title_match": 8,
        "h_structure": 6,
        "content_depth": 8,
        "readability": 4,
        "url_evergreen": 4,

        # Links & media (15)
        "internal_links": 5,
        "external_links": 4,
        "images_alt": 6,

        # E-E-A-T proxies (30)
        "author": 8,
        "organization_schema": 6,
        "person_schema": 4,
        "about_contact": 4,
        "citations": 4,
        "jsonld": 4,

        # Freshness & updates (15)
        "fresh_pub": 6,
        "fresh_mod": 6,
        "canonical_ok": 3,

        # Quality & safety (10)
        "readable_viewport": 3,
        "noindex_penalty": 3,
        "stuffing_penalty": 4,
    }

    total_points = sum(weights.values())
    points = 0.0
    recs = []

    # --- On-page relevance & structure ---
    # Title match proxy: title present, not super short/long, similar to h1
    title = signals.get("title") or ""
    h1 = signals.get("h1") or ""
    title_ok = 0
    if 25 <= len(title) <= 70:
        title_ok += 0.6
    if h1 and title and (h1.lower() in title.lower() or title.lower() in h1.lower()):
        title_ok += 0.4
    points += title_ok * weights["title_match"]
    if title_ok < 0.6:
        recs.append("Tighten your <title>: keep ~25–70 chars and align closely with the H1 / primary intent.")

    # Heading structure
    h2s = signals.get("h2s") or []
    h_ok = 1.0 if (h1 and len(h2s) >= 2) else 0.5 if (h1 or len(h2s) >= 1) else 0.0
    points += h_ok * weights["h_structure"]
    if h_ok < 1.0:
        recs.append("Use a clear heading hierarchy: one H1 and 2–6 descriptive H2s covering sub-topics.")

    # Content depth
    wc = signals.get("word_count", 0)
    # Reward 800–2500 words range, scale softly
    if wc <= 300:
        depth = 0.2
        recs.append("Add more substance: target at least ~800 words of original, high-value content.")
    elif wc <= 800:
        depth = 0.6
        recs.append("Consider expanding the piece with examples, data, or FAQs to deepen topical coverage.")
    elif wc <= 3000:
        depth = 1.0
    else:
        depth = 0.9  # very long is okay but not always better
    points += depth * weights["content_depth"]

    # Readability
    fre = signals.get("reading_ease", 0)
    # Reward 40–80, penalize extremes
    if fre < 30:
        rscore = 0.2
        recs.append("Improve readability (shorter sentences, clearer language, subheadings, bullets).")
    elif fre < 40:
        rscore = 0.5
    elif fre <= 80:
        rscore = 1.0
    elif fre <= 90:
        rscore = 0.8
    else:
        rscore = 0.6
    points += rscore * weights["readability"]

    # URL evergreen (avoid years/dates in path)
    evergreen = 1.0
    if signals.get("url_has_year") or signals.get("url_has_date"):
        evergreen = 0.2
        recs.append("Use evergreen URLs (avoid embedding years/dates in the path).")
    points += evergreen * weights["url_evergreen"]

    # --- Links & media ---
    intern = signals.get("internal_links", 0)
    extern = signals.get("external_links", 0)
    img_alt_pct = signals.get("img_alt_pct", 0.0)

    # Internal links: reward 3–15 contextual internal links
    if intern == 0:
        il = 0.0
        recs.append("Add contextual internal links to related pages to reinforce topical clusters.")
    elif intern < 3:
        il = 0.5
        recs.append("Add a few more contextual internal links (aim for 3–15).")
    elif intern <= 20:
        il = 1.0
    else:
        il = 0.9
    points += il * weights["internal_links"]

    # External citations: reward 1–10 high-quality external links
    if extern == 0:
        el = 0.0
        recs.append("Cite reputable external sources (1–10) to support claims and improve trust.")
    elif extern <= 10:
        el = 1.0
    else:
        el = 0.8
    points += el * weights["external_links"]

    # Image alts
    if img_alt_pct >= 0.9:
        ia = 1.0
    elif img_alt_pct >= 0.6:
        ia = 0.8
    elif img_alt_pct > 0:
        ia = 0.5
        recs.append("Add descriptive alt text to images (aim for >60% coverage).")
    else:
        ia = 0.0
        recs.append("Add images with descriptive alt text to demonstrate experience and improve accessibility.")
    points += ia * weights["images_alt"]

    # --- E-E-A-T proxies ---
    if signals.get("author_present"):
        ap = 1.0
    else:
        ap = 0.0
        recs.append("Add clear author attribution with credentials and an author page.")
    points += ap * weights["author"]

    org_schema = 1.0 if signals.get("org_schema") else 0.0
    if org_schema == 0.0:
        recs.append("Add Organization schema (name, logo, sameAs profiles) site-wide.")
    points += org_schema * weights["organization_schema"]

    person_schema = 1.0 if signals.get("person_schema") else 0.0
    if person_schema == 0.0:
        recs.append("Add Person schema for the author (name, jobTitle, sameAs).")
    points += person_schema * weights["person_schema"]

    about_contact = 1.0 if signals.get("about_contact_present") else 0.0
    if about_contact == 0.0:
        recs.append("Link prominently to About and Contact pages to improve transparency and trust.")
    points += about_contact * weights["about_contact"]

    citations_present = 1.0 if extern > 0 else 0.0
    points += citations_present * weights["citations"]

    jsonld_present = 1.0 if signals.get("jsonld_present") else 0.0
    if jsonld_present == 0.0:
        recs.append("Add JSON-LD structured data (Article/BlogPosting) with publish/modify dates and author.")
    points += jsonld_present * weights["jsonld"]

    # --- Freshness & updates ---
    months_pub = signals.get("months_since_pub", math.inf)
    months_mod = signals.get("months_since_mod", math.inf)

    # Reward publish within 24 months
    if months_pub <= 6:
        fp = 1.0
    elif months_pub <= 18:
        fp = 0.8
    elif months_pub <= 24:
        fp = 0.6
    else:
        fp = 0.2
        recs.append("Refresh old content or add a clear updated date if materially revised.")
    points += fp * weights["fresh_pub"]

    # Reward modified within 12 months
    if months_mod <= 3:
        fm = 1.0
    elif months_mod <= 12:
        fm = 0.8
    elif math.isfinite(months_mod):
        fm = 0.4
    else:
        fm = 0.2
        recs.append("Log a meaningful ‘last updated’ revision (not just cosmetic date changes).")
    points += fm * weights["fresh_mod"]

    canonical_ok = 1.0 if (signals.get("canonical") and not signals.get("canonical_offsite")) else 0.6 if signals.get("canonical") else 0.8
    if not signals.get("canonical"):
        recs.append("Declare a canonical URL to avoid duplicate/variant dilution.")
    if signals.get("canonical_offsite"):
        recs.append("Canonical points off-site — verify this is intentional to avoid losing equity.")
    points += canonical_ok * weights["canonical_ok"]

    # --- Quality & safety ---
    viewport_ok = 1.0 if signals.get("viewport_ok") else 0.6
    if viewport_ok < 1.0:
        recs.append("Add a responsive meta viewport for mobile friendliness.")
    points += viewport_ok * weights["readable_viewport"]

    noindex_pen = 1.0 if not signals.get("robots_noindex") else 0.0
    if noindex_pen == 0.0:
        recs.append("Page is noindexed — remove noindex or use indexable settings if you want it to rank.")
    points += noindex_pen * weights["noindex_penalty"]

    stuff_risk = signals.get("stuffing_risk", 0.0)
    stuff_component = (1.0 - min(1.0, stuff_risk))  # higher risk → lower score
    if stuff_risk >= 0.5:
        recs.append("Reduce repeated keyword usage; vary phrasing and emphasize value for the reader.")
    points += stuff_component * weights["stuffing_penalty"]

    # Aggregate to 100
    score_100 = round(points / total_points * 100, 1)

    # Breakdown table
    breakdown = pd.DataFrame(
        [
            ("Title / Intent Match", round(title_ok * weights["title_match"], 2), weights["title_match"]),
            ("Heading Structure", round(h_ok * weights["h_structure"], 2), weights["h_structure"]),
            ("Content Depth", round(depth * weights["content_depth"], 2), weights["content_depth"]),
            ("Readability", round(rscore * weights["readability"], 2), weights["readability"]),
            ("Evergreen URL", round(evergreen * weights["url_evergreen"], 2), weights["url_evergreen"]),
            ("Internal Links", round(il * weights["internal_links"], 2), weights["internal_links"]),
            ("External Citations", round(el * weights["external_links"], 2), weights["external_links"]),
            ("Image Alt Coverage", round(ia * weights["images_alt"], 2), weights["images_alt"]),
            ("Author Attribution", round(ap * weights["author"], 2), weights["author"]),
            ("Org Schema", round(org_schema * weights["organization_schema"], 2), weights["organization_schema"]),
            ("Person Schema", round(person_schema * weights["person_schema"], 2), weights["person_schema"]),
            ("About/Contact Links", round(about_contact * weights["about_contact"], 2), weights["about_contact"]),
            ("JSON-LD Present", round(jsonld_present * weights["jsonld"], 2), weights["jsonld"]),
            ("Freshness (Publish)", round(fp * weights["fresh_pub"], 2), weights["fresh_pub"]),
            ("Freshness (Modify)", round(fm * weights["fresh_mod"], 2), weights["fresh_mod"]),
            ("Canonical OK", round(canonical_ok * weights["canonical_ok"], 2), weights["canonical_ok"]),
            ("Viewport / Mobile", round(viewport_ok * weights["readable_viewport"], 2), weights["readable_viewport"]),
            ("Noindex (Penalty)", round(noindex_pen * weights["noindex_penalty"], 2), weights["noindex_penalty"]),
            ("Keyword Stuffing (Penalty)", round(stuff_component * weights["stuffing_penalty"], 2), weights["stuffing_penalty"]),
        ],
        columns=["Signal", "Points Awarded", "Max Points"]
    )

    # Top recommendations (deduplicate, limit)
    seen = set()
    unique_recs = []
    for r in recs:
        if r not in seen:
            unique_recs.append(r)
            seen.add(r)
    return score_100, breakdown, unique_recs[:10]


# -----------------------------
# Extraction pipeline
# -----------------------------
def analyze_url(url: str):
    html_text = fetch_html(url)
    soup = BeautifulSoup(html_text, "html.parser")

    base_domain = get_domain(url)

    # Meta basics
    title = clean_text(soup.title.get_text()) if soup.title else ""
    meta_desc = ""
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        meta_desc = clean_text(md["content"])

    # Canonical
    canonical = None
    canonical_offsite = False
    link_canon = soup.find("link", rel=lambda x: x and "canonical" in x.lower())
    if link_canon and link_canon.get("href"):
        canonical = absolute_url(link_canon["href"], url)
        canonical_offsite = get_domain(canonical) != base_domain

    # Robots noindex
    robots_noindex = False
    mr = soup.find("meta", attrs={"name": re.compile(r"robots", re.I)})
    if mr and mr.get("content"):
        if re.search(r"noindex", mr["content"], re.I):
            robots_noindex = True

    # Viewport
    viewport_ok = bool(soup.find("meta", attrs={"name": re.compile(r"viewport", re.I)}))

    # Headings
    h1_tag = soup.find("h1")
    h1 = clean_text(h1_tag.get_text()) if h1_tag else ""
    h2s = [clean_text(h.get_text()) for h in soup.find_all("h2")][:20]

    # Main content node and text
    main = extract_main_content(soup)
    text_blocks = extract_text_blocks(main)
    full_text = clean_text(" ".join(text_blocks))
    word_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", full_text))
    reading_ease = flesch_reading_ease(full_text)

    # Images + alt coverage
    images = main.find_all("img")
    img_count = len(images)
    with_alt = sum(1 for im in images if (im.get("alt") and clean_text(im.get("alt"))))
    img_alt_pct = (with_alt / img_count) if img_count else 0.0

    # Links
    links = [a.get("href") for a in main.find_all("a", href=True)]
    links = [absolute_url(h, url) for h in links]
    internals = [h for h in links if is_internal(h, base_domain)]
    externals = [h for h in links if not is_internal(h, base_domain)]

    # About / Contact presence
    lower_links = [h.lower() for h in links]
    about_contact = any(
        any(x in h for x in ["/about", "/about-us", "/contact", "/contact-us", "/impressum", "/company"])
        for h in lower_links
    )

    # Author detection
    author_present = False
    author_names = set()
    for sel, attrs in AUTHOR_META_KEYS:
        for tag in soup.find_all(sel, attrs=attrs):
            # meta name=author
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

    # Dates
    def find_first_date(keys):
        for sel, attrs in keys:
            tag = soup.find(sel, attrs=attrs)
            if tag:
                if tag.name == "meta" and tag.get("content"):
                    dt = try_parse_date(tag["content"])
                    if dt:
                        return dt
                else:
                    txt = clean_text(tag.get_text())
                    dt = try_parse_date(txt)
                    if dt:
                        return dt
        return None

    published_dt = find_first_date(DATE_META_KEYS)
    modified_dt = find_first_date(MODIFIED_META_KEYS)

    # JSON-LD / schema
    ld = extract_ldjson(soup)
    ld_types = set()

    org_schema = False
    person_schema = False

    def walk_ld(obj):
        if isinstance(obj, dict):
            t = obj.get("@type")
            if isinstance(t, list):
                for x in t:
                    ld_types.add(str(x))
            elif isinstance(t, str):
                ld_types.add(t)
            # flags
            if str(t).lower() in {"organization", "org"}:
                nonlocal org_schema
                org_schema = True
            if str(t).lower() in {"person"}:
                nonlocal person_schema
                person_schema = True
            for v in obj.values():
                walk_ld(v)
        elif isinstance(obj, list):
            for v in obj:
                walk_ld(v)

    for blob in ld:
        walk_ld(blob)

    jsonld_present = bool(ld)

    # URL patterns
    url_has_year = bool(re.search(YEAR_IN_URL, url))
    url_has_date = bool(re.search(DATE_IN_URL, url))

    # Stuffing risk
    stuffing_risk = keyword_stuffing_score(full_text)

    # Freshness
    months_pub = months_since(published_dt)
    months_mod = months_since(modified_dt)

    # Summarize signals
    signals = {
        "title": title,
        "meta_desc": meta_desc,
        "h1": h1,
        "h2s": h2s,
        "word_count": word_count,
        "reading_ease": reading_ease,
        "internal_links": len(internals),
        "external_links": len(externals),
        "img_count": img_count,
        "img_alt_pct": img_alt_pct,
        "author_present": author_present,
        "author_names": list(author_names),
        "published_dt": published_dt,
        "modified_dt": modified_dt,
        "months_since_pub": months_pub,
        "months_since_mod": months_mod,
        "url_has_year": url_has_year,
        "url_has_date": url_has_date,
        "viewport_ok": viewport_ok,
        "canonical": canonical,
        "canonical_offsite": canonical_offsite,
        "robots_noindex": robots_noindex,
        "jsonld_present": jsonld_present,
        "ld_types": sorted(ld_types),
        "org_schema": org_schema,
        "person_schema": person_schema,
        "about_contact_present": about_contact,
        "stuffing_risk": stuffing_risk,
    }
    return signals


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Content Rank Score", page_icon="📈", layout="wide")

st.title("📈 Content Rank Score (0–100)")
st.caption(
    "Scrapes a URL, evaluates on-page + E-E-A-T proxy signals, and returns a score with recommendations."
)

with st.sidebar:
    st.header("Analyze a Page")
    url = st.text_input("URL to analyze", placeholder="https://example.com/article")
    ua = st.text_input("Custom User-Agent (optional)", value=HEADERS["User-Agent"])
    run = st.button("Run Analysis", type="primary")
    st.markdown("—")
    st.caption("Tip: Use evergreen URLs; make meaningful updates; add author schema & internal links.")

if run:
    if not url:
        st.error("Please enter a URL.")
        st.stop()

    with st.spinner("Fetching and analyzing…"):
        try:
            if ua and ua.strip():
                HEADERS["User-Agent"] = ua.strip()
            signals = analyze_url(url)
            score, breakdown, recs = score_page(signals)
        except requests.exceptions.RequestException as e:
            st.error(f"Network error fetching the page: {e}")
            st.stop()
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success(f"Rank Score: **{score}/100**")

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Words", f"{signals['word_count']:,}")
    with c2:
        st.metric("Readability (FRE)", f"{signals['reading_ease']:.0f}")
    with c3:
        st.metric("Internal Links", str(signals["internal_links"]))
    with c4:
        st.metric("External Links", str(signals["external_links"]))
    with c5:
        alt_pct = int(round(signals["img_alt_pct"] * 100))
        st.metric("Images w/ Alt", f"{alt_pct}%")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Score Breakdown", "Signals", "Recommendations", "Debug"])

    with tab1:
        st.subheader("Score Breakdown")
        st.dataframe(breakdown, use_container_width=True, hide_index=True)
        st.caption("Points Awarded vs. Max Points (weights) for each signal.")

    with tab2:
        st.subheader("Extracted Signals")
        left, right = st.columns([2, 1])
        with left:
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

        with right:
            st.markdown("**Technical / Meta**")
            st.write(pd.DataFrame(
                {
                    "Key": [
                        "Viewport Meta", "Canonical Present", "Canonical Off-site",
                        "Robots Noindex", "JSON-LD Present", "Org Schema", "Person Schema",
                        "URL Has Year", "URL Has Date",
                    ],
                    "Value": [
                        signals["viewport_ok"],
                        bool(signals["canonical"]),
                        signals["canonical_offsite"],
                        signals["robots_noindex"],
                        signals["jsonld_present"],
                        signals["org_schema"],
                        signals["person_schema"],
                        signals["url_has_year"],
                        signals["url_has_date"],
                    ],
                }
            ))
            if signals["ld_types"]:
                st.markdown("**Detected schema.org @type:** " + ", ".join(signals["ld_types"]))

    with tab3:
        st.subheader("Top Recommendations")
        if recs:
            for r in recs:
                st.markdown(f"- {r}")
        else:
            st.info("Nice! No immediate gaps detected based on current heuristics.")

        st.markdown("---")
        st.caption(
            "This score is heuristic and based on publicly inferable on-page signals and common E-E-A-T proxies. "
            "It does not use private ranking signals."
        )

    with tab4:
        st.subheader("Debug / Raw Numbers")
        st.json({
            "url": url,
            "word_count": signals["word_count"],
            "reading_ease": signals["reading_ease"],
            "internal_links": signals["internal_links"],
            "external_links": signals["external_links"],
            "img_count": signals["img_count"],
            "img_alt_pct": signals["img_alt_pct"],
            "author_present": signals["author_present"],
            "published_dt": signals["published_dt"].isoformat() if signals["published_dt"] else None,
            "modified_dt": signals["modified_dt"].isoformat() if signals["modified_dt"] else None,
            "months_since_pub": None if math.isinf(signals["months_since_pub"]) else round(signals["months_since_pub"], 1),
            "months_since_mod": None if math.isinf(signals["months_since_mod"]) else round(signals["months_since_mod"], 1),
            "url_has_year": signals["url_has_year"],
            "url_has_date": signals["url_has_date"],
            "viewport_ok": signals["viewport_ok"],
            "canonical": signals["canonical"],
            "canonical_offsite": signals["canonical_offsite"],
            "robots_noindex": signals["robots_noindex"],
            "jsonld_present": signals["jsonld_present"],
            "ld_types": signals["ld_types"],
            "about_contact_present": signals["about_contact_present"],
            "stuffing_risk": round(signals["stuffing_risk"], 3),
        })

else:
    st.info("Enter a URL in the sidebar and click **Run Analysis**.")
