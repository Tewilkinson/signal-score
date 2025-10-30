import streamlit as st
import re
import math
import time
import json
import html
import asyncio
import nest_asyncio
import statistics
from datetime import datetime, timezone
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import requests
from bs4 import BeautifulSoup

# ---------- Optional CSR rendering (pyppeteer) ----------
RENDER_AVAILABLE = True
try:
    from pyppeteer import launch
except Exception:
    RENDER_AVAILABLE = False

# ---------- Optional OpenAI ----------
OPENAI_AVAILABLE = True
try:
    import openai
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="SignalScore (Content Scoring & Brief Builder)", layout="wide")

# =========================
# Utilities
# =========================

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def now_utc():
    return datetime.now(timezone.utc)

def jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def tokens(s: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[a-z0-9]+", s or "")]

def word_count(s: str) -> int:
    return len(tokens(s))

def get_secrets(name: str, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default

# =========================
# Fetch & Render
# =========================

async def _render_html(url: str, timeout: int = 25) -> str:
    browser = await launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
    try:
        page = await browser.newPage()
        await page.setViewport({"width": 1366, "height": 768})
        await page.setUserAgent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari")
        await page.goto(url, {"waitUntil": "networkidle2", "timeout": timeout * 1000})
        # give CSR-heavy a bit more time
        await asyncio.sleep(2.0)
        html_content = await page.content()
        return html_content
    finally:
        await browser.close()

def fetch_html(url: str, allow_render: bool = True) -> Tuple[str, str]:
    """
    Returns (html, mode) where mode in {'rendered','raw'}.
    """
    # Try rendered first
    if allow_render and RENDER_AVAILABLE:
        try:
            nest_asyncio.apply()
            html_content = asyncio.get_event_loop().run_until_complete(_render_html(url))
            if html_content and len(html_content) > 500:
                return html_content, "rendered"
        except Exception:
            pass

    # Fallback to raw requests
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if resp.ok:
            return resp.text, "raw"
    except Exception:
        pass
    return "", "error"

# =========================
# Parse page
# =========================

@dataclass
class PageData:
    url: str
    mode: str
    title: str
    h1: str
    h2s: List[str]
    text: str
    html: str
    images: int
    videos: int
    links: List[Tuple[str, str]]  # (href, anchor)
    dates_visible: List[datetime]
    schema_present: bool
    has_faq_schema: bool
    ssl_ok: bool
    redirects_ok: bool
    mobile_meta_ok: bool
    aria_basic_ok: bool
    main_distance_px: int
    ad_iframes: int

def parse_dates_visible(soup: BeautifulSoup) -> List[datetime]:
    out = []
    # <time datetime="...">
    for t in soup.find_all("time"):
        dt = t.get("datetime") or t.text
        if not dt:
            continue
        dt = dt.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                out.append(datetime.fromisoformat(dt.replace("Z","+00:00")) if "T" in dt else datetime.strptime(dt, fmt))
                break
            except Exception:
                continue
    # year patterns in text (rough)
    text = soup.get_text(separator=" ")
    for y in re.findall(r"\b(20[1-5][0-9])\b", text):
        try:
            out.append(datetime(int(y), 1, 1))
        except Exception:
            pass
    return out

def first_meaningful_content_distance(soup: BeautifulSoup) -> int:
    # Heuristic: count DOM depth until first long-ish <p> or list item appears
    px = 600  # default penalty-ish
    try:
        p = soup.find(["p", "article", "main"])
        if p:
            # pretend each parent adds 60px of scroll distance
            depth = 0
            cur = p
            while cur and cur.parent is not None:
                depth += 1
                cur = cur.parent
            px = min(1200, depth * 60)
    except Exception:
        pass
    return px

def detect_ads(soup: BeautifulSoup) -> int:
    # Count ad iframes/scripts heuristics
    ad_ifr = 0
    for i in soup.find_all("iframe"):
        src = (i.get("src") or "").lower()
        if any(k in src for k in ["adsystem", "doubleclick", "adservice", "googlesyndication"]):
            ad_ifr += 1
    return ad_ifr

def parse_page(url: str, allow_render: bool = True) -> Optional[PageData]:
    html_content, mode = fetch_html(url, allow_render=allow_render)
    if not html_content:
        return None
    soup = BeautifulSoup(html_content, "html.parser")

    title = clean_text(soup.title.text if soup.title else "")
    h1_el = soup.find("h1")
    h1 = clean_text(h1_el.get_text(" ")) if h1_el else ""
    h2s = [clean_text(h.get_text(" ")) for h in soup.find_all("h2")][:30]

    # visible text (rough)
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = clean_text(soup.get_text(" "))

    imgs = len(soup.find_all("img"))
    videos = len(soup.find_all("video"))

    links = []
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        anchor = clean_text(a.get_text(" "))
        if href and not href.startswith("#") and not href.lower().startswith("javascript:"):
            links.append((href, anchor))

    # JSON-LD presence
    schema_present = any(s.get("type","").lower()=="application/ld+json" for s in soup.find_all("script"))
    faq_schema = False
    for s in soup.find_all("script", {"type":"application/ld+json"}):
        try:
            data = json.loads(s.string or "{}")
            if isinstance(data, dict) and data.get("@type") in ["FAQPage", "HowTo", "Article", "NewsArticle", "BlogPosting"]:
                faq_schema = faq_schema or (data.get("@type") == "FAQPage")
        except Exception:
            continue

    # Technical sanity checks (lightweight)
    ssl_ok = url.lower().startswith("https://")
    redirects_ok = True  # requests/pypeteer already followed; deep checks omitted to keep fast
    mobile_meta_ok = bool(soup.find("meta", attrs={"name":"viewport"}))
    aria_basic_ok = any(el.has_attr("role") for el in soup.find_all(True, attrs={"role": True}))
    main_distance_px = first_meaningful_content_distance(soup)
    ad_iframes = detect_ads(soup)
    dates = parse_dates_visible(soup)

    return PageData(
        url=url, mode=mode, title=title, h1=h1, h2s=h2s, text=text, html=html_content,
        images=imgs, videos=videos, links=links, dates_visible=dates, schema_present=schema_present,
        has_faq_schema=faq_schema, ssl_ok=ssl_ok, redirects_ok=redirects_ok,
        mobile_meta_ok=mobile_meta_ok, aria_basic_ok=aria_basic_ok,
        main_distance_px=main_distance_px, ad_iframes=ad_iframes
    )

# =========================
# Scoring schema
# =========================

@dataclass
class SubScore:
    name: str
    max_points: float
    value: float
    reason: str

class Scorer:
    def __init__(self, page: PageData, keywords: List[str]):
        self.page = page
        self.keywords = [k.strip() for k in keywords if k.strip()]
        self.text_tokens = tokens(page.text)
        self.title_tokens = tokens(page.title)
        self.h1_tokens = tokens(page.h1)
        self.h2_text = " ".join(page.h2s)
        self.h2_tokens = tokens(self.h2_text)

    # -------- Content Quality & Coverage (40) --------
    def title_match(self) -> SubScore:
        sim = max(jaccard(tokens(k), self.title_tokens) for k in self.keywords) if self.keywords else 0
        presence = any(k.lower() in " ".join(self.title_tokens) for k in self.keywords)
        val = 0.6 * (sim * 10) + 0.4 * (10 if presence else 0)  # 0-10 internal
        val = min(10, val) / 10 * 6
        reason = f"Title match Jaccard~{sim:.2f}; presence={presence}"
        return SubScore("Title–Query Alignment", 6, val, reason)

    def originality_proxy(self) -> SubScore:
        # unique trigram ratio as a weak originality proxy
        toks = self.text_tokens
        if len(toks) < 200:
            return SubScore("Originality (proxy)", 4, 0.8, "Short text—uncertain uniqueness")
        trigrams = [" ".join(toks[i:i+3]) for i in range(len(toks)-2)]
        uniq_ratio = len(set(trigrams)) / max(1, len(trigrams))
        score = max(0.0, min(1.0, (uniq_ratio - 0.6) / 0.4))  # normalize 0.6-1.0 -> 0-1
        return SubScore("Originality (proxy)", 4, 4*score, f"Unique trigram ratio {uniq_ratio:.2f}")

    def length_vs_intent(self) -> SubScore:
        wc = word_count(self.page.text)
        # crude desired length: informational 1500–2500; transactional 600–1200
        kw = " ".join(self.keywords).lower()
        transactional = any(x in kw for x in ["buy","price","discount","deal","vs ","compare","best"])
        lo, hi = (600, 1200) if transactional else (1500, 2500)
        if wc < lo: score = wc/lo
        elif wc > hi:  # penalize bloat
            over = min(2.0, wc/hi)
            score = 1.0 - (over-1.0)*0.4
        else:
            score = 1.0
        score = max(0.0, min(1.0, score))
        return SubScore("Content Length vs Intent", 8, 8*score, f"{wc} words vs target {lo}-{hi}")

    def topic_saturation(self) -> SubScore:
        # use H2 coverage vs keyword tokens as proxy for breadth
        coverage = 0.0
        for k in self.keywords:
            kt = tokens(k)
            coverage += jaccard(kt, self.h2_tokens)
        if self.keywords:
            coverage /= len(self.keywords)
        score = max(0.0, min(1.0, coverage * 2.0))  # scale up a bit
        return SubScore("Topic Saturation (subtopics)", 8, 8*score, f"H2 coverage proxy {coverage:.2f}")

    def readability(self) -> SubScore:
        # Flesch Reading Ease (approx)
        text = self.page.text
        sentences = max(1, len(re.findall(r"[.!?]", text)))
        words = max(1, word_count(text))
        syllables = max(1, len(re.findall(r"[aeiouy]{1,2}", text.lower())))
        fre = 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
        # Normalize: 30 poor -> 90 excellent
        fre = max(0, min(100, fre))
        score = (fre - 30) / 60
        score = max(0.0, min(1.0, score))
        return SubScore("Readability", 4, 4*score, f"FRE≈{fre:.0f}")

    def media_and_schema(self) -> SubScore:
        media_ok = (self.page.images >= 3) or (self.page.videos >= 1)
        schema_ok = self.page.schema_present
        score = 0.0
        if media_ok: score += 0.6
        if schema_ok: score += 0.4
        return SubScore("Media & Structured Data", 4, 4*score, f"imgs={self.page.images}, schema={schema_ok}")

    def human_effort(self) -> SubScore:
        # proxy: citations (external links), unique entities (capitalized terms), and length
        externals = [href for href,_ in self.page.links if href.startswith("http")]
        citations = len(externals)
        caps = len(set(re.findall(r"\b[A-Z][a-z]{2,}\b", self.page.text)))  # rough entities
        wc = word_count(self.page.text)
        score = 0.0
        if wc>1200: score += 0.3
        score += min(0.3, citations/10*0.3)
        score += min(0.4, caps/80*0.4)
        score = min(1.0, score)
        return SubScore("Human Effort & Sources (proxy)", 6, 6*score, f"citations≈{citations}, entities≈{caps}")

    # -------- Entities, Links & Authority (22) --------
    def entity_alignment(self) -> SubScore:
        # keyword tokens appearing near capitalized entities in text
        pairs = 0
        for k in self.keywords:
            kpat = re.compile(re.escape(k), re.I)
            for m in kpat.finditer(self.page.text):
                window = self.page.text[max(0, m.start()-80):m.end()+80]
                if re.search(r"\b[A-Z][a-z]{2,}\b", window):
                    pairs += 1
        score = min(1.0, pairs/6)
        return SubScore("Entity/Knowledge Alignment (proxy)", 6, 6*score, f"entity-keyword co-occurrences≈{pairs}")

    def internal_link_semantics(self) -> SubScore:
        anchors = [a.lower() for _,a in self.page.links if domain(self.page.url)==domain(_)]
        rel_hits = 0
        for k in self.keywords:
            rel_hits += sum(1 for a in anchors if k.lower() in a)
        score = min(1.0, rel_hits/6)
        return SubScore("Internal Link Relevance", 5, 5*score, f"relevant internal anchors≈{rel_hits}")

    def onsite_prominence(self) -> SubScore:
        # proxy: count internal links TO other pages vs total links; too few suggests orphan-ish
        d = domain(self.page.url)
        internal = sum(1 for href,_ in self.page.links if domain(href)==d)
        total = len(self.page.links)
        ratio = internal / max(1,total)
        score = min(1.0, ratio / 0.5)  # 50%+ internal looks good for hubs
        return SubScore("Onsite Prominence (proxy)", 4, 4*score, f"internal/total links={internal}/{total}")

    def anchor_integrity_spam_absence(self) -> SubScore:
        anchors = [a for _,a in self.page.links]
        generic = sum(1 for a in anchors if re.search(r"\bclick here|read more|this link\b", a.lower()))
        stuffing = 1 if re.search(r"\b(best|cheap|deal)\b.{0,20}\1", self.page.text.lower()) else 0
        score = 1.0 - min(1.0, (generic/20 + stuffing*0.5))
        return SubScore("Anchor Integrity / Spam Absence", 4, 4*score, f"generic≈{generic}, stuffing={bool(stuffing)}")

    def authority_proxy(self) -> SubScore:
        # super light proxy: number of unique external root domains linked-out (not in), treats as editorial quality
        outs = set(domain(h) for h,_ in self.page.links if h.startswith("http"))
        count = len([d for d in outs if d and not d.endswith(domain(self.page.url))])
        score = min(1.0, count/10)
        return SubScore("Authority Proxy (outbound diversity)", 3, 3*score, f"unique outbound domains≈{count}")

    # -------- Freshness & Provenance (12) --------
    def explicit_byline_date(self) -> SubScore:
        if not self.page.dates_visible:
            return SubScore("Byline Date Recency", 4, 1.0, "No dates detected—neutral")
        latest = max(self.page.dates_visible)
        days = (now_utc().date() - latest.date()).days
        if days <= 180: score = 1.0
        elif days <= 540: score = 0.6
        else: score = 0.2
        return SubScore("Byline Date Recency", 4, 4*score, f"latest date ~{days} days ago")

    def semantic_recency(self) -> SubScore:
        # count of year mentions near ‘update’, ‘report’, ‘statistics’, etc.
        mentions = len(re.findall(r"(202[0-9]).{0,20}(report|update|stats|statistics|market|trend)", self.page.text.lower()))
        score = min(1.0, mentions/3)
        return SubScore("Semantic Recency", 3, 3*score, f"fresh-fact mentions≈{mentions}")

    def meaningful_updates(self) -> SubScore:
        # proxy: presence of 'Updated' label + date
        upd = 1 if re.search(r"updated\s*(on|:)?\s*\d", self.page.text.lower()) else 0
        return SubScore("Meaningful Update Signal", 3, 3*(1.0 if upd else 0.3), f"updated label detected={bool(upd)}")

    def syntactic_date(self) -> SubScore:
        url = self.page.url
        m = re.search(r"/(20[1-5]\d)/(0[1-9]|1[0-2])/", url)
        return SubScore("Syntactic/URL Date", 2, 2*(1.0 if bool(m) else 0.5), "url contains YYYY/MM" if m else "no URL date")

    # -------- Technical & UX (18) --------
    def technical_trust(self) -> SubScore:
        # SSL present + no obvious 'http://' + not too many redirects (unknown -> neutral)
        ssl = 1.0 if self.page.ssl_ok else 0.3
        redir = 1.0 if self.page.redirects_ok else 0.5
        return SubScore("Technical Trust", 4, 4*(0.6*ssl+0.4*redir), f"ssl={self.page.ssl_ok}, redirects_ok={self.page.redirects_ok}")

    def mobile_usability(self) -> SubScore:
        return SubScore("Mobile Usability Basics", 3, 3*(1.0 if self.page.mobile_meta_ok else 0.3), f"viewport meta={self.page.mobile_meta_ok}")

    def clutter_and_distance(self) -> SubScore:
        # penalize large main content distance + ad iframes
        dist = self.page.main_distance_px  # ~0..1200
        dist_pen = min(1.0, dist/1000)
        ad_pen = min(1.0, self.page.ad_iframes/4)
        score = max(0.0, 1.0 - 0.6*dist_pen - 0.4*ad_pen)
        return SubScore("Clutter & Main Content Distance", 4, 4*score, f"distance≈{dist}px, ad_iframes={self.page.ad_iframes}")

    def crawl_stability(self) -> SubScore:
        # CSR fragility proxy: if mode == rendered and raw failed earlier, slight penalty removed; we only know mode now
        mode_bonus = 1.0 if self.page.mode == "raw" else 0.7  # raw means easy to crawl
        return SubScore("Crawl Stability / CSR Fragility", 4, 4*mode_bonus, f"fetched mode={self.page.mode}")

    def accessibility_basics(self) -> SubScore:
        # alt text presence fraction (rough)
        alt_ok = 0.0
        try:
            soup = BeautifulSoup(self.page.html, "html.parser")
            imgs = soup.find_all("img")
            if imgs:
                alt_ok = sum(1 for i in imgs if i.get("alt")) / len(imgs)
        except Exception:
            pass
        score = 0.5*(1.0 if self.page.aria_basic_ok else 0.5) + 0.5*min(1.0, alt_ok)
        return SubScore("Accessibility Basics", 3, 3*score, f"ARIA roles={self.page.aria_basic_ok}, alt%≈{alt_ok:.0%}")

    # -------- Intent & Spam (8) --------
    def intent_match(self) -> SubScore:
        kw = " ".join(self.keywords).lower()
        page = (self.page.title + " " + self.page.h1 + " " + " ".join(self.page.h2s)).lower()
        info_kw = any(x in kw for x in ["what","how","guide","learn","overview","definition"])
        trans_kw = any(x in kw for x in ["buy","price","deal","coupon","discount","order"])
        info_page = any(x in page for x in ["guide","how to","what is","overview","explained"])
        trans_page = any(x in page for x in ["price","pricing","buy","plans","comparison","vs"])
        aligned = (info_kw and info_page) or (trans_kw and trans_page) or (not info_kw and not trans_kw)
        score = 1.0 if aligned else 0.4
        return SubScore("Intent Match", 5, 5*score, f"aligned={aligned}")

    def spam_guard(self) -> SubScore:
        text = self.page.text.lower()
        # keyword stuffing proxy: top keyword repetition / length
        hits = 0
        for k in self.keywords:
            hits += len(re.findall(re.escape(k.lower()), text))
        dens = hits / max(1, word_count(text))
        gibberish = 1 if re.search(r"[a-z]{30,}", text) else 0
        score = max(0.0, 1.0 - min(1.0, dens*20 + gibberish*0.5))
        return SubScore("Spam / Stuffing Guard", 3, 3*score, f"kw density≈{dens:.4f}, gibberish={bool(gibberish)}")

    def score_all(self) -> Tuple[float, List[SubScore]]:
        subs = []
        # Content
        subs += [self.title_match(), self.originality_proxy(), self.length_vs_intent(),
                 self.topic_saturation(), self.readability(), self.media_and_schema(), self.human_effort()]
        # Entities/Links/Authority
        subs += [self.entity_alignment(), self.internal_link_semantics(), self.onsite_prominence(),
                 self.anchor_integrity_spam_absence(), self.authority_proxy()]
        # Freshness
        subs += [self.explicit_byline_date(), self.semantic_recency(), self.meaningful_updates(), self.syntactic_date()]
        # Technical/UX
        subs += [self.technical_trust(), self.mobile_usability(), self.clutter_and_distance(),
                 self.crawl_stability(), self.accessibility_basics()]
        # Intent & Spam
        subs += [self.intent_match(), self.spam_guard()]

        total = sum(s.value for s in subs)
        # cap to 100 just in case of rounding noise
        total = max(0.0, min(100.0, total))
        return total, subs

# =========================
# Recommendations
# =========================

def recommendations_from_scores(subs: List[SubScore]) -> List[str]:
    tips = []
    for s in subs:
        if s.value <= 0.5 * s.max_points:
            if "Title–Query" in s.name:
                tips.append("Tighten the <title> to include the core keyword and reduce stopwords; align wording with searcher phrasing.")
            elif "Content Length" in s.name:
                tips.append("Expand or trim content to the intent-appropriate range; add missing sections and examples.")
            elif "Topic Saturation" in s.name:
                tips.append("Add missing subtopics under new H2s; mirror the breadth found in top results.")
            elif "Entity/Knowledge" in s.name:
                tips.append("Introduce key entities, stats, and sources tied to the topic; cite authoritative references.")
            elif "Internal Link" in s.name:
                tips.append("Add internal links from relevant hub pages using descriptive anchors.")
            elif "Media & Structured" in s.name:
                tips.append("Add relevant images/diagrams and valid JSON-LD (Article/FAQ/HowTo) to enable rich results.")
            elif "Byline Date" in s.name or "Semantic Recency" in s.name or "Meaningful Update" in s.name:
                tips.append("Refresh the article with 2023–2025 data points, add an 'Updated' note, and republish.")
            elif "Technical Trust" in s.name:
                tips.append("Ensure HTTPS, clean canonicals, and avoid redirect chains or 4xx/5xx in primary paths.")
            elif "Clutter" in s.name:
                tips.append("Reduce ad density and bring the first paragraph above the fold.")
            elif "Mobile" in s.name:
                tips.append("Add a responsive viewport meta, fix horizontal scroll, and check tap target spacing.")
            elif "Accessibility" in s.name:
                tips.append("Add descriptive alt text to images and basic ARIA roles/landmarks.")
            elif "Intent Match" in s.name:
                tips.append("Reframe headings to match the user intent (informational vs. transactional).")
            elif "Spam" in s.name:
                tips.append("Reduce repeated exact-match keywords; vary phrasing and improve sentence naturalness.")
            elif "Originality" in s.name:
                tips.append("Add unique insight: proprietary data, quotes, screenshots, or step-by-step detail.")
    # de-dup and limit
    dedup = []
    for t in tips:
        if t not in dedup:
            dedup.append(t)
    return dedup[:12]

# =========================
# SERPAPI & Brief Builder
# =========================

def serpapi_search(q: str, num: int = 10) -> dict:
    key = get_secrets("serpapi_key")
    if not key:
        return {}
    params = {
        "engine": "google",
        "q": q,
        "num": num,
        "api_key": key,
        "hl": "en",
        "gl": "us",
    }
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {}

def extract_paa(json_obj: dict) -> List[str]:
    out = []
    for item in json_obj.get("related_questions", []) or []:
        q = item.get("question")
        if q: out.append(q)
    return out

def estimate_comp_wordcount(json_obj: dict, cap:int=5) -> int:
    urls = []
    for r in (json_obj.get("organic_results") or [])[:cap]:
        link = r.get("link")
        if link:
            urls.append(link)
    lengths = []
    for u in urls[:cap]:
        try:
            html_txt, _ = fetch_html(u, allow_render=False)
            soup = BeautifulSoup(html_txt, "html.parser")
            for t in soup(["script","style","noscript"]): t.decompose()
            lengths.append(word_count(soup.get_text(" ")))
        except Exception:
            continue
    if not lengths:
        return 1800
    # median for robustness
    med = int(statistics.median(lengths))
    # bump a bit to "beat"
    return int(min(4000, max(800, med + 200)))

def openai_outline(prompt: str) -> str:
    key = get_secrets("openai_api_key")
    if not key or not OPENAI_AVAILABLE:
        return ""
    try:
        openai.api_key = key
        # gpt-4o-mini or gpt-4o; keep output short and structured
        msg = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a concise SEO content strategist."},
                      {"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=700
        )
        return msg.choices[0].message.content.strip()
    except Exception:
        return ""

def build_brief(keywords: List[str]) -> Dict[str, any]:
    q = ", ".join(keywords)
    serp = serpapi_search(q, num=10)
    paa = extract_paa(serp)
    wc = estimate_comp_wordcount(serp, cap=5)

    prompt = f"""
Create a beat-the-SERP brief for: {q}
Return sections:
1) SEO Title ideas (≤60 chars) – 5 options
2) H1 – 1 option
3) H2 outline – 8–12 sections covering all expected subtopics
4) Bulleted list ideas
5) FAQs – prioritize People Also Ask when relevant
6) Notes for writers – angle, evidence, examples to add

Target word count: ~{wc}
Be concise, return markdown bullet lists. No preamble.
"""
    outline = openai_outline(prompt) or f"""**Target Word Count:** ~{wc}

**H2 Outline (starter):**
- Definition & Overview
- Key Benefits
- How It Works
- Best Practices / Checklist
- Common Mistakes
- Tools & Alternatives
- Pricing / ROI (if applicable)
- FAQs

**People Also Ask (seed):**
{chr(10).join(f"- {p}" for p in paa[:8]) or "- (Add PAA once SERPAPI key is set)"}"""

    return {
        "wordcount": wc,
        "paa": paa,
        "outline_markdown": outline
    }

# =========================
# UI
# =========================

st.title("SignalScore — Content Scoring & Brief Builder")

with st.sidebar:
    st.markdown("### Input")
    url = st.text_input("Page URL")
    kw_input = st.text_area("Keyword(s) (one per line or comma-separated)")
    allow_render = st.checkbox("Robust render (pyppeteer) for CSR pages", value=True)
    run = st.button("Analyze")

tabs = st.tabs(["Score", "Recommendations", "Brief Builder"])

if run:
    if not url or not kw_input.strip():
        st.error("Please provide a URL and at least one keyword.")
        st.stop()

    # normalize keywords
    kws = [k.strip() for part in kw_input.split("\n") for k in part.split(",")]
    kws = [k for k in kws if k]

    with st.spinner("Fetching & parsing page…"):
        page = parse_page(url, allow_render=allow_render)

    if not page:
        st.error("Could not fetch or render the page.")
        st.stop()

    # ---------- SCORE ----------
    scorer = Scorer(page, kws)
    total, subs = scorer.score_all()

    with tabs[0]:
        st.subheader("Overall Score")
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            st.metric("SignalScore", f"{total:.1f}/100")
        with c2:
            st.write(f"Fetch mode: **{page.mode}**")
            st.caption(f"Title: {page.title[:90] + ('…' if len(page.title)>90 else '')}")
        with c3:
            st.write(f"Detected H2s: {len(page.h2s)} | Images: {page.images} | Videos: {page.videos}")

        st.markdown("#### Breakdown")
        st.dataframe(
            {
                "Signal": [s.name for s in subs],
                "Score": [f"{s.value:.2f} / {s.max_points}" for s in subs],
                "Reason": [s.reason for s in subs]
            },
            use_container_width=True
        )

        st.markdown("#### Page Facts")
        st.write(f"**H1:** {page.h1 or '—'}")
        if page.h2s:
            st.write("**H2s (first 10):**")
            st.write("\n".join(f"- {h}" for h in page.h2s[:10]))

    # ---------- RECOMMENDATIONS ----------
    with tabs[1]:
        st.subheader("Prioritized Recommendations")
        tips = recommendations_from_scores(subs)
        if not tips:
            st.success("No glaring issues detected. Consider incremental improvements.")
        else:
            for t in tips:
                st.markdown(f"- {t}")

        st.caption("Recommendations are derived from low-scoring signals and aligned to your rubric’s intent.")

    # ---------- BRIEF BUILDER ----------
    with tabs[2]:
        st.subheader("Beat-the-SERP Brief")
        with st.spinner("Gathering SERP context and drafting outline…"):
            brief = build_brief(kws)
        st.write(f"**Target Word Count:** ~{brief['wordcount']}")
        st.markdown("#### Suggested Outline & Assets")
        st.markdown(brief["outline_markdown"])
        if brief["paa"]:
            st.markdown("#### People Also Ask (SERPAPI)")
            st.write("\n".join(f"- {p}" for p in brief["paa"][:12]))
        else:
            st.info("Connect a SERPAPI key in secrets to fetch People Also Ask automatically.")

else:
    st.info("Enter a URL + keyword(s) on the left and click **Analyze**.")
