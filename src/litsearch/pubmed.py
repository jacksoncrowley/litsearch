"""PubMed search via NCBI E-utilities (esearch + efetch).

Free, no API key required. Rate limit: 3 req/sec without key, 10 req/sec with.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class Paper:
    pmid: str
    title: str
    abstract: str
    authors: str         # "First Last; First Last; ..."
    journal: str
    pub_date: str        # YYYY-MM-DD
    doi: str
    url: str

    # Set after keyword matching
    matched_groups: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    relevance_reason: str = ""



def esearch(query: str, retmax: int = 500) -> tuple[list[str], int, str, str]:
    """Search PubMed. Returns (pmids, total_count, webenv, querykey)."""
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": str(retmax),
        "retmode": "json",
        "usehistory": "y",
    })
    url = f"{ESEARCH}?{params}"
    with urllib.request.urlopen(url, timeout=30) as f:
        data = json.load(f)

    result = data.get("esearchresult", {})
    idlist = result.get("idlist", [])
    count = int(result.get("count", 0))
    webenv = result.get("webenv", "")
    querykey = result.get("querykey", "")
    return idlist, count, webenv, querykey


def efetch_batch(pmids: list[str]) -> Optional["ET.ElementTree"]:
    """Fetch full XML records for PMIDs."""
    if not pmids:
        return None
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    })
    url = f"{EFETCH}?{params}"
    with urllib.request.urlopen(url, timeout=60) as f:
        return ET.parse(f)


def fetch_all(webenv: str, querykey: str, count: int, batch_size: int = 200) -> list["ET.ElementTree"]:
    """Fetch all records using history server. Returns list of parsed XML trees."""
    results: list["ET.ElementTree"] = []
    for retstart in range(0, count, batch_size):
        params = urllib.parse.urlencode({
            "db": "pubmed",
            "query_key": querykey,
            "WebEnv": webenv,
            "retstart": str(retstart),
            "retmax": str(batch_size),
            "retmode": "xml",
            "rettype": "abstract",
        })
        url = f"{EFETCH}?{params}"
        with urllib.request.urlopen(url, timeout=120) as f:
            results.append(ET.parse(f))
        time.sleep(0.35)
    return results


def extract_papers(xml_roots: list["ET.ElementTree"]) -> list[Paper]:
    """Parse PubmedArticleSet XML into Paper objects."""
    papers: list[Paper] = []
    for root in xml_roots:
        for article in root.findall(".//PubmedArticle"):
            medline = article.find("MedlineCitation")
            if medline is None:
                continue

            pmid_el = medline.find("PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            art = medline.find("Article")
            if art is None:
                continue

            # Title
            title_el = art.find("ArticleTitle")
            title = "".join(title_el.itertext()) if title_el is not None else ""

            # Abstract
            abs_parts: list[str] = []
            abs_el = art.find("Abstract")
            if abs_el is not None:
                for at in abs_el.findall("AbstractText"):
                    abs_parts.append("".join(at.itertext()))
            abstract = " ".join(abs_parts)

            # Authors
            authors: list[str] = []
            author_list = art.find("AuthorList")
            if author_list is not None:
                for auth in author_list.findall("Author"):
                    ln = auth.find("LastName")
                    fn = auth.find("ForeName")
                    if ln is not None and ln.text:
                        name = ln.text
                        if fn is not None and fn.text:
                            name = f"{fn.text} {name}"
                        authors.append(name)

            # Journal
            journal = ""
            journal_el = art.find("Journal")
            if journal_el is not None:
                jt = journal_el.find("Title")
                if jt is not None and jt.text:
                    journal = jt.text

            # PubDate
            pub_date = ""
            ad_el = art.find("ArticleDate")
            if ad_el is not None:
                y = ad_el.find("Year")
                m = ad_el.find("Month")
                d = ad_el.find("Day")
                if y is not None:
                    yt = y.text or ""
                    mt = ((m.text if m is not None else "01") or "01").zfill(2)
                    dt = ((d.text if d is not None else "01") or "01").zfill(2)
                    pub_date = f"{yt}-{mt}-{dt}"

            # DOI
            doi = ""
            for eid in art.findall("ELocationID"):
                if eid.get("EIdType") == "doi":
                    doi = eid.text or ""

            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            papers.append(Paper(
                pmid=pmid, title=title, abstract=abstract,
                authors="; ".join(authors), journal=journal,
                pub_date=pub_date, doi=doi, url=url,
            ))

    return papers


def search(
    terms: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_days: int = 1,
) -> list[Paper]:
    """Run a PubMed search over the date range, return all Paper objects.

    Builds a broad OR query from `terms` (all keyword group terms from config)
    to fetch candidates; client-side scoring does fine-grained filtering.
    """
    if not terms:
        print("No keywords configured — nothing to search.")
        return []

    if end_date is None:
        end_date = date.today().isoformat()
    if start_date is None:
        start_date = (date.today() - timedelta(days=lookback_days)).isoformat()

    start_pub = start_date.replace("-", "/")
    end_pub = end_date.replace("-", "/")

    term_clauses = " OR ".join(f'"{t}"[Title/Abstract]' for t in terms)
    query = (
        f'("{start_pub}"[Date - Publication] : "{end_pub}"[Date - Publication]) '
        f'AND ({term_clauses})'
    )

    print(f"PubMed: searching {start_date} to {end_date}...")
    pmids, total_count, webenv, querykey = esearch(query)

    if not pmids:
        print("  No results.")
        return []

    print(f"  Found {total_count} papers, fetching metadata...")

    if total_count <= 500:
        root = efetch_batch(pmids)
        xml_roots = [root] if root is not None else []
    else:
        xml_roots = fetch_all(webenv, querykey, total_count)

    papers = extract_papers(xml_roots)
    print(f"  Retrieved {len(papers)} papers.")
    return papers
