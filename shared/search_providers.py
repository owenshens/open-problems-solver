"""Enhanced mathematical search providers.

Integrates multiple sources for mathematical research:
- arXiv: Preprint repository
- zbMATH: Mathematical literature database
- MathOverflow: Q&A community

All functions gracefully degrade on API failures.
"""

from __future__ import annotations

from typing import Optional


def search_arxiv(query: str, *, max_results: int = 10) -> list[dict]:
    """Search arXiv for mathematical papers.

    Returns: List of paper metadata dicts with title, authors, abstract, url, etc.
    """
    try:
        import arxiv

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for paper in search.results():
            results.append({
                'title': paper.title,
                'authors': ', '.join(a.name for a in paper.authors[:5]),
                'abstract': paper.summary[:500],
                'url': paper.pdf_url,
                'published': paper.published.strftime('%Y-%m-%d'),
                'categories': ', '.join(paper.categories[:3]),
            })
        return results
    except ImportError:
        print("Warning: arxiv package not installed. Run: pip install arxiv")
        return []
    except Exception as e:
        print(f"Warning: arXiv search failed: {e}")
        return []


def search_arxiv_formatted(query: str, *, max_results: int = 10) -> str:
    """Search arXiv and return formatted results."""
    results = search_arxiv(query, max_results=max_results)
    if not results:
        return "(No arXiv results)"

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"{i}. {r['title']}\n"
            f"   Authors: {r['authors']}\n"
            f"   Published: {r['published']}\n"
            f"   Categories: {r['categories']}\n"
            f"   Abstract: {r['abstract']}\n"
            f"   URL: {r['url']}"
        )
    return "\n\n".join(formatted)


def search_zbmath(query: str, *, max_results: int = 10) -> list[dict]:
    """Search zbMATH Open API for mathematical literature.

    Returns: List of document metadata dicts.
    """
    try:
        import requests

        base_url = "https://api.zbmath.org/v1/document/_search"
        params = {'query': query, 'size': max_results, 'pretty': 'true'}
        headers = {'Accept': 'application/json'}

        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        results = []

        for hit in data.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append({
                'title': source.get('title', 'No title'),
                'authors': ', '.join(a.get('name', '') for a in source.get('authors', [])[:5]),
                'year': str(source.get('year', 'N/A')),
                'summary': source.get('abstract', '')[:500],
                'zbmath_id': source.get('zbmath_id', ''),
                'keywords': ', '.join(source.get('keywords', [])[:5]),
            })

        return results
    except ImportError:
        print("Warning: requests package not installed. Run: pip install requests")
        return []
    except Exception as e:
        print(f"Warning: zbMATH search failed: {e}")
        return []


def search_zbmath_formatted(query: str, *, max_results: int = 10) -> str:
    """Search zbMATH and return formatted results."""
    results = search_zbmath(query, max_results=max_results)
    if not results:
        return "(No zbMATH results)"

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"{i}. {r['title']}\n"
            f"   Authors: {r['authors']}\n"
            f"   Year: {r['year']}\n"
            f"   zbMATH ID: {r['zbmath_id']}\n"
            f"   Keywords: {r['keywords']}\n"
            f"   Summary: {r['summary']}"
        )
    return "\n\n".join(formatted)


def search_mathoverflow(query: str, *, max_results: int = 5) -> list[dict]:
    """Search MathOverflow via Stack Exchange API.

    Returns: List of question metadata dicts.
    """
    try:
        import requests

        url = "https://api.stackexchange.com/2.3/search/advanced"
        params = {
            'q': query,
            'site': 'mathoverflow',
            'pagesize': max_results,
            'sort': 'votes',
            'filter': 'withbody',
            'order': 'desc',
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        results = []

        for item in data.get('items', []):
            results.append({
                'title': item.get('title', ''),
                'question': item.get('body_markdown', '')[:500],
                'score': item.get('score', 0),
                'answer_count': item.get('answer_count', 0),
                'url': item.get('link', ''),
                'tags': ', '.join(item.get('tags', [])[:5]),
            })

        return results
    except ImportError:
        print("Warning: requests package not installed. Run: pip install requests")
        return []
    except Exception as e:
        print(f"Warning: MathOverflow search failed: {e}")
        return []


def search_mathoverflow_formatted(query: str, *, max_results: int = 5) -> str:
    """Search MathOverflow and return formatted results."""
    results = search_mathoverflow(query, max_results=max_results)
    if not results:
        return "(No MathOverflow results)"

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"{i}. {r['title']}\n"
            f"   Score: {r['score']} | Answers: {r['answer_count']}\n"
            f"   Tags: {r['tags']}\n"
            f"   Question: {r['question']}\n"
            f"   URL: {r['url']}"
        )
    return "\n\n".join(formatted)
