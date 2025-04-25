import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")



def crawl(directory):

    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}  # Remove self-links

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages



def transition_model(corpus, page, damping_factor):
    distribution = {}
    num_pages = len(corpus)
    linked_pages = corpus[page]

    if linked_pages:
        # Probability for each page: random jump + damping
        for p in corpus:
            distribution[p] = (1 - damping_factor) / num_pages
        for linked_page in linked_pages:
            distribution[linked_page] += damping_factor / len(linked_pages)
    else:
        # No links: jump to any page equally
        for p in corpus:
            distribution[p] = 1 / num_pages

    return distribution



import random

def sample_pagerank(corpus, damping_factor, n):
    page_rank = dict.fromkeys(corpus.keys(), 0)
    pages = list(corpus.keys())
    page = random.choice(pages)  # Start with a random page

    for _ in range(n):
        page_rank[page] += 1
        model = transition_model(corpus, page, damping_factor)
        page = random.choices(list(model.keys()), weights=model.values(), k=1)[0]

    # Normalize the ranks to sum to 1
    for page in page_rank:
        page_rank[page] /= n

    return page_rank



def iterate_pagerank(corpus, damping_factor, tolerance=0.001):
    num_pages = len(corpus)
    ranks = {page: 1 / num_pages for page in corpus}
    new_ranks = ranks.copy()

    # Create a reverse mapping for incoming links
    incoming_links = {page: set() for page in corpus}
    for page in corpus:
        for linked_page in corpus[page]:
            if linked_page in incoming_links:
                incoming_links[linked_page].add(page)

    while True:
        for page in corpus:
            total = 0
            for incoming in incoming_links[page]:
                num_links = len(corpus[incoming]) or num_pages
                total += ranks[incoming] / (len(corpus[incoming]) or num_pages)

            new_ranks[page] = (1 - damping_factor) / num_pages + damping_factor * total

        # Check if changes are within the tolerance
        converged = all(abs(new_ranks[page] - ranks[page]) < tolerance for page in corpus)
        ranks = new_ranks.copy()

        if converged:
            break

    return ranks



if __name__ == "__main__":
    main()