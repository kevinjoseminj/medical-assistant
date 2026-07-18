"""Curated corpus definition for the ingestion pipeline.

MEDLINEPLUS_TOPICS are search terms passed to the MedlinePlus web service
(https://wsearch.nlm.nih.gov/ws/query, rettype=topic) to pull official
full-summary health topic content. EXTRA_SOURCES are individual WHO fact
sheet pages fetched as HTML to fill in topics MedlinePlus covers thinly.

Note: CDC.gov fact sheets were dropped from EXTRA_SOURCES — cdc.gov returns
HTTP 403 to automated fetches (bot protection) regardless of headers used,
confirmed against several /about/index.html pages. WHO's fact sheet pages
have no such block.
"""

MEDLINEPLUS_TOPICS = [
    "diabetes",
    "type 2 diabetes",
    "high blood pressure",
    "asthma",
    "seasonal influenza",
    "common cold",
    "migraine",
    "depression",
    "anxiety",
    "seasonal allergies",
    "back pain",
    "gastroesophageal reflux disease",
    "eczema",
    "urinary tract infection",
    "strep throat",
    "high cholesterol",
    "obesity",
    "osteoarthritis",
    "irritable bowel syndrome",
    "insomnia",
    "anemia",
    "hypothyroidism",
    "pneumonia",
    "bronchitis",
    "sinusitis",
    "kidney stones",
    "gallstones",
    "acne",
    "psoriasis",
]

# (url, label) — label is used as the citation title if the page's <title>
# tag can't be parsed cleanly.
EXTRA_SOURCES = [
    ("https://www.who.int/news-room/fact-sheets/detail/tuberculosis", "WHO: Tuberculosis"),
    ("https://www.who.int/news-room/fact-sheets/detail/measles", "WHO: Measles"),
    ("https://www.who.int/news-room/fact-sheets/detail/diabetes", "WHO: Diabetes"),
    ("https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight", "WHO: Obesity and Overweight"),
    ("https://www.who.int/news-room/fact-sheets/detail/depression", "WHO: Depression"),
    ("https://www.who.int/news-room/fact-sheets/detail/asthma", "WHO: Asthma"),
    ("https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death", "WHO: The Top 10 Causes of Death"),
]
