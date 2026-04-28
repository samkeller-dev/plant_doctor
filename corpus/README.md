# Corpus

This directory contains the reference documents indexed by the RAG pipeline. Each file is a single horticulture topic written as a short article with a `## Sources` section attributing real, open-access extension publications and botanical-garden references.

These documents are authored for this repo (not scraped) and are intended to be representative of the kind of grounded, citable material a real plant-care reference system would draw from. They are deliberately conservative: they describe well-established, mainstream horticulture practice, and they avoid fabricating specific URLs.

## Files

| File | Topic |
|---|---|
| `yellowing_leaves.md` | Chlorosis patterns and diagnostic logic |
| `root_rot_overwatering.md` | Root rot pathology, causes, recovery |
| `underwatering.md` | Drought stress, hydrophobic soil, recovery |
| `spider_mites.md` | Identification, conditions, treatment |
| `fungus_gnats.md` | Life cycle, BTI treatment, prevention |
| `scale_insects.md` | Soft/armored scale and mealybug ID and control |
| `light_requirements.md` | Indoor light vocabulary and species map |
| `repotting.md` | When/when-not to repot and how |

## Sources cited across the corpus

- Royal Horticultural Society (RHS) — `rhs.org.uk/advice`
- Missouri Botanical Garden — Plant Finder and care guides
- University of Maryland Extension — Home & Garden Information Center
- Pennsylvania State Extension
- University of Florida IFAS Extension
- Clemson Cooperative Extension — HGIC
- University of Minnesota Extension
- Colorado State University Extension
- North Carolina State Extension
- University of California Statewide IPM Program
- University of Illinois Extension

## Adding new documents

Drop a new `.md` file in this directory and re-run `python scripts/ingest.py` (or `docker compose run --rm app python scripts/ingest.py`) to rebuild the index. The citation-validation guardrail uses the set of filenames present here as the allow-list, so the model cannot cite a document that does not exist on disk.
