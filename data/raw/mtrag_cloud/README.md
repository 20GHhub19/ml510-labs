# MTRAG Cloud

Run `python scripts/prepare_s8.py` from the repository root. It downloads the official
Cloud passage corpus and benchmark evidence at a recorded Git revision, verifies
the references, and prepares a local BM25 index and the small CPU embedding model. Only retrieved candidates
are embedded when a query runs.

Keep these downloaded files unchanged. Indexes and the verification manifest live
under `data/processed/`; neither belongs in Git.

The archive measured at the pinned revision contains 72,442 unique passage records,
including three empty passages excluded from search. The upstream README lists
61,022; the manifest records both counts. We retain all original raw records.

Source: [IBM MTRAG benchmark](https://github.com/IBM/mt-rag-benchmark).
The repository supplies an [Apache 2.0 license](https://github.com/IBM/mt-rag-benchmark/blob/main/LICENSE).
IBM documentation retains its attribution and linked source terms; the course MIT
license does not replace third-party terms.

Follow the [S8 guide](../../../docs/s8_assistant.md).
