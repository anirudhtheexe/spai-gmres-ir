# BSPAI–GMRES–IR — Paper Index

Last checked: 11 October 2026

## Purpose

Shared literature index for the BSPAI–GMRES–IR CUDA/parallel computing project. Start with the two Carson–Khan papers, then the GPU SPAI and mixed-precision literature. A DOI or accessible abstract is **not** evidence that the complete paper was read. Update the access and review columns after retrieving and examining full text.

**Access key:** **Open PDF/preprint** = a public full-text route was identified; **Open article** = publisher labels the article open access; **Access uncertain** = citation/abstract found, but a freely downloadable publisher PDF was not established. **Review status:** all entries are **Not yet reviewed in Codex**.

## Priority papers (numbers preserved from our latest project shortlist)

| Prior # | Paper | Year | Relevance | Publication / DOI | Public full-text route | Access | Review |
|---|---|---|---|---|---|---|---|
| 1 | **Mixed Precision Iterative Refinement with Sparse Approximate Inverse Preconditioning** — Erin Carson, Noaman Khan | 2023 | Introduces SPAI–GMRES–IR; essential baseline | [SIAM / DOI](https://doi.org/10.1137/22M1487709) | [arXiv:2202.10204](https://arxiv.org/pdf/2202.10204) | Open PDF/preprint | Not yet reviewed |
| 2 | **Mixed Precision Iterative Refinement with Adaptive Precision Sparse Approximate Inverse Preconditioning** — Noaman Khan, Erin Carson | 2025 | Develops BSPAI–GMRES–IR; central algorithm | [Springer / DOI](https://doi.org/10.1007/s00366-025-02187-z) | [arXiv:2307.03914](https://arxiv.org/pdf/2307.03914) | Open article + preprint | Not yet reviewed |
| 7 | **A Thread-Adaptive Sparse Approximate Inverse Preconditioning Algorithm on Multi-GPUs** — Jiaquan Gao, Q. Chen, G. He | 2021 | Multi-GPU construction of SPAI | [DOI](https://doi.org/10.1016/j.parco.2020.102724) | [Publisher page](https://www.sciencedirect.com/science/article/abs/pii/S0167819120301083) | Access uncertain | Not yet reviewed |
| 8 | **A Parallel Sparse Approximate Inverse Preconditioning Algorithm Based on MPI and CUDA** — Yizhou Wang, Wenhao Li, Jiaquan Gao | 2021 | Hybrid MPI/CUDA SPAI; existing GPU baseline ideas | [DOI](https://doi.org/10.1016/j.tbench.2021.100007) | [Open PDF (TBench)](https://www.tbench.org/single/1-9.pdf) | Open PDF | Not yet reviewed |
| 11 | **A Study of Mixed Precision Strategies for GMRES on GPUs** — Jennifer A. Loe et al. | 2021 | GMRES mixed-precision GPU design | [arXiv abstract](https://arxiv.org/abs/2109.01232) | [arXiv PDF](https://arxiv.org/pdf/2109.01232) | Open PDF/preprint | Not yet reviewed |
| 15 | **Adaptive Precision Sparse Matrix–Vector Product and Its Application to Krylov Solvers** — Stef Graillat et al. | 2024 | Precision bucketing for SpMV and Krylov methods | [SIAM / DOI](https://doi.org/10.1137/22M1522619) | [Publisher article](https://epubs.siam.org/doi/10.1137/22M1522619) | Access uncertain | Not yet reviewed |
| 16 | **Data-driven Mixed Precision Sparse Matrix Vector Multiplication for GPUs** — Khalid Ahmad, Hari Sundar, Mary Hall | 2019 | GPU mixed-precision SpMV strategy | [DOI](https://doi.org/10.1145/3371275) | [ResearchGate listing](https://www.researchgate.net/publication/338023441_Data-driven_Mixed_Precision_Sparse_Matrix_Vector_Multiplication_for_GPUs) | Access uncertain | Not yet reviewed |
| 17 | **Block-wise Dynamic Mixed-Precision for Sparse Matrix-Vector Multiplication on GPUs** — Zhixiang Zhao et al. | 2024 | Blockwise adaptive GPU SpMV | [Springer / DOI](https://doi.org/10.1007/s11227-024-05949-6) | [Open publisher article](https://link.springer.com/article/10.1007/s11227-024-05949-6) | Open article | Not yet reviewed |

## Foundational references (additional; not assigned earlier-list numbers)

| Paper | Year | Relevance | Link | Access | Review |
|---|---|---|---|---|---|
| **GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems** — Youcef Saad, Martin H. Schultz | 1986 | Original GMRES algorithm | [SIAM / DOI](https://doi.org/10.1137/0907058) | Publisher labels free access | Not yet reviewed |
| **Parallel Preconditioning with Sparse Approximate Inverses** — Marcus J. Grote, Thomas Huckle | 1997 | Foundational SPAI construction and parallelism | [DOI](https://doi.org/10.1137/S1064827594276552); [public PDF](https://citeseerx.ist.psu.edu/document?doi=5b0082cd21e5aef0891a04e5143f908115487947&repid=rep1&type=pdf) | Open PDF | Not yet reviewed |
| **Accelerating the Solution of Linear Systems by Iterative Refinement in Three Precisions** — Erin Carson, Nicholas J. Higham | 2018 | Mixed-precision iterative refinement foundation | [SIAM / DOI](https://doi.org/10.1137/17M1140819) | Publisher labels open access | Not yet reviewed |

## Code repositories to inspect (not papers)

- [Noaman67khan/SPAI-GMRES-IR](https://github.com/Noaman67khan/SPAI-GMRES-IR)
- [Noaman67khan/ADAP-SPAI-GMRES-IR](https://github.com/Noaman67khan/ADAP-SPAI-GMRES-IR)

Repository contents, licenses, numerical correctness, and degree of parallelization are **not yet verified in this index**.

## Agent workflow

1. Begin with papers **1 and 2** to establish the mathematical algorithm, precision choices, stopping rules, and published experiments.
2. Use papers **7 and 8** to assess prior SPAI GPU and MPI/CUDA work; do not claim GPU SPAI is novel.
3. Use papers **11, 15, 16, 17** for GPU GMRES/SpMV and adaptive precision design alternatives.
4. For every paper, distinguish: (a) abstract/metadata inspected, (b) full text obtained, (c) method understood, (d) results independently reproduced.
5. Do not present a paper's unverified implementation details, performance numbers, code availability, or future-work claims as established facts.
6. If a PDF cannot be accessed, search legally available author manuscripts, repositories, or institutional copies; otherwise mark **full text unavailable** and work from the abstract with that limitation stated.

## Remaining bibliography work

Our earlier project conversation referred to a **21-paper list**. This file currently verifies and indexes **11 distinct papers** (the eight priority papers plus three foundations), not a reconstruction of the full 21. The other entries should be added only after their exact titles and bibliographic details are recovered and verified. Earlier numbering for the eight priority papers is deliberately preserved; gaps do not mean papers are missing from the literature generally.
