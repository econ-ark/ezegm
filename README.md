---
title: Title Page
---

# The Endogenous Grid Method for Epstein-Zin Preferences

*Solving consumption-savings problems with recursive utility*

**Alan Lujan** [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--5289--7054-green)](https://orcid.org/0000-0002-5289-7054)

Krieger School of Arts and Sciences, Johns Hopkins University, Washington, DC 20036, USA; [Econ-ARK](https://econ-ark.org)

Corresponding author: alujan@jhu.edu

---

## Abstract

The endogenous grid method (EGM) accelerates dynamic programming by inverting the Euler equation, but it appears incompatible with Epstein-Zin preferences where the value function enters the Euler equation. This paper shows that a power transformation resolves the difficulty. The resulting algorithm requires no root-finding, achieves speed gains of one to two orders of magnitude over value function iteration, and improves accuracy by more than one order of magnitude. Holding accuracy constant, the speedup is two to three orders of magnitude. VFI and time iteration face a speed-accuracy tradeoff; EGM sidesteps it entirely.

---

## Highlights

- Power transformation $W = V^{1-\rho}$ extends EGM to Epstein-Zin preferences
- Algorithm eliminates root-finding while tracking policy and value
- Speed gains of 50-100x over VFI/TI; 150-630x at equal accuracy
- Accuracy improves by over one order of magnitude versus VFI and TI
- Method generalizes to risk-sensitive preferences and limiting cases

---

## Keywords

Endogenous grid method; Epstein-Zin preferences; Recursive utility; Dynamic programming; Consumption-savings

**JEL Classification:** C61, C63, D15

---

## Requirements

- Python >= 3.12
- JAX, NumPy, SciPy, Numba, QuantEcon, Matplotlib

Install dependencies with:

```bash
uv sync
```

---

## Reproducibility

Code to reproduce all results is available at <https://github.com/econ-ark/ezegm>.

To run the benchmarks:

```bash
cd code
uv run python benchmark_paper.py
```

---

## Repository Structure

```
ezegm/
+-- code/                    # Implementation and benchmarks
|   +-- ez_egm.py           # Main EGM algorithm for Epstein-Zin
|   +-- benchmark_paper.py  # Reproduce all paper results
|   +-- ...
+-- content/
|   +-- figures/            # Generated figures
|   +-- references.bib      # Bibliography
|   +-- paper/              # Paper source (MyST Markdown)
|   |   +-- ezegm.md        # Main paper
|   |   +-- ezegm_appendix.md   # Online appendix
|   |   +-- ezegm_full.md   # Combined (includes paper + appendix)
|   +-- exports/            # Build artifacts (PDFs)
|       +-- ezegm.pdf       # Paper PDF
|       +-- ezegm_appendix.pdf  # Appendix PDF
|       +-- ezegm_full.pdf  # Full PDF
+-- myst.yml                # MyST configuration
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{lujan2026egm,
  title={The Endogenous Grid Method for Epstein-Zin Preferences},
  author={Lujan, Alan},
  year={2026},
  url={https://github.com/econ-ark/ezegm}
}
```

---

## License

- **Content:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code:** [MIT](https://opensource.org/licenses/MIT)

(c) 2026 Alan Lujan

---

## Acknowledgments

I am grateful to Christopher D. Carroll for years of mentorship, for detailed feedback on this paper, and for sustained support throughout my career. I also thank Matthew N. White for extensive discussions and collaboration that shaped this work.

---

## Funding

This work was supported by the Alfred P. Sloan Foundation [[G-2025-79177](https://sloan.org/grant-detail/g-2025-79177)].

---

## Declaration of Generative AI and AI-Assisted Technologies in the Manuscript Preparation Process

During the preparation of this work the author used Claude (Anthropic) to assist with code development and manuscript editing. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.
