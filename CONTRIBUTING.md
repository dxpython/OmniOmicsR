# Contributing to OmniOmicsR

Thank you for your interest in contributing to **OmniOmicsR**!  
This project aims to provide a scalable, production-quality framework for multi-omics analysis, machine learning, spatial and single-cell workflows, and clinical outcome modeling in R.  
We welcome contributions from developers, researchers, data scientists, and the open-source community.

This document outlines the guidelines and best practices for contributing.

---

## 🧭 Table of Contents

- [How to Contribute](#how-to-contribute)
- [Reporting Issues](#reporting-issues)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Code Style Guidelines](#code-style-guidelines)
- [S4 Class Design Standards](#s4-class-design-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Branching Strategy](#branching-strategy)
- [Performance Expectations](#performance-expectations)
- [Community Standards](#community-standards)

---

# 📌 How to Contribute

There are several ways to contribute:

- Fix bugs or inconsistencies
- Add new functions or modules (ML, spatial, clinical, single-cell, etc.)
- Optimize performance (R / C++ / parallel processing)
- Improve documentation and vignettes
- Add new benchmark scenarios or simulated datasets
- Suggest enhancements to the S4 class architecture

Before contributing major features, please **open an Issue** to discuss your proposal.

---

# 🐞 Reporting Issues

Please report issues through GitHub:

👉 https://github.com/dxpython/OmniOmicsR/issues

When reporting:

- Use a clear issue title
- Describe the expected and actual behavior
- Include a **minimal reproducible example**
- Provide your `sessionInfo()` output
- Attach logs or error messages when possible
- Specify OS and R version

Example:

```r
sessionInfo()
