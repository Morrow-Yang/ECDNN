# ECDNN — Efficient Computing of Deep Neural Networks

> 深度学习中的高效计算方法

This repository contains a **course-material index** for the **Efficient Computing of Deep Neural Networks** summer course. 

**Note:** The actual slide PDFs and legacy assignment/lab files are **not hosted here** out of respect for copyright. See [How to Obtain the Materials](#how-to-obtain-the-materials) below.

## Course Information

| Item | Detail |
|---|---|
| **Course Title (English)** | Efficient Computing of Deep Neural Networks |
| **Course Title (Chinese)** | 深度学习中的高效计算方法 |
| **Instructor** | Bei Yu |
| **Department** | CSE Department, CUHK |
| **Email** | byu@cse.cuhk.edu.hk |
| **Session** | 2025 Summer (slides updated July 4, 2026) |
| **Local Archive** | `/Users/morrow/Documents/26summer/ECDNN` |

## Repository Structure

```
ECDNN/
├── 2026-slides/          # Lecture slide index (PDFs not included)
│   ├── Lec1-intro.pdf
│   ├── Im1-GEMM.pdf
│   ├── Im2-Direct-Conv.pdf
│   ├── Im3-Winograd.pdf
│   ├── Im4-Sparse-Conv.pdf
│   ├── Im5-CUDA.pdf
│   ├── Im7-TVM.pdf
│   ├── Mo1-pruning.pdf
│   ├── Mo2-Decomp.pdf
│   ├── Mo3-quant.pdf
│   ├── Mo4-BNN.pdf
│   ├── Mo5-KD.pdf
│   └── Mo6-NAS.pdf
└── legacy/               # Index of prior-session materials (files not included)
    ├── ECDNN_Assignment1.pdf
    ├── ECDNN_Assignment2.pdf
    ├── Answer_Assignment1.pdf
    ├── Answer_Assignment2.pdf
    ├── Lab1-slides.pdf
    ├── Lab1-report.pdf
    ├── Lab2-slides.pdf
    ├── Lab2-report.pdf
    ├── answer_lab1.zip
    └── answer_lab2.zip
```

## How to Obtain the Materials

- **Current 2026 slides**: Please contact the course instructor, **Bei Yu** (`byu@cse.cuhk.edu.hk`), CSE Department, CUHK, or refer to the official course distribution channel.
- **Legacy 2025 materials**: Available via [PKU Hub — plib course material #937](https://pkuhub.cn/material/937).

## Legacy Materials — Attribution

The `legacy/` entries refer to materials from the **ECDNN 2025 Summer** session:

- **Source / Origin:** [PKU Hub — plib course material #937](https://pkuhub.cn/material/937)
- **Contents:** Assignments, reference answers, lab slides and reports from the 2025 summer offering.
- **Please note:** Answer keys and lab solutions are provided by the original course staff. Use them responsibly and in accordance with any applicable academic-integrity policies.

## Syllabus Overview (from Lecture 1)

| Model Level (Mo) | Implementation Level (Im) |
|---|---|
| Mo1 Pruning | Im1 GEMM |
| Mo2 Decomposition | Im2 Direct Conv |
| Mo3 Quantization | Im3 Winograd |
| Mo4 BNN | Im4 Sparse Conv |
| Mo5 Knowledge Distillation (KD) | Im5 CUDA |
| Mo6 Neural Architecture Search (NAS) | Im7 TVM |

## Disclaimer

Slide content is from the course instructor (Bei Yu, CUHK); legacy materials are sourced from [PKU Hub — plib course material #937](https://pkuhub.cn/material/937). All rights belong to their respective authors. This repository is an educational index only and does not redistribute the original copyrighted files.
