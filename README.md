# ECDNN — Efficient Computing of Deep Neural Networks

> 深度学习中的高效计算方法

Personal course-material archive for the **Efficient Computing of Deep Neural Networks** summer course.

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
├── 2026-slides/          # Lecture slides for the current session
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
└── legacy/               # Materials from previous ECDNN offerings
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

## Legacy Materials — Attribution & Usage

The files under `legacy/` are retained from the **ECDNN 2025 Summer** session. They are included here for reference and self-study only:

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

Slide content is from the course instructor; legacy materials are sourced from [PKU Hub — plib course material #937](https://pkuhub.cn/material/937). All rights belong to their respective authors. This repository is shared for educational reference only.
