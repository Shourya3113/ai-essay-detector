# Admissions Essay Dataset — Composition, Prompt Distribution & Ethical Framework

## Executive Summary

This document outlines the design principles, taxonomic composition, prompt distribution, anonymization protocol, and ethical guidelines governing the benchmark dataset ($N = 500$) created for the **AI Admissions Essay Detector**.

To ensure scientific validity and high ecological fidelity to real-world college admissions scenarios, the dataset captures the full continuum of student writing: authentic human personal statements, synthetic pure AI generations across multiple frontier LLMs, and hybrid essays produced via human-AI collaborative workflows.

```
                      DATASET STRATIFICATION (N = 500 ESSAYS)
   ┌──────────────────────────────────┬──────────────────────────────────┐
   │ Cohort                           │ Sample Count & Representation   │
   ├──────────────────────────────────┼──────────────────────────────────┤
   │ 1. Organic Human Essays (H)      │ 200 essays (40.0%)               │
   │    • Native English Writers      │   - 140 essays                   │
   │    • ESL / Non-Native Writers    │   - 60 essays (TOEFL/IELTS cohort)│
   │ 2. Synthetic Pure AI Essays (A)  │ 200 essays (40.0%)               │
   │    • OpenAI GPT-4o / Turbo       │   - 70 essays                    │
   │    • Anthropic Claude 3.5 Sonnet │   - 65 essays                    │
   │    • Google Gemini 1.5 Pro       │   - 45 essays                    │
   │    • Meta Llama 3 70B            │   - 20 essays                    │
   │ 3. Hybrid / AI-Polished (M)      │ 100 essays (20.0%)               │
   │    • Grammarly AI Flow Revision  │   - 35 essays                    │
   │    • ChatGPT Line Editing        │   - 35 essays                    │
   │    • Claude Structural Polish    │   - 30 essays                    │
   └──────────────────────────────────┴──────────────────────────────────┘
```

---

## 1. Dataset Taxonomy & Composition

### 1.1 Cohort H: Organic Human Baseline ($N = 200$)
- **Temporal Baseline**: 100 essays sourced from pre-November 2021 verified archival repositories (prior to the public release of ChatGPT/GPT-3.5) to guarantee zero synthetic contamination.
- **Contemporary Baseline**: 100 essays from 2023–2025 verified student authors who provided handwritten journals, version control histories (Google Docs revision logs), and signed integrity declarations.
- **Sub-cohort Representation**:
  - **STEM Research Statements ($n = 45$)**: Technical terminology, experimental methodologies, scientific exposition.
  - **Humanities & Creative Narratives ($n = 55$)**: Metaphorical prose, non-linear chronological structures, dialogues.
  - **Socioeconomic & Adversity Narratives ($n = 40$)**: Personal reflections on community, family, and hardship.
  - **ESL / International Applicants ($n = 60$)**: Verified non-native English speakers from diverse linguistic backgrounds (East Asian, South Asian, Hispanic, Eastern European, Middle Eastern).

### 1.2 Cohort A: Synthetic Pure AI Generations ($N = 200$)
Generated across the top 4 modern LLM families to prevent model-specific overfitting. 

Sampling parameters varied systematically across:
- Temperature $T \in [0.3, 1.0]$
- Top-p nucleus sampling $p \in [0.85, 0.98]$
- Prompt strategies: Zero-Shot, Persona-Conditioned, Few-Shot, and Adversarially Constrained.

### 1.3 Cohort M: Human-AI Hybrid & Polished ($N = 100$)
Authentic human baseline drafts modified using standard student-accessible AI workflows:
- **Grammarly AI / Wordtune**: Automated sentence restructuring, clausal smoothing, tone elevation.
- **ChatGPT Copyediting**: Prompted with *"Fix my grammatical errors, improve word choice, and enhance transitions."*
- **Claude Structural Revision**: Prompted with *"Elevate this personal statement to an Ivy League standard while maintaining my core story."*

---

## 2. Prompt Distribution & Templates

All synthetic generations were structured around the 7 official **Common Application Prompts** and typical institutional supplemental prompts.

### Prompt Catalog & Parameter Matrix

```
                      PROMPT DISTRIBUTION MATRIX
  ┌────────────────────────────────────────────────────────────┬───────┐
  │ Common App Prompt 1: Background, Identity, or Interest     │ 18.0% │
  │ Common App Prompt 2: Lessons from Obstacles & Failure      │ 16.0% │
  │ Common App Prompt 3: Reflecting on a Challenged Belief     │ 12.0% │
  │ Common App Prompt 4: Gratitude & Unexpected Appreciation   │ 10.0% │
  │ Common App Prompt 5: Personal Growth & Milestone Event     │ 18.0% │
  │ Common App Prompt 6: Topic or Idea That Captivates You     │ 14.0% │
  │ Common App Prompt 7: Open Topic of Choice                  │ 12.0% │
  └────────────────────────────────────────────────────────────┴───────┘
```

### Representative Prompt Engineering Templates

#### Template 1: Standard Zero-Shot Generation
```text
System: You are an expert college admissions consultant.
User: Write a compelling 650-word Common Application essay responding to the following prompt: 
"Some students have a background, identity, interest, or talent that is so meaningful they believe their application would be incomplete without it. If this sounds like you, then please share your story."
Focus on a student who developed an obsession with competitive robotics and community repair cafes. 
Maintain a formal, reflective academic tone suitable for elite universities.
```

#### Template 2: Persona-Conditioned High School Senior
```text
System: You are a high school senior applying to undergraduate computer science programs.
User: Write a personal statement about your experience building open-source educational software. 
Adopt an authentic, passionate voice. Reflect on both technical roadblocks and the emotional reward of seeing younger students use your tools. Avoid generic corporate buzzwords.
```

#### Template 3: Adversarial Evasion Prompt (Style Masking)
```text
System: You are a creative writer attempting to write in an organic, human cadence.
User: Write an essay about learning traditional culinary techniques from your grandmother. 
Constraints:
1. Deliberately alternate between very short punchy sentences (3-5 words) and complex multi-clause sentences (35+ words).
2. Avoid transition words like "Furthermore", "Moreover", "In conclusion", "It is important to note".
3. Use expressive punctuation like em-dashes (—) and parenthetical thoughts.
4. Include 1-2 colloquial expressions and specific sensory details.
```

#### Template 4: AI Polishing & Proofreading Workflow
```text
System: You are an academic editor.
User: Below is a draft of a student's college admissions essay. Please polish the prose:
- Correct all grammatical, punctuation, and syntactical errors.
- Elevate vocabulary choices to be more sophisticated and articulate.
- Smooth out transitions between paragraphs to maximize logical cohesion.
- Do not alter the underlying narrative facts.

[INSERT HUMAN ESSAY DRAFT]
```

---

## 3. Anonymization, PII Scrubbing & Privacy Protection

College admissions essays contain highly sensitive personal information, including names, family medical histories, socioeconomic backgrounds, immigration statuses, and identifiable academic institutions.

### 4-Stage De-Identification Protocol

```
 Raw Essay ──► [Stage 1: spaCy NER Extraction] ──► [Stage 2: Regex PII Scrubbing]
                                                            │
 Final Anonymized Dataset ◄── [Stage 4: Human Audit] ◄── [Stage 3: Entity Substitution]
```

1. **Named Entity Recognition (NER)**: Automated parsing using `en_core_web_trf` (spaCy transformer model) to tag `PERSON`, `ORG`, `GPE` (cities/countries), `DATE`, and `FACILITY`.
2. **Regex Pattern Sanitization**: High-precision matching for email addresses, phone numbers, social media handles, physical addresses, and school IDs.
3. **Synthetic Substitution**: Rather than leaving disruptive `[REDACTED]` tokens that distort language model perplexity, identified entities were systematically replaced with plausible synthetic counterparts:
   - Specific high schools $\rightarrow$ *"Oakridge High School"*, *"St. Jude Academy"*
   - Specific family members $\rightarrow$ *"my uncle"*, *"my grandmother"*
   - Specific regional towns $\rightarrow$ *"a small coastal town in Oregon"*
4. **Human Review**: 100% of human essays underwent manual line-by-line verification by two independent auditors prior to dataset ingestion.

---

## 4. Ethical Framework & Terms of Use

### 4.1 Responsible Use in University Admissions
The **AI Admissions Essay Detector** is built on the strict ethical principle that **automated tools must never make unassisted rejection or disciplinary decisions**.

1. **Diagnostic Triage Only**: Detector outputs should serve as an initial triage signal for human admissions counselors to conduct qualitative holistic reviews.
2. **Mandatory Human-in-the-Loop**: A high AI probability score triggers an invite for an oral interview or supplemental portfolio review, never an automatic application dismissal.
3. **Right to Explanation & Transparency**: Applicants flagged during evaluation must be provided with the feature evidence dossier (e.g., token perplexity map, phrase matches) and granted a standard administrative appeal channel.

### 4.2 Dataset Licensing & Open Science
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (**CC BY-NC 4.0**).
- **Intended Use**: This dataset is released strictly for non-commercial academic research, pedagogical benchmarking, and computational linguistics development.
- **Commercial Restrictions**: Commercial exploitation, training commercial ghostwriting tools to evade detection, or re-identifying student authors is strictly prohibited.

---

## 5. Limitations & Threats to Validity

1. **Domain Specificity**: The linguistic distributions in this dataset are calibrated for high school and undergraduate college admissions essays (500–800 words). Generalization to legal briefs, medical papers, or middle-school creative writing may require threshold recalibration.
2. **Frontier Model Evolution**: As future generative architectures (e.g., GPT-5, Claude 4) evolve richer syntactic sampling dynamics, continuous benchmarking against updated model families will be required.
3. **Adversarial Paraphrasing**: Heavy paraphrasing via specialized humanizers (combining back-translation with lexical synonym scramblers) can shift statistical markers toward the boundary decision region.
