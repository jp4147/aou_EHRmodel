This repository contains code to reproduce analyses from:

[Toward Scalable Early Cancer Detection: Evaluating EHR-Based Predictive Models Against Traditional Screening Criteria](https://arxiv.org/abs/2511.11293)

We develop and evaluate EHR-based predictive models for identifying high-risk individuals across multiple cancer types using OMOP-formatted data.

Key components:

Cancer cohort identification using OMOP concept hierarchy
GPT-4o–based classification of cancer-related conditions
Risk prediction using XGBoost and EHR foundation models
Clinical utility evaluation using lift, PPV, and NNS

To identify and categorize cancer-related diagnoses in OMOP-structured EHR data, we first extracted all descendant concepts of “malignant neoplastic disease” (OMOP concept ID 443392). We then used a GPT-4o–based prompt approach to classify these concept names into predefined cancer types (e.g., colorectal, liver, skin). The prompt and cancer type definitions were refined iteratively with clinician input
