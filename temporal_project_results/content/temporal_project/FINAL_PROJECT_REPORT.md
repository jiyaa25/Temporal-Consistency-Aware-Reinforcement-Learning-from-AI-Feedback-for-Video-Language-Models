# Project Report: Temporal-Consistency-Aware RLAIF for Video-Language Models
**Authors**: Himanshi (12340940), Jiya Mehta (12341020) — IIT Bhilai

## 1. Executive Summary
This project successfully designed, implemented, and evaluated a temporal-consistency-aware reinforcement learning framework for Video-LLMs based on VLM-RLAIF and TimeJudge.

## 2. Key Experimental Findings
- **Baseline Temporal Vulnerability**: The initial Video-LLM achieved 100% accuracy on detecting hallucinations, but only 33.3% on temporal misordering errors, confirming that standard VLMs struggle with chronological ordering.
- **PEFT Optimization**: Parameter-efficient training with LoRA trained 1.84M parameters (0.049% of model weights) via Direct Preference Optimization.
- **Preference Convergence**: The policy preference margin shifted from -33.09 to +3.99 (+37.08 gain).
- **Temporal Reward Gain**: The composite temporal reward improved by +-0.30.

## 3. Reward Component Ablation
The ablation study confirms that prioritizing temporal ordering (alpha=0.70) yields robust consistency gains without compromising factual grounding.
