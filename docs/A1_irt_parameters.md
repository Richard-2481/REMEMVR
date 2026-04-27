---
layout: default
title: "A1. IRT Item Parameters"
parent: Home
nav_order: 1
---

# A1. IRT Item Parameters

Item Response Theory (IRT) calibration was performed using a two-parameter logistic model with a two-pass purification procedure. Items with discrimination (*a*) < 0.4 or \|difficulty (*b*)\| > 3.0 were flagged for removal in the second pass.

[Download CSV]({{ site.baseurl }}/data/A1_irt_item_parameters.csv){: .btn .btn-primary }
[View on GitHub](https://github.com/Richard-2481/REMEMVR/blob/main/data/A1_irt_item_parameters.csv){: .btn }

## Summary

| | Retained | Removed | Total |
|---|---------|---------|-------|
| **What** | 24 | 5 | 29 |
| **Where** | 32 | 18 | 50 |
| **When** | 13 | 12 | 25 |
| **Total** | 69 | 35 | 104 |

## Columns

| Column | Description |
|--------|-------------|
| `item_name` | Item identifier (e.g., TQ_ICR-N-i1). Prefix encodes paradigm (IFR/ICR/IRE = item free recall/cued recall/recognition; RFR/RRE = room free recall/recognition). |
| `discrimination` | IRT discrimination parameter (*a*). Higher values indicate the item better differentiates between ability levels. |
| `difficulty` | IRT difficulty parameter (*b*). Higher values indicate harder items. |
| `domain` | Memory domain: what (item identity), where (spatial location), when (temporal order). |
| `retrieval_condition` | Retrieval paradigm: free_recall, cued_recall, or recognition. |
| `schema` | Schema congruence: common (room-neutral), congruent (room-appropriate), or incongruent (room-inappropriate). |
| `purified` | Purification status: retained (included in final analyses) or removed (excluded). |
| `exclusion_reason` | Reason for removal, if applicable (e.g., "\|b\| > 3.0", "a < 0.4"). |

## Notes

- Parameters are from the domain-stratified calibration (RQ 5.2.1), which provides domain-specific estimates.
- The two-pass procedure first fits all items, then removes those exceeding thresholds and refits.
- 69 of 104 items (66.3%) were retained in the purified set used for trajectory analyses.
- GitHub renders CSV files as interactive sortable tables.
