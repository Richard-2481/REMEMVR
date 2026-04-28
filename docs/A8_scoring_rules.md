---
layout: default
title: "A8. Scoring Rules and Foil Materials"
parent: Home
nav_order: 9
---

# A8. Scoring Rules and Foil Materials

This appendix documents the complete test materials used in the REMEMVR study, including item specifications, scoring criteria, verbal and visual foils, and the test question bank.

---

## 1. Item Map

Each of the four VR rooms contained 6 interactive items that participants picked up from a starting location and placed at a designated end location. Positions labelled A-D are furniture locations within the room; positions 1-4 are numbered pillars.

[Download CSV]({{ site.baseurl }}/data/A8_item_map.csv){: .btn .btn-primary }
[View on GitHub](https://github.com/Richard-2481/REMEMVR/blob/main/data/A8_item_map.csv){: .btn }

### Bathroom

| Item | Start | Start Label | End | End Label |
|------|-------|-------------|-----|-----------|
| Ducky | in | B | in | D |
| TV Remote | out | 2 | in | B |
| T Brushes | out | 3 | out | 4 |
| Book | in | C | out | 3 |
| Hammer | in | A | in | C |
| Glasses | out | 1 | in | A |

### Bedroom

| Item | Start | Start Label | End | End Label |
|------|-------|-------------|-----|-----------|
| Alarm Clock | in | D | out | 4 |
| Magazine | in | C | in | D |
| Picture Frame | out | 1 | in | C |
| Frying Pan | in | B | in | A |
| Phone | out | 4 | out | 2 |
| Knife | out | 3 | in | B |

### Kitchen

| Item | Start | Start Label | End | End Label |
|------|-------|-------------|-----|-----------|
| Letters | out | 2 | in | C |
| Cutlery | in | A | out | 2 |
| Toaster | out | 4 | in | A |
| Motor Oil | out | 1 | out | 3 |
| Keys | in | D | in | B |
| Toilet Paper | in | B | in | D |

### Living Room

| Item | Start | Start Label | End | End Label |
|------|-------|-------------|-----|-----------|
| Drill | in | D | in | B |
| Sleep Mask | out | 4 | out | 1 |
| Lamp | out | 2 | in | D |
| Handbag | in | A | out | 2 |
| Laptop | in | C | in | A |
| Newspaper | out | 3 | in | C |

*Note:* "in" = furniture location within the room (A-D); "out" = numbered pillar (1-4). Items were counterbalanced across participants (see Counterbalancing below).

---

## 2. Verbal Foils

Each of the 24 interactive items had 13 semantically similar foil words used as alternatives in cued recall questions. Items are grouped by position (i1-i6), with each room receiving one item per position.

[Download CSV]({{ site.baseurl }}/data/A8_verbal_foils.csv){: .btn .btn-primary }
[View on GitHub](https://github.com/Richard-2481/REMEMVR/blob/main/data/A8_verbal_foils.csv){: .btn }

---

## 3. Visual Foils

Visual foils used in recognition tests are provided in the [`assets/images/`](https://github.com/Richard-2481/REMEMVR/tree/main/assets/images) directory:

| Folder | Files | Content |
|--------|-------|---------|
| [`Items/`](https://github.com/Richard-2481/REMEMVR/tree/main/assets/images/Items) | 189 | Object foil images for item recognition (6 visually similar foils per item) |
| [`Items/Test/`](https://github.com/Richard-2481/REMEMVR/tree/main/assets/images/Items/Test) | 20 | Correct target item images |
| [`Portraits/`](https://github.com/Richard-2481/REMEMVR/tree/main/assets/images/Portraits) | 43 | Portrait stimuli (adult, grandparent, teenager, toddler; 8 per age group) |
| [`Landscapes/`](https://github.com/Richard-2481/REMEMVR/tree/main/assets/images/Landscapes) | 39 | Landscape/scene images (beach, desert, forest, mountain, sea, snow, valley) |
| [`Rooms/`](https://github.com/Richard-2481/REMEMVR/tree/main/assets/images/Rooms) | 13 | VR room photos and floor plan layouts |
| [`Cues/`](https://github.com/Richard-2481/REMEMVR/tree/main/assets/images/Cues) | 4 | Numbered pillar images used as spatial cues |
| [`Colours/`](https://github.com/Richard-2481/REMEMVR/tree/main/assets/images/Colours) | 12 | Colour and number label stimuli |

---

## 4. Test Questions

The complete question bank administered to participants across all test sections. Metadata collection rows (user IDs, timestamps, counters) have been excluded.

[Download CSV]({{ site.baseurl }}/data/A8_test_questions.csv){: .btn .btn-primary }
[View on GitHub](https://github.com/Richard-2481/REMEMVR/blob/main/data/A8_test_questions.csv){: .btn }

**174 participant-facing questions across 8 sections:**

| Section Prefix | Section | Questions |
|----------------|---------|-----------|
| Slp | Sleep questionnaire | 15 |
| RFR | Room Free Recall | 24 |
| IFR | Item Free Recall | 53 |
| TCR | Task Cued Recall | 17 |
| ICR | Item Cued Recall | 71 |
| RRE | Room Recognition | 30 |
| IRE | Item Recognition | 71 |
| Str | Strategy and feedback | 19 |

---

## 5. Item Counterbalancing

The counterbalancing scheme ensured an even distribution of item types across rooms and positions.

[Download CSV]({{ site.baseurl }}/data/A8_item_counterbalance.csv){: .btn .btn-primary }
[View on GitHub](https://github.com/Richard-2481/REMEMVR/blob/main/data/A8_item_counterbalance.csv){: .btn }

---

## 6. Scoring Rules

### Free-Text Responses
- Responses limited to 1-2 words
- Misspellings accepted if conceptually correct
- Alternative language terms accepted if conceptually correct
- All 5,200 VR free-text responses scored by a single rater (the candidate)
- Pre-specified scoring rules provided clear decision criteria for each item

### Multiple-Choice Responses
- Binary scoring: 1 (correct) or 0 (incorrect)
- Partial credit was initially explored (adjacent spatial locations = 0.5, ordinal answers once removed = 0.5, twice removed = 0.25) but abandoned due to edge-position bias affecting 17.5% of responses

### Confidence Ratings
- 5-point Likert scale (1 = Guess/No Memory to 5 = Absolutely Certain)
- Rescaled to continuous 0-1 metric: 1 = 0, 2 = 0.25, 3 = 0.50, 4 = 0.75, 5 = 1.0

---

## 7. Procedure Instructions Booklet

The complete cognitive testing procedure manual (RAVLT, RPM administration and scoring protocols).

<object data="{{ site.baseurl }}/assets/pdfs/procedure instructions booklet.pdf" type="application/pdf" width="100%" height="800px">
  <p>Unable to display PDF. <a href="{{ site.baseurl }}/assets/pdfs/procedure instructions booklet.pdf">Download the Procedure Instructions Booklet</a>.</p>
</object>
