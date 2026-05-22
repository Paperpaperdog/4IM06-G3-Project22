This file notes our progress each week: meetings, TODO, progress...

# Meet May 6th (W1)
First meeting with professor Quentin Bammey, communicating project details, setting up this week's TODO, fixing next meeting time (next week)

### Main idea of project: 
Develop methods to distinguish the **jpeg compression** and **ai generation + resampling images**.

### General project steps:
1. Detection of resampling: distinguishing 2-multiples upsampling remains difficult to do, confused with jpeg compressed 
2. Estimate the actual resampling factor
3. A contrario analysis 

### TODO this week:
- Set up discord group to communicate efficiently. Set up Github repository
- Read paper, understand and implement it [paper_link](https://bammey.com/resampling_detection.pdf)

# Meet May 15th (W2)
Follow-up meeting: reviewed progress on paper reproduction and discussed next research directions.

### Main idea of project:
Continue implementing the project baseline and improve understanding of method details.

### General project steps:
1. Reproduce the baseline implementation from the paper.
2. Validate implementation behavior with controlled experiments.
3. Understand and document key details (assumptions, limits, and failure cases).

### Future work / inspiration:
1. Resampling ambiguity: check whether different original sizes/states can map to the same peak after conversion to a common target (e.g., `a -> c` and `b -> c`).
2. JPEG issue: investigate the previously mentioned false-positive behavior related to possible 8x compression/resampling confusion.
3. New optimization direction: propose and test additional ideas if promising.

### TODO this week:
- Run controlled experiments to verify baseline behavior.
- Improve understanding of implementation details and record observations.
- Prepare a short list of possible future-work experiments.

# Meet May 22th (W3)
This meeting focused on scope control and next-step direction choice.

### Questions asked and answers:
1. Appendix E in the paper mentions the non-aligned case. Should we implement it now?
- Answer: No. Keep the simplest aligned implementation first (the current one).

2. For RAISE-1K, TIFF images are slow to process. Should we add extra acceleration code from the paper?
- Answer: No for now. Convert TIFF to PNG and continue experiments.

### Main idea of project:
Stay focused on implementing and validating the aligned baseline, while running small pilot studies for future directions.

### General project steps:
1. Keep current aligned pipeline as the main implementation track.
2. Use TIFF -> PNG preprocessing for practical runtime.
3. Run short pilot experiments to decide the next main direction.

### Future work / inspiration:
1. **Idea 1 (JPEG / x8 / x16):** test whether compression-related periodic signals can be distinguished from true resampling cues.
2. **Idea 2 (CNN with k = -1, 0, 1):** test which correlation pattern is strongest, while temporarily neglecting minor low-probability cases.
3. For Idea 2, focus not only on peak location but also on other mathematical aspects (e.g., peak height, width, shape, side structures) for fast and direct analysis before deep learning.

### TODO this week:
- Convert selected RAISE TIFF samples to PNG and prepare a clean experiment subset.
- Start a first pilot for Idea 1 (JPEG/x8/x16 confusion check).
- Start a first pilot for Idea 2 (k = -1, 0, 1 comparison, including shape-related metrics beyond location).
- Compare pilot outcomes and decide one main direction for the next stage.
