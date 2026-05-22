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