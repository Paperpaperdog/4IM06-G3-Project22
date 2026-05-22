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

# Meet May 13th (W2) (todo, needs notes more precisely)
先复现，然后他建议了两个方向去研究：
1. 假如我们重采样两张图获得了同一个结果（从a到c，从b到c），有可能ab不是一个size，但是到c之后都是同一个peak表现。这会导致我们还原出现问题（不知道具体是哪个size），所以我们需要具体做实验，看看是不是这样的，然后去寻找另一个观察角度（maybe别的频谱图）去区分他们
2.之前他说的那个角度：解决jpeg压缩在我们这里会变成8倍压缩误报的问题
3. 自己再找出新的优化方向
