# Introduction to Computation in Information Science

## 1. Computational Techniques

1. Computational techniques are methods used for problem-solving. From something that has happened, computations help us understand the world in which we live.
2. Computational Techniques can be associated with or without data.
3. **Modelling** is one of the computational techniques we focus, besides that, we also focus **Simulation or Sampling, and Optimization**.
4. **Information Science** is an interdisciplinary field focused on the effective management of information, including its collection, organization, storage, retrieval, and **analysis**. It involves the study and application of processes and techniques to manage information in various formats, including **digital, textual, and multimedia.**

5. It follows this formula: Computational Techniques + Information Science = Computation in Information.

## 2 P and NP Problem

1. **P**: P stands for **Polynomial Time**. A problem is in the class P if there exists an algorithm that can solve any instance of the problem in polynomial time, which is expressed mathematically as $O(n^k)$, where $n$ is the size of the input and k is a constant.

2. **NP**: NP stands for **Nondeterministic Polynomial Time**. A problem is in the class NP if, given a potential solution, we can verify whether it’s correct in polynomial time. However, finding the solution itself might be computational intractable.

::: tip Examples of P and NP Problem

1. For **P Problem**: Insertion sort is a P Problem, since its time complexity is $O(n^2)$.

2. For **NP Problem**: Traveling Salesman Problem, TSP（旅行商问题）. Given a list of cities and the distances between each pair of cities, the problem is to find the shortest possible route that visits each city exactly once and returns to the origin city. 

	> You can check whether a given route is the shortest possible route in polynomial time, but finding the shortest route itself is computationally intractable. Its complexity is $O(n!)$.

	```python
	from itertools import permutations
	
	def traveling_salesman_problem(graph, start):
		# 获取所有城市的索引
		vertices = list(range(len(graph)))
		vertices.remove(start)
		
		# 生成所有可能的路径
		min_path = float('inf')
		for perm in permutations(vertices):
			current_pathweight = 0
			k = start
			for j in perm:
				current_pathweight += graph[k][j]
				k = j
			current_pathweight += graph[k][start]
			
			# 更新最短路径
			min_path = min(min_path, current_pathweight)
		
		return min_path
	
	# 示例用法
	graph = [
		[0, 10, 15, 20],
		[10, 0, 35, 25],
		[15, 35, 0, 30],
		[20, 25, 30, 0]
	]
	start = 0
	print(traveling_salesman_problem(graph, start))  # 输出: 80
	```


:::



## 3 Computational Thinking

1. **Computational thinking** refers to thought processes involved in **formulating problems and their solutions** so that the solutions are represented in a form that can be effectively carried out by an information processing agent

2. From Problems abstracted by math or logic, to its solutions, no matter what form it is, say model, algorithm or simulation, the problem-solving process is called **Computational Thinking**.

3. It consists the following steps.

	> 1. **Decomposition**. Break up a complex problem into smaller, more solvable, more manageable problems.
	> 2. **Data Preparation**. Ensure that the raw data is transformed into a clean, consistent, and structure format required for analysis or further processing.
	> 	1. **Data Collection**: Collect raw data from source domain.
	> 	2. **Data Loading**: Load data into experimental environment.
	> 	3. **Data Cleaning**: Identify and correct errors or inconsistencies. For example, remove duplicates, handle missing values or duplicates, correct data entry errors, or inconsistencies.
	> 		- Missing Values: Use `NULL, NaN, None`to label them.
	> 		- Duplicated: Uniquify them.
	> 		- Inconsistencies: Check the type and format of them.
	> 	4. **Data Transformation**: Transform data into a suitable format for analysis. Data with different scales may affect the analysis.
	> 3. **Pattern Recognition**. For current problem,. select appropriate computations, which are usually integrated in Math equations or models. Identifying patterns, trends, or relationships within the digitized data could facilitated the problem-solving process.
	> 4. **Algorithm Design**. Algorithm Design: design an algorithm, which follows a step-by-step strategy, to solve the computational problem.

	## 4 Computational Modelling

	1. It consists with 4 steps: Physical or Cyber Reality -> Observations (Data) -> Computational Model -> Decisions or Knowledge.
	2. System: a group of interacting or interrelated elements that act according to a set of rules to form a unified whole.
	3. System consists of:
		1. Parameters: quantities that are fixed or can be controlled
		2. Variables: quantities that are determined by mathematical or logical relationships
		3. State variables: the minimum set of variables to completely describe a system at a point in time
	4. System can be classified into:
		1. Static: state variables are independent of time
		2. Dynamic: state variables are a function of time
			1. Continuous time: states are continuous function of time
			2. Discrete time: states are only defined at certain time points
			3.  Combined: system contains both continuous and discrete variables
		3. Continuous State: state variables can take on values from a continuous range.
		4. Discrete State: state variables can take on values from a discrete range.
		5. Deterministic:  state variables can be predicted with certainty
		6. Stochastic: state variables include some source of randomness
	5. Process-Driven and Data-Driven
		1.  A process-based model is the mathematical (and normally computer-based) representation of one or several processes characterizing the functioning of well-delimited physical or social phenomena.
		2. The data-driven models build relationships between input and output data, without worrying too much about the underlying processes, using statistical / machine learning techniques
	6. Steps in the computational modelling process:
		1. Analyze the problem: List system properties.
		2. Formulate a model
		3. Translate the model to computer and solve: Platform used will depend on the focused system.
		4. Verify and validate the model’s solution. 
			1. Verify: make sure the program runs correctly, there are no bugs and no programming errors
			2. Validation: determine if the computational experiment mirrors the desired detail of an experiment conducted on a real system (model results vs. ground truth)
		5. Design experiments: Specify what is to be measured, Insights from preliminary results may help the improvement,  Result interpretation and analysis
		6. Document and report on the model: Model design and assumptions, Model solution technique, algorithm summary, Verification and validation technique, Results, analysis and conclusions, Maintain the model
