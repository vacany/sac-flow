# Ideas and stuff for future and clarity

- Development layout
- Visuals for images on server
- Drawio inside
- output from drawio to notebook
- What is the final output? 
  - Codes in scripts is the final one
  - development is in notebooks

- Days in notebooks?
- jupyter cells can be easily transfered to functions even unexpectadly

- Smoothness increase convergence Speed? - To paper?
  - init time is heavy though

- Simulator of tiny objects but with real physics
  - top down view annotation with colors?
  - project with boys?
  - problem: how to design dense sensors?

- Find use case first! That helps to define problem

# NN of flow should be also smooth!
  - object dont move towards each other
  - should deal with local minima of flow inside one structure
  - It solve probably both of those cases

# Iteration of experiments
  - Reproducible
  - Arguments should input also why, what to achieve 
  - store codes
  - store metrics, meta data, data for visuals, inference
  - When checking exps, navigate to folder and check the results and parameters
  - choose exps to compare
  - always state baseline and use-case solution
    - would like to see baseline results when looking at this --- this exp run baseline within?
  
  - start with: 
    - argparser [x]
    - logging [x]
    - bash execution [x]
    - continue with future improvements 

# Experiments
 - Parameters
   - Iterations, Speed, metrics, nbr of points
   - exp variance
   - Range of lidar (NSF uses original 80m?)
   - 
# From NSF
    - "Note that we used the raw point cloud from the lidar
    sensor and did not crop the data to a small range."
    - "Practically, we chose 2m as a threshold to Chamf to
    eliminate large point distance in baseline, not DT"
# Visibility
    - should be rigid transformation?
    - Go by yaw and pitch and keep instance until divided by huge depth

- continue with exp?
- new vis with dbscan? better interpretable
  - can be also done with KNN (one instace = one KNN)
- iterable parameters inside model?
- 
# Valeo
    - show normals estimate and intensity
