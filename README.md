### All data sets used in this experiment are in the dataset folder:

In real_data, a real data set of Beijing is used as an example. 
The longitude, latitude and semantic attributes of each node are recorded in real_data.csv, 
and the road network distance between each two nodes is recorded in od_distance_metrix.csv.

TSPLIB records the conventional TSP data set, and each csv file records the node's horizontal and vertical coordinates (not latitude and longitude).

### Algorithms adapted to TSPLIB data sets are stored in the folder Algorithms for TSPLIB data sets:

- **GA-LKH.py** uses genetic algorithm to select the optional nodes that meet the semantic constraints, and then uses LKH algorithm to obtain the path.

- **HC-LKH.py** uses the mountain climbing algorithm to select the optional nodes that meet the semantic constraints, and then uses the LKH algorithm to obtain the path.

- **PSO-LKH.py** uses particle swarm optimization algorithm to select the optional nodes that meet the semantic constraints, and then uses LKH algorithm to obtain the path.

- **SRC-LKH.py** is a method proposed by us, which deeply analyzes the data set by aggregating and extracting high-quality subsets, selects appropriate nodes to satisfy semantic constraints, and uses LKH algorithm to obtain the path.

    

### Algorithms that adapt to real data sets are stored in the folder Algorithms for real data:
    
- **Real_Data_GA-LKH.py** is a GA-LKH algorithm adapted to real data sets.
    
- **Real_Data_HC-LKH.py** is an HC-LKH algorithm adapted to real data sets.
    
- **Real_Data_PSO-LKH.py** is a PSO-LKH algorithm adapted to real data sets.

- **Real_Data_SRC-LKH.py** is a SRC-LKH algorithm adapted to real data sets.
    
    
