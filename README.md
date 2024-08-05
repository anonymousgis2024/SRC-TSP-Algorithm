All data sets used in this experiment are in the dataset folder:

    In real_data, a real data set of Beijing is used as an example. 
    The longitude, latitude and semantic attributes of each node are recorded in real_data.csv, 
    and the road network distance between each two nodes is recorded in od_distance_metrix.csv.

    TSPLIB records the conventional TSP data set, 
    and each csv file records the node's horizontal and vertical coordinates (not latitude and longitude).

Algorithms adapted to TSPLIB data sets are stored in the folder Algorithms for TSPLIB data sets:

    SRC-LKH.py is a method proposed by us, which deeply analyzes the data set by aggregating and extracting high-quality subsets, 
    selects appropriate nodes to satisfy semantic constraints, and uses LKH algorithm to obtain the path.

    SRC-LKH without NC.py skips the aggregation process and treats each semantic node as a separate cluster.
    The rest are the same as SRC-LKH.
    
    SRC-LKH without SPR.py skips the process of subsetting the cluster and only needs to calculate the cost of the cluster. 
    The rest are the same as SRC-LKH.
    
    SRC-LKH without Pert.py SRC-LKH without Pert.py skips the process of constructing perturbing candidate sets and path optimization.
    The rest are the same as SRC-LKH.

Algorithms that adapt to real data sets are stored in the folder Algorithms for real data:
    
    Real_Data_scen_1_SRC-LKH.py applies to scenario 1.
    The first application scenario is described as follows: A tourist with a child plans a 4-day Recreational Vehicle (RV) trip to Beijing. As shown in Figure 11(a), 14 scenic locations are designated as mandatory locations. In addition to these required sites, the tourists also wish to visit one location from each of the following six categories: children's amusement parks, traditional cultural centers, renowned university campuses, pedestrian streets, aquariums, and science museums. The objective is to select one location from each of these six semantic categories while minimizing the total distance traveled, including visits to all 14 mandatory scenic locations and returning to the start location.
    
    Real_Data_scen_2_SRC-LKH.py applies to scenario 2.
    The second scenario evaluates distance costs by comparing the number of semantic categories of locations. This analysis aids tourists in determining whether visiting more semantically diverse locations significantly increases travel distance. Consider a tourist who wishes to visit multiple locations from 7 semantic categories (including scenic areas as a semantic category), with only one location chosen per category. The question arises: Is there a substantial increase in distance cost when selecting 5 categories versus 7? To address this, we employed the path planning method proposed in this study.
    
    Real_Data_scen_3_SRC-LKH.py applies to scenario 3.
    The third scenario investigates the impact of different semantic location combinations on distance costs when tourists select four out of six semantic categories to visit. This analysis aims to identify the optimal combination that minimizes overall travel distance. The study utilized input data consisting of 1 start/end location and 111 locations across 6 semantic categories. Given the selection criteria of 4 out of 6 categories, 15 possible combinations were analyzed, with path planning conducted for each combination. Figure 13 illustrates four representative results, while Figure 14 presents the distance costs for all 15 combinations. 

    
    
    