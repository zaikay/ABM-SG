# network/metrics.py V4

"""
Network analysis metrics for the prosumer simulation.
"""
import numpy as np
import networkx as nx

class NetworkMetrics:
    """
    Analyzes network structures and adoption patterns.
    """
    def __init__(self, model):
        """
        Initialize the network metrics analyzer.
        
        Args:
            model: Mesa model instance with a network grid
        """
        self.model = model
        self.G = model.grid.G  # Get the underlying NetworkX graph
    
    def get_adoption_clusters(self):
        """
        Identify clusters of prosumer adoption.
        
        Returns:
            dict: Metrics about adoption clusters
        """
        # Create a subgraph of only prosumer nodes
        prosumer_nodes = []
        
        # Fixed approach to extract prosumer nodes
        for node, data in self.G.nodes(data=True):
            # Check if 'agent' is in data and properly access the agent object
            if 'agent' in data and hasattr(data['agent'], 'is_prosumer'):
                if data['agent'].is_prosumer:
                    prosumer_nodes.append(node)
        
        if not prosumer_nodes:
            return {"clusters": 0, "largest_cluster": 0, "avg_cluster_size": 0}
        
        # Create prosumer subgraph
        prosumer_subgraph = self.G.subgraph(prosumer_nodes)
        
        # Find connected components (clusters)
        clusters = list(nx.weakly_connected_components(prosumer_subgraph))
        
        return {
            "clusters": len(clusters),
            "largest_cluster": max([len(c) for c in clusters]) if clusters else 0,
            "avg_cluster_size": np.mean([len(c) for c in clusters]) if clusters else 0
        }
    
    def get_adoption_homophily(self):
        """
        Calculate homophily metrics for prosumer adoption.
        
        Returns:
            dict: Homophily metrics
        """
        # Count edges between different types of nodes
        prosumer_prosumer = 0
        prosumer_nonprosumer = 0
        nonprosumer_nonprosumer = 0
        
        for u, v in self.G.edges():
            # Safely access agent attributes
            u_data = self.G.nodes[u]
            v_data = self.G.nodes[v]
            
            if 'agent' in u_data and 'agent' in v_data:
                u_agent = u_data['agent']
                v_agent = v_data['agent']
                
                if hasattr(u_agent, 'is_prosumer') and hasattr(v_agent, 'is_prosumer'):
                    if u_agent.is_prosumer and v_agent.is_prosumer:
                        prosumer_prosumer += 1
                    elif not u_agent.is_prosumer and not v_agent.is_prosumer:
                        nonprosumer_nonprosumer += 1
                    else:
                        prosumer_nonprosumer += 1
        
        total_edges = prosumer_prosumer + prosumer_nonprosumer + nonprosumer_nonprosumer
        
        if total_edges == 0:
            return {"homophily_index": 0}
        
        # Calculate homophily index
        expected_prosumer_prosumer = (prosumer_prosumer + prosumer_nonprosumer/2) ** 2 / total_edges
        homophily_index = (prosumer_prosumer - expected_prosumer_prosumer) / (total_edges - expected_prosumer_prosumer) if total_edges > expected_prosumer_prosumer else 0
        
        return {"homophily_index": homophily_index}
    
    def get_income_class_adoption(self):
        """
        Calculate adoption rates by income class.
        
        Returns:
            dict: Adoption rates by income class
        """
        # Count households and prosumers by income class
        class_counts = {}
        prosumer_counts = {}
        
        for node, data in self.G.nodes(data=True):
            agent = data['agent']
            income_class = agent.income_class
            
            if income_class not in class_counts:
                class_counts[income_class] = 0
                prosumer_counts[income_class] = 0
            
            class_counts[income_class] += 1
            
            if agent.is_prosumer:
                prosumer_counts[income_class] += 1
        
        # Calculate adoption rates
        adoption_rates = {}
        for income_class in class_counts:
            if class_counts[income_class] > 0:
                adoption_rates[income_class] = prosumer_counts[income_class] / class_counts[income_class]
            else:
                adoption_rates[income_class] = 0
        
        return {
            "class_counts": class_counts,
            "prosumer_counts": prosumer_counts,
            "adoption_rates": adoption_rates
        }