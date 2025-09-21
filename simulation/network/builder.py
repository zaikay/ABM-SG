# network/builder.py V4

"""
Social network creation for the prosumer simulation.
"""
import numpy as np
import networkx as nx
from mesa.space import NetworkGrid

class NetworkBuilder:
    """
    Builds social networks for household interactions with efficient neighbor storage.
    """
    def __init__(self, model, config):
        """
        Initialize the network builder.
        
        Args:
            model: Mesa model instance
            config: Simulation configuration
        """
        self.model = model
        self.config = config
        self.neighbors_per_household = config["neighbors_per_household"]
    
    def create_spatial_network(self, households):
        """
        Create a social network based on spatial proximity.
        ENHANCED: Also stores neighbor lists directly on household objects for efficiency.
        
        Args:
            households: List of household agents
            
        Returns:
            NetworkGrid: Mesa network grid with enhanced household neighbor data
        """
        print(f"Building spatial network for {len(households)} households...")
        
        # Create a new directed graph
        G = nx.DiGraph()
        
        # Add nodes (households) to the graph
        for household in households:
            G.add_node(household.unique_id, agent=household)
        
        # Assign spatial positions to households (in a 2D grid)
        num_households = len(households)
        grid_size = int(np.ceil(np.sqrt(num_households)))
        
        positions = {}
        for i, household in enumerate(households):
            row = i // grid_size
            col = i % grid_size
            
            # Add some random jitter to positions
            jitter_x = np.random.uniform(-0.2, 0.2)
            jitter_y = np.random.uniform(-0.2, 0.2)
            
            positions[household.unique_id] = (col + jitter_x, row + jitter_y)
            household.pos = (col + jitter_x, row + jitter_y)
        
        # ENHANCED: Build neighbor relationships and store on households
        for household in households:
            neighbor_data = self._find_and_store_neighbors(
                household, households, positions
            )
            
            # Store neighbor information directly on household
            household.spatial_neighbors = neighbor_data['neighbors']
            household.spatial_neighbor_ids = neighbor_data['neighbor_ids']
            
            # Create graph connections (for Mesa NetworkGrid compatibility)
            for neighbor_household, distance in household.spatial_neighbors:
                # Create bidirectional connection with distance-based weight
                weight = 1.0 / distance if distance > 0 else 1.0
                G.add_edge(household.unique_id, neighbor_household.unique_id, weight=weight)
                G.add_edge(neighbor_household.unique_id, household.unique_id, weight=weight)
        
        # Create and return the network grid
        network_grid = NetworkGrid(G)
        
        print(f"✅ Spatial network built:")
        print(f"   {len(households)} households")
        print(f"   {self.neighbors_per_household} neighbors per household")
        print(f"   {len(G.edges())} total network connections")
        
        return network_grid
    
    def _find_and_store_neighbors(self, target_household, all_households, positions):
        """
        Find neighbors for a household and prepare storage data.
        
        Args:
            target_household: Household to find neighbors for
            all_households: All households in simulation  
            positions: Dictionary of household positions
            
        Returns:
            dict: Neighbor data ready for storage on household
        """
        target_pos = positions[target_household.unique_id]
        distances = []
        
        # Calculate distances to all other households
        for other_household in all_households:
            if other_household.unique_id != target_household.unique_id:
                other_pos = positions[other_household.unique_id]
                
                # Euclidean distance
                dist = np.sqrt((target_pos[0] - other_pos[0])**2 + 
                              (target_pos[1] - other_pos[1])**2)
                distances.append((other_household, dist))
        
        # Sort by distance and select closest neighbors
        distances.sort(key=lambda x: x[1])
        closest_neighbors = distances[:self.neighbors_per_household]
        
        # Prepare data for household storage
        neighbor_ids = {neighbor.unique_id for neighbor, _ in closest_neighbors}
        
        return {
            'neighbors': closest_neighbors,      # [(household, distance), ...]
            'neighbor_ids': neighbor_ids,        # {id1, id2, ...} for fast lookup
            'avg_distance': sum(d for _, d in closest_neighbors) / len(closest_neighbors) if closest_neighbors else 0
        }
    
    def get_network_statistics(self, households):
        """
        Calculate network statistics for analysis and validation.
        
        Args:
            households: List of household agents with built networks
            
        Returns:
            dict: Network statistics
        """
        if not households or not hasattr(households[0], 'spatial_neighbors'):
            return {"error": "Networks not built yet"}
        
        total_connections = sum(len(h.spatial_neighbors) for h in households)
        distances = []
        
        for household in households:
            for neighbor, distance in household.spatial_neighbors:
                distances.append(distance)
        
        # Check network symmetry
        symmetric_connections = 0
        total_checked = 0
        
        for household in households:
            for neighbor, _ in household.spatial_neighbors:
                total_checked += 1
                if (hasattr(neighbor, 'spatial_neighbor_ids') and 
                    household.unique_id in neighbor.spatial_neighbor_ids):
                    symmetric_connections += 1
        
        return {
            'total_households': len(households),
            'total_connections': total_connections,
            'avg_connections_per_household': total_connections / len(households),
            'avg_distance': sum(distances) / len(distances) if distances else 0,
            'min_distance': min(distances) if distances else 0,
            'max_distance': max(distances) if distances else 0,
            'network_symmetry_rate': symmetric_connections / total_checked if total_checked > 0 else 0,
            'positions_assigned': sum(1 for h in households if hasattr(h, 'pos') and h.pos is not None)
        }