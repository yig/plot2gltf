import numpy as np
from plot2gltf import GLTFGeometryExporter

class GLTFDemo:
    def __init__(self):
        self.exporter = GLTFGeometryExporter()
    
    def demo_cube(self):
        """Create a colorful cube using triangles"""
        cube_vertices = np.array([
            # Front face
            [-0.5, -0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [ 0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5],
            # Back face
            [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [ 0.5,  0.5, -0.5],
            [-0.5,  0.5, -0.5],
        ])
        
        cube_faces = [
            [0, 1, 2], [0, 2, 3],  # Front
            [1, 5, 6], [1, 6, 2],  # Right
            [5, 4, 7], [5, 7, 6],  # Back
            [4, 0, 3], [4, 3, 7],  # Left
            [3, 2, 6], [3, 6, 7],  # Top
            [4, 5, 1], [4, 1, 0],  # Bottom
        ]
        
        self.exporter.add_triangles(cube_vertices, cube_faces)
        self.exporter.add_text([0, 0, 0.6], "Cube", size=0.4, color=(1, 1, 1))

    def demo_axes(self):
        """Create coordinate axes using lines"""
        axis_vertices = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        
        axis_edges = [
            [0, 1],  # X-axis
            [0, 2],  # Y-axis
            [0, 3]   # Z-axis
        ]
        
        self.exporter.add_lines(axis_vertices, axis_edges, color=(1, 0, 0))
        # Add axis labels
        self.exporter.add_text([1.1, 0, 0], "X", size=0.5, color=(1, 0, 0))
        self.exporter.add_text([0, 1.1, 0], "Y", size=0.5, color=(0, 1, 0))
        self.exporter.add_text([0, 0, 1.1], "Z", size=0.5, color=(0, 0, 1))

    def demo_spiral(self):
        """Create a spiral point cloud"""
        t = np.linspace(0, 6*np.pi, 50)
        spiral_points = np.column_stack([
            0.3 * np.cos(t) + 1.5,
            0.3 * np.sin(t) + 1.5,
            0.1 * t
        ])
        
        self.exporter.add_points(spiral_points, color=(1, 1, 0))
        self.exporter.add_text([1.5, 1.5, 2], "Spiral", size=0.5, color=(1, 1, 0))

    def demo_normals(self):
        """Demonstrate normal vectors"""
        normal_positions = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ]
        
        normal_directions = [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ]
        
        self.exporter.add_normals(normal_positions, normal_directions)
        self.exporter.add_text([0, 0, 1.2], "Normals", size=0.5)

    def demo_house(self):
        """Create a simple house shape"""
        house_vertices = [
            [2, 0, 0],    # 0: bottom left
            [3, 0, 0],    # 1: bottom right
            [3, 1, 0],    # 2: top right
            [2, 1, 0],    # 3: top left
            [2.5, 1.5, 0] # 4: roof top
        ]
        
        house_faces = [
            [0, 1, 2], [0, 2, 3],  # walls
            [3, 2, 4]              # roof
        ]
        
        self.exporter.add_triangles(house_vertices, house_faces, color=(0.8, 0.4, 0.2))
        self.exporter.add_text([2.5, 1.7, 0], "House", size=0.5, color=(0.8, 0.4, 0.2))

    def demo_spheres(self):
        """Demonstrate sphere points"""
        # Create a row of spheres with different sizes
        sphere_centers = [
            [-1, 2, 0],    # Small sphere
            [-0.5, 2, 0],  # Medium sphere
            [0, 2, 0],     # Large sphere
        ]
        
        # Different sized spheres
        self.exporter.add_spheres([sphere_centers[0]], radius=0.05, color=(1, 0, 0), unlit=False)  # Red
        self.exporter.add_spheres([sphere_centers[1]], radius=0.1, color=(0, 1, 0), unlit=False)   # Green
        self.exporter.add_spheres([sphere_centers[2]], radius=0.15, color=(0, 0, 1), unlit=False)  # Blue
        
        # Add labels
        self.exporter.add_text([-1, 2.3, 0], "Small", size=0.15, color=(1, 0, 0))
        self.exporter.add_text([-0.5, 2.3, 0], "Medium", size=0.15, color=(0, 1, 0))
        self.exporter.add_text([0, 2.3, 0], "Large", size=0.15, color=(0, 0, 1))
        
        # Create a grid of spheres to show positioning
        grid_size = 3
        spacing = 0.3
        for i in range(grid_size):
            for j in range(grid_size):
                center = [
                    1 + i * spacing,
                    2 + j * spacing,
                    0
                ]
                self.exporter.add_spheres([center], radius=0.05, color=(1, 0.5, 0))  # Orange
        
        self.exporter.add_text([1.3, 2.8, 0], "Sphere Grid", size=0.2, color=(1, 0.5, 0))

    def demo_cylinder_strip(self):
        """Demonstrate cylinder strips with different configurations"""
        # Create a helix curve for more interesting demonstration
        t = np.linspace(0, 4*np.pi, 40)
        helix_points = np.column_stack([
            2 + 0.5*np.cos(t),
            0.5*np.sin(t),
            0.3*t
        ])
        
        # Create three examples side by side
        # 1. Basic cylinder strip without spheres
        points1 = helix_points + [-2, 0, 0]  # Shifted left
        self.exporter.add_cylinder_strips(points1, radius=0.03, color=(0, 0.8, 0.8), 
                                        add_spheres=False)
        self.exporter.add_text([-2, 1, 0], "Without Spheres", size=0.5)
        
        # 2. Cylinder strip with spheres
        points2 = helix_points.copy()  # Center
        self.exporter.add_cylinder_strips(points2, radius=0.03, color=(0.8, 0.4, 0))
        self.exporter.add_text([2, 1, 0], "With Spheres", size=0.5)
        
        # 3. Thicker version
        points3 = helix_points + [2, 0, 0]  # Shifted right
        self.exporter.add_cylinder_strips(points3, radius=0.05, color=(0.4, 0.8, 0))
        self.exporter.add_text([4, 1, 0], "Thicker", size=0.5)

    def demo_normal_arrows(self):
        """Demonstrate normal arrows"""
        normal_points = [
            [3, 0, 0],
            [3.5, 0, 0],
            [4, 0, 0]
        ]
        normal_dirs = [
            [0, 0.5, 0],
            [0.2, 0.5, 0.2],
            [0, 0.5, 0.3]
        ]
        self.exporter.add_normal_arrows(
            normal_points, normal_dirs,
            shaft_radius=0.02,
            head_radius=0.04,
            color=(0.8, 0.2, 0.8)
        )
        self.exporter.add_text([3.5, 0.8, 0], "Normal Arrows", size=0.3)

    def demo_text_sizes(self):
        """Demonstrate different text sizes"""
        texts = [
            {"pos": [-1, -1, 0], "text": "Small", "size": 0.5},
            {"pos": [-1, -1, 1], "text": "Medium", "size": 0.8},
            {"pos": [-1, -1, 2], "text": "Large", "size": 1.2}
        ]
        
        for text_info in texts:
            self.exporter.add_text(
                text_info["pos"],
                text_info["text"],
                size=text_info["size"],
                color=(0, 1, 1)
            )

    def run(self, features=None):
        """
        Run the demo with specified features.
        
        Parameters:
        features : list of str or None
            List of features to demo. Available features:
            - 'cube': Basic cube with triangles
            - 'axes': Coordinate axes
            - 'spiral': Spiral point cloud
            - 'normals': Normal vectors
            - 'house': Simple house shape
            - 'spheres': Sphere points
            - 'cylinder_strip': Connected cylinders
            - 'normal_arrows': Normal arrows with cones
            - 'text_sizes': Text size comparison
            If None, all features will be demonstrated.
        """
        all_features = [
            'cube', 'axes', 'spiral', 'normals', 'house',
            'spheres', 'cylinder_strip', 'normal_arrows', 'text_sizes'
        ]
        
        features = features or all_features
        
        # Validate features
        invalid_features = set(features) - set(all_features)
        if invalid_features:
            raise ValueError(f"Invalid features: {invalid_features}. "
                           f"Available features are: {all_features}")
        
        # Run selected demos
        demo_map = {
            'cube': self.demo_cube,
            'axes': self.demo_axes,
            'spiral': self.demo_spiral,
            'normals': self.demo_normals,
            'house': self.demo_house,
            'spheres': self.demo_spheres,
            'cylinder_strip': self.demo_cylinder_strip,
            'normal_arrows': self.demo_normal_arrows,
            'text_sizes': self.demo_text_sizes
        }
        
        for feature in features:
            demo_map[feature]()
        
        # Save the result
        self.exporter.save("demo_scene.gltf")
        print(f"Demo scene saved as 'demo_scene.gltf' with features: {features}")

if __name__ == "__main__":
    import sys
    
    # Get features from command line arguments, if provided
    features = sys.argv[1:] if len(sys.argv) > 1 else None
    
    demo = GLTFDemo()
    demo.run(features)
