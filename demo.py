import numpy as np
from plot2gltf import GLTFGeometryExporter

def create_demo_scene():
    exporter = GLTFGeometryExporter()
    
    # 1. Create a colorful cube using triangles
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
    
    # Add cube with auto-generated colors
    exporter.add_triangles(cube_vertices, cube_faces)
    exporter.add_text([0, 0, 0.6], "Cube", size=0.4, color=(1, 1, 1))
    
    # Size comparison text examples
    texts = [
        {"pos": [-1, -1, 0], "text": "Small", "size": 0.3},
        {"pos": [-1, -1, 0.5], "text": "Medium", "size": 0.5},
        {"pos": [-1, -1, 1], "text": "Large", "size": 0.7}
    ]
    
    # 2. Create coordinate axes using lines
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
    
    # Add coordinate axes with specific colors
    exporter.add_lines(axis_vertices, axis_edges, color=(1, 0, 0))  # Red for axes
    
    # Add labels for axes with larger size
    exporter.add_text([1.1, 0, 0], "X", size=0.3, color=(1, 0, 0))
    exporter.add_text([0, 1.1, 0], "Y", size=0.3, color=(0, 1, 0))
    exporter.add_text([0, 0, 1.1], "Z", size=0.3, color=(0, 0, 1))
    
    

    
    # 3. Create a point cloud in a spiral pattern
    t = np.linspace(0, 6*np.pi, 50)
    spiral_points = np.column_stack([
        0.3 * np.cos(t) + 1.5,
        0.3 * np.sin(t) + 1.5,
        0.1 * t
    ])
    
    # Add points with a specific color
    exporter.add_points(spiral_points, color=(1, 1, 0))  # Yellow points
    exporter.add_text([1.5, 1.5, 2], "Spiral", size=0.5, color=(1, 1, 0))
    
    
    # 4. Add some normal vectors to demonstrate orientation
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
    
    # Add normals with auto-generated colors
    exporter.add_normals(normal_positions, normal_directions)
    exporter.add_text([0, 0, 1.2], "Normals", size=0.5)
    
    
    # 5. Create a simple house shape using triangles
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
    
    # Add house with specific color
    exporter.add_triangles(house_vertices, house_faces, color=(0.8, 0.4, 0.2))
    exporter.add_text([2.5, 1.7, 0], "House", size=0.5, color=(0.8, 0.4, 0.2))
    
    
    # 6. Add some floating text at different sizes
    texts = [
        {"pos": [-1, -1, 0], "text": "Small", "size": 0.3},
        {"pos": [-1, -1, 0.5], "text": "Medium", "size": 0.5},
        {"pos": [-1, -1, 1], "text": "Large", "size": 0.7}
    ]
    
    for text_info in texts:
        exporter.add_text(
            text_info["pos"],
            text_info["text"],
            size=text_info["size"],
            color=(0, 1, 1)  # Cyan color for all text
        )
    
    # Create some spherical points
    sphere_centers = [
        [1.5, 0, 0],
        [2.0, 0, 0],
        [2.5, 0, 0]
    ]
    exporter.add_spheres(sphere_centers, radius=0.1, color=(1, 0.5, 0))  # Orange spheres
    exporter.add_text([2.0, 0.3, 0], "Spheres", size=0.3)
    
    # Create a curved line strip using cylinders
    t = np.linspace(0, 2*np.pi, 20)
    curve_points = np.column_stack([
        2 + 0.5*np.cos(t),
        0.5*np.sin(t),
        np.zeros_like(t)
    ])
    exporter.add_cylinder_strips(curve_points, radius=0.03, color=(0, 0.8, 0.8))  # Cyan tube
    exporter.add_text([2, 1, 0], "Cylinder Strip", size=0.3)
    
    # Create some normal arrows
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
    exporter.add_normal_arrows(normal_points, normal_dirs, 
                             shaft_radius=0.02, 
                             head_radius=0.04, 
                             color=(0.8, 0.2, 0.8))  # Purple arrows
    exporter.add_text([3.5, 0.8, 0], "Normal Arrows", size=0.3)
    
    # Save the scene
    exporter.save("demo_scene.gltf")
    print("Demo scene saved as 'demo_scene.gltf'")

if __name__ == "__main__":
    create_demo_scene()
