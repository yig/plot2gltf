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
    exporter.add_text([0, 0, 0.6], "Cube", size=0.2, color=(1, 1, 1))
    
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
    
    # Add labels for axes
    exporter.add_text([1.1, 0, 0], "X", size=0.1, color=(1, 0, 0))
    exporter.add_text([0, 1.1, 0], "Y", size=0.1, color=(0, 1, 0))
    exporter.add_text([0, 0, 1.1], "Z", size=0.1, color=(0, 0, 1))
    
    # 3. Create a point cloud in a spiral pattern
    t = np.linspace(0, 6*np.pi, 50)
    spiral_points = np.column_stack([
        0.3 * np.cos(t) + 1.5,
        0.3 * np.sin(t) + 1.5,
        0.1 * t
    ])
    
    # Add points with a specific color
    exporter.add_points(spiral_points, color=(1, 1, 0))  # Yellow points
    exporter.add_text([1.5, 1.5, 2], "Spiral", size=0.15, color=(1, 1, 0))
    
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
    exporter.add_text([0, 0, 1.2], "Normals", size=0.15)
    
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
    exporter.add_text([2.5, 1.7, 0], "House", size=0.15, color=(0.8, 0.4, 0.2))
    
    # 6. Add some floating text at different sizes
    texts = [
        {"pos": [-1, -1, 0], "text": "Small", "size": 0.1},
        {"pos": [-1, -1, 0.5], "text": "Medium", "size": 0.2},
        {"pos": [-1, -1, 1], "text": "Large", "size": 0.3}
    ]
    
    for text_info in texts:
        exporter.add_text(
            text_info["pos"],
            text_info["text"],
            size=text_info["size"],
            color=(0, 1, 1)  # Cyan color for all text
        )
    
    # Save the scene
    exporter.save("demo_scene.gltf")
    print("Demo scene saved as 'demo_scene.gltf'")

if __name__ == "__main__":
    create_demo_scene()
