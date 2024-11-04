## About

A [Claude](https://claude.ai)-created Python module to save different types of geometry to `glTF`. You can view the result in any GLTF viewer, such as:

- Online viewers like <https://gltf-viewer.donmccurdy.com/>
- Three.js-based web viewers
- The Blender 3D editor
- Microsoft's Windows 3D Viewer

Key features:

1. Supports multiple geometry types: triangles, lines, points, and normals
2. Automatic color generation using the golden ratio for visually distinct colors
3. Optional manual color specification
4. All geometries are combined into a single GLTF file
5. Proper material setup with metallic-roughness PBR workflow
6. Text anchored at a point in space.

Here's how to use it:

```python
# Example usage:
from plot2gltf import GLTFGeometryExporter

exporter = GLTFGeometryExporter()

# Add a triangle mesh
vertices = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0]
]
faces = [[0, 1, 2]]
exporter.add_triangles(vertices, faces, color=(1, 0, 0))  # Red triangles

# Add some lines
line_vertices = [
    [0, 0, 0],
    [1, 1, 1]
]
edges = [[0, 1]]
exporter.add_lines(line_vertices, edges)  # Auto-generated color

# Add points
points = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0]
]
exporter.add_points(points, color=(0, 1, 0))  # Green points

# Add normals
normal_points = [[0, 0, 0]]
normal_directions = [[0, 0, 1]]
exporter.add_normals(normal_points, normal_directions, color=(0, 0, 1))  # Blue normals

# Save the file
exporter.save("output.gltf")
```


## Demo

There is a weird, Claude-generated comprehensive demo. Run it via:

```
python demo.py
```

The output file is named `demo_scene.gltf`.

This demo file showcases:

1. Geometric Primitives:
   - Triangles (cube and house)
   - Lines (coordinate axes)
   - Points (spiral pattern)
   - Normals (orientation vectors)

2. Text Features:
   - Different sizes
   - Different colors
   - Labels for geometric objects
   - Stand-alone text examples

3. Color Usage:
   - Manual color specification
   - Automatic color generation
   - Different colors for different object types

4. Complex Shapes:
   - A cube with multiple faces
   - A spiral point cloud
   - A simple house shape
   - Coordinate axes

The scene includes a variety of objects arranged in a way that makes it easy to see all the features.
