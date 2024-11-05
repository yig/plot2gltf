"""A module for creating GLTF files with various geometric primitives and 3D text labels."""

__version__ = "1.0.2"

import numpy as np
from pygltflib import GLTF2, Buffer, BufferView, Accessor, Mesh, Primitive, Node, Scene, Material
import colorsys
import base64
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import io

# Import constants from pygltflib
from pygltflib.validator import (
    ARRAY_BUFFER, 
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    UNSIGNED_INT,
    POINTS,
    LINES,
    TRIANGLES,
    LINEAR,
    LINEAR_MIPMAP_LINEAR,
    CLAMP_TO_EDGE
)

class GLTFGeometryExporter:
    """
    A class for creating GLTF files with various geometric primitives and 3D text labels.
    
    Supports:
    - Triangle meshes
    - Line segments and line strips (as hairlines or cylinder tubes)
    - Points (as dots or spheres)
    - Normal vectors (as lines or arrows)
    - Text labels
    
    3D geometry can be colored with either lit (shaded) or unlit (constant color) materials.
    """
    
    def __init__(self):
        """Initialize a new GLTF geometry exporter."""
        self.gltf = GLTF2()
        self.gltf.scenes = [Scene(nodes=[])]
        self.gltf.nodes = []
        self.gltf.meshes = []
        self.gltf.materials = []
        self.gltf.buffers = []
        self.gltf.bufferViews = []
        self.gltf.accessors = []
        self.gltf.textures = []
        self.gltf.images = []
        self.gltf.samplers = []
        
        self.color_index = 0
        self.all_data = bytearray()
        
        # Cache for unit-sized primitive meshes
        self._mesh_cache = {
            'sphere': {},    # segments -> mesh_index
            'cylinder': {},  # segments -> mesh_index
            'cone': {}      # segments -> mesh_index
        }
    
    def _get_unique_color(self):
        """Generate a unique color using HSV color space"""
        hue = (self.color_index * 0.618033988749895) % 1.0  # golden ratio conjugate
        self.color_index += 1
        return colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    
    def _add_to_buffer(self, data):
        """Add data to the buffer and return offset"""
        offset = len(self.all_data)
        self.all_data.extend(data.tobytes())
        return offset
    
    def _create_buffer_view(self, data, target):
        """Create a buffer view for the given data"""
        offset = self._add_to_buffer(data)
        buffer_view = BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=data.nbytes,
            target=target
        )
        self.gltf.bufferViews.append(buffer_view)
        return len(self.gltf.bufferViews) - 1
    
    def _create_accessor(self, buffer_view_index, component_type, count, accessor_type, min_vals=None, max_vals=None):
        """Create an accessor for the given buffer view"""
        accessor = Accessor(
            bufferView=buffer_view_index,
            componentType=component_type,
            count=count,
            type=accessor_type,
            min=min_vals,
            max=max_vals
        )
        self.gltf.accessors.append(accessor)
        return len(self.gltf.accessors) - 1
    
    def _create_material(self, color=None, texture_index=None, unlit=False):
        """
        Create a material with the given color or texture
        
        Parameters:
            color: (r,g,b) tuple or None for auto-color
            texture_index: index of texture or None for solid color
            unlit: if True, creates an unlit material that ignores lighting
        """
        if color is None:
            color = self._get_unique_color()
        
        material = {
            "pbrMetallicRoughness": {
                "baseColorFactor": [*color, 1.0] if texture_index is None else [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.5
            },
            "alphaMode": "BLEND" if texture_index is not None else "OPAQUE",
            "doubleSided": True
        }
        
        if texture_index is not None:
            material["pbrMetallicRoughness"]["baseColorTexture"] = {
                "index": texture_index
            }
        
        if unlit:
            # Add KHR_materials_unlit extension
            material["extensions"] = {
                "KHR_materials_unlit": {}
            }
            if "extensions" not in self.gltf.extensions:
                self.gltf.extensions = []
            if "KHR_materials_unlit" not in self.gltf.extensions:
                self.gltf.extensions.append("KHR_materials_unlit")
            if not hasattr(self.gltf, 'extensionsUsed'):
                self.gltf.extensionsUsed = []
            if "KHR_materials_unlit" not in self.gltf.extensionsUsed:
                self.gltf.extensionsUsed.append("KHR_materials_unlit")
        
        # Create the material
        mat = Material(**material)
        self.gltf.materials.append(mat)
        return len(self.gltf.materials) - 1
    
    def _create_node(self, mesh_index, translation=None, rotation=None, scale=None):
        """Create a node with transformation and mesh reference"""
        node = Node(mesh=mesh_index)
        
        if translation is not None:
            node.translation = list(translation)
        if rotation is not None:
            node.rotation = list(rotation)
        if scale is not None:
            node.scale = list(scale)
        
        self.gltf.nodes.append(node)
        node_index = len(self.gltf.nodes) - 1
        self.gltf.scenes[0].nodes.append(node_index)
        return node_index
    
    def _create_primitive_geometry(self, material_index, vertices, indices, normals=None):
        """
        Create buffer views, accessors, and a mesh primitive from geometry data.
        """
        # Create buffer views
        vertex_view_idx = self._create_buffer_view(vertices, ARRAY_BUFFER)
        if normals is not None: normal_view_idx = self._create_buffer_view(normals, ARRAY_BUFFER)
        index_view_idx = self._create_buffer_view(indices, ELEMENT_ARRAY_BUFFER)
        
        # Create accessors
        vertex_accessor_idx = self._create_accessor(
            vertex_view_idx, 
            FLOAT, 
            len(vertices), 
            "VEC3",
            vertices.min(axis=0).tolist(),
            vertices.max(axis=0).tolist()
        )
        
        index_accessor_idx = self._create_accessor(
            index_view_idx, 
            UNSIGNED_INT, 
            len(indices), 
            "SCALAR"
        )
        
        if normals is not None:
            normal_accessor_idx = self._create_accessor(
                normal_view_idx, 
                FLOAT, 
                len(normals), 
                "VEC3",
                [-1, -1, -1],
                [1, 1, 1]
            )
        
        # Create primitive
        primitive = Primitive(
            attributes={
                "POSITION": vertex_accessor_idx
            },
            indices=index_accessor_idx,
            material=material_index,
            mode=TRIANGLES
        )
        if normals is not None: primitive.attributes["NORMAL"] = normal_accessor_idx
        
        # Create mesh with the primitive
        mesh = Mesh(primitives=[primitive])
        self.gltf.meshes.append(mesh)
        return len(self.gltf.meshes) - 1
    
    def _clone_mesh_with_material(self, template_mesh_index, material_index ):
        """
        Clone a mesh with new material but reuse its geometry accessors.
        """
        template_mesh = self.gltf.meshes[template_mesh_index]
        template_primitive = template_mesh.primitives[0]
        
        # Create new primitive reusing geometry accessors
        new_primitive = Primitive(
            attributes=template_primitive.attributes.copy(),
            indices=template_primitive.indices,
            material=material_index,
            mode=template_primitive.mode
        )
        
        # Create new mesh with the primitive
        new_mesh = Mesh(primitives=[new_primitive])
        self.gltf.meshes.append(new_mesh)
        return len(self.gltf.meshes) - 1
    
    def _create_sphere_mesh(self, segments=16):
        """Create a unit sphere mesh centered at origin"""
        vertices = []
        normals = []
        indices = []
        
        # Create vertices and normals
        for phi in np.linspace(0, np.pi, segments):
            for theta in np.linspace(0, 2*np.pi, segments):
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                vertices.append([x, y, z])
                normals.append([x, y, z])  # For sphere, normal is same as position
        
        # Create triangles
        for i in range(segments - 1):
            for j in range(segments - 1):
                current = i * segments + j
                next_h = current + 1
                next_v = (i + 1) * segments + j
                next_vh = next_v + 1
                indices.extend([current, next_h, next_v])
                indices.extend([next_h, next_vh, next_v])
        
        # Close the gaps
        for i in range(segments - 1):
            current = i * segments + (segments - 1)
            next_v = (i + 1) * segments + (segments - 1)
            first_in_row = i * segments
            first_next_row = (i + 1) * segments
            indices.extend([current, first_in_row, next_v])
            indices.extend([first_in_row, first_next_row, next_v])
        
        return (np.array(vertices, dtype=np.float32),
                np.array(normals, dtype=np.float32),
                np.array(indices, dtype=np.uint32))
    
    def _create_cylinder_mesh(self, segments=16):
        """Create a unit cylinder mesh (radius=1.0, height=1.0)"""
        vertices = []
        normals = []
        indices = []
        
        # Create circles of vertices for top and bottom
        for y in [-0.5, 0.5]:
            for i in range(segments):
                angle = 2 * np.pi * i / segments
                x = np.cos(angle)
                z = np.sin(angle)
                vertices.append([x, y, z])
                normals.append([x, 0, z])  # Side normal
        
        # Create side triangles
        for i in range(segments):
            i1 = i
            i2 = (i + 1) % segments
            i3 = i + segments
            i4 = ((i + 1) % segments) + segments
            indices.extend([i1, i2, i3])
            indices.extend([i2, i4, i3])
        
        # Add end caps
        center_bottom = len(vertices)
        vertices.append([0, -0.5, 0])
        normals.append([0, -1, 0])
        
        center_top = len(vertices)
        vertices.append([0, 0.5, 0])
        normals.append([0, 1, 0])
        
        # Add cap vertices
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = np.cos(angle)
            z = np.sin(angle)
            
            # Bottom cap
            vertices.append([x, -0.5, z])
            normals.append([0, -1, 0])
            
            # Top cap
            vertices.append([x, 0.5, z])
            normals.append([0, 1, 0])
        
        # Add cap triangles
        bottom_start = center_top + 1
        top_start = bottom_start + segments
        
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.extend([
                center_bottom,
                bottom_start + next_i,
                bottom_start + i
            ])
            indices.extend([
                center_top,
                top_start + i,
                top_start + next_i
            ])
        
        return (np.array(vertices, dtype=np.float32),
                np.array(normals, dtype=np.float32),
                np.array(indices, dtype=np.uint32))
    
    def _create_cone_mesh(self, segments=16):
        """Create a unit cone mesh (radius=1.0, height=1.0)"""
        vertices = []
        normals = []
        indices = []
        
        # Create base vertices
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = np.cos(angle)
            z = np.sin(angle)
            vertices.append([x, -0.5, z])
            
            # Calculate normal for side
            normal = np.array([x, 0.5, z])
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)
        
        # Add tip
        tip_index = len(vertices)
        vertices.append([0, 0.5, 0])
        normals.append([0, 1, 0])
        
        # Add base center
        base_center = len(vertices)
        vertices.append([0, -0.5, 0])
        normals.append([0, -1, 0])
        
        # Create side triangles
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.extend([i, next_i, tip_index])
        
        # Create base triangles
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.extend([base_center, next_i, i])
        
        return (np.array(vertices, dtype=np.float32),
                np.array(normals, dtype=np.float32),
                np.array(indices, dtype=np.uint32))
    
    def _create_text_texture(self, text, font_size=256, color=(1, 1, 1), font_path=None):
        """Create a texture containing the given text with higher resolution"""
        # Increased base font size significantly
        base_font_size = font_size  # Was 128, now 256
        padding = font_size // 4  # Reduced relative padding
        
        # Create font (with error handling for missing fonts)
        font = None
        if font_path is not None:
            try:
                font = PIL.ImageFont.truetype(font_path, base_font_size)
            except OSError:
                print( "Warning: Couldn't open font path:", font_path )
        if font is None:
            try:
                from pathlib import Path
                font = PIL.ImageFont.truetype( Path( __file__ ).parent / "fonts" / "DejaVuSerif.ttf", base_font_size)
            except OSError:
                font = PIL.ImageFont.load_default()
                print("Warning: Using default font, text may appear pixelated")
        
        # Get text size for proper aspect ratio
        # Use getbbox for more accurate text bounds
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create image with proper aspect ratio and power-of-two dimensions
        # Add extra padding to prevent text from touching edges
        img_width = text_width + padding * 2
        img_height = text_height + padding * 2
        
        # Round up to nearest power of 2 for better texture performance
        def next_power_of_2(x):
            return 1 if x == 0 else 2**(x - 1).bit_length()
        
        #img_width = next_power_of_2(img_width)
        #img_height = next_power_of_2(img_height)
        
        # Create image with alpha channel
        image = PIL.Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(image)
        
        # Draw text centered in image
        text_x = (img_width - text_width) // 2
        text_y = (img_height - text_height) // 2
        rgb_color = tuple(int(c * 255) for c in color)
        draw.text((text_x, text_y), text, font=font, fill=(*rgb_color, 255))
        
        # Convert to PNG bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        png_bytes = buffer.getvalue()
        
        # Add image to GLTF
        self.gltf.images.append({
            "uri": f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"
        })
        
        # Add sampler with linear filtering for better quality
        self.gltf.samplers.append({
            "magFilter": LINEAR,
            "minFilter": LINEAR_MIPMAP_LINEAR,
            "wrapS": CLAMP_TO_EDGE,
            "wrapT": CLAMP_TO_EDGE
        })
        
        # Add texture
        self.gltf.textures.append({
            "sampler": len(self.gltf.samplers) - 1,
            "source": len(self.gltf.images) - 1
        })
        
        return len(self.gltf.textures) - 1

    def add_triangles(self, vertices, faces, normals=None, color=None, unlit=False):
        """
        Add triangle mesh to the GLTF file.
        
        Parameters:
            vertices: list of [x,y,z] coordinates defining the mesh vertices
            faces: list of [i,j,k] indices defining triangles (3 vertices per face)
            normals: optional [x,y,z] normals for each vertex. If unspecified, flat shading is used.
            color: optional (r,g,b) color tuple. If None, a unique color will be generated
            unlit: if True, creates constant-color geometry; if False, uses lit materials (default: False)
        
        Returns:
            tuple: (vertices array, faces array) of the added geometry
        
        Example:
            vertices = [[0,0,0], [1,0,0], [0,1,0]]
            faces = [[0,1,2]]
            exporter.add_triangles(vertices, faces, color=(1,0,0))  # Red triangle
        """
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(faces, dtype=np.uint32).flatten()
        if normals is not None: normals = np.array(normals, dtype=np.float32)
        
        '''
        # Calculate face normals for flat shading
        v0 = vertices[faces[:,0]]
        v1 = vertices[faces[:,1]]
        v2 = vertices[faces[:,2]]
        normals = np.cross(v1 - v0, v2 - v0)
        normals = normals / np.linalg.norm(normals, axis=1)[:,None]
        
        # Duplicate vertices for flat shading
        flat_vertices = np.vstack([v0, v1, v2])
        flat_normals = np.repeat(normals, 3, axis=0)
        flat_indices = np.arange(len(flat_vertices), dtype=np.uint32)
        
        mesh_index = self._create_primitive_geometry(
            flat_vertices, flat_normals, flat_indices, color, unlit)
        '''
        
        mesh_index = self._create_primitive_geometry(
            self._create_material(color, unlit=unlit),
            vertices, indices, normals=normals
            )
        
        # Create single node without transformation
        self._create_node(mesh_index)
        
        return mesh_index


    def add_linestrip(self, points, color=None):
        """
        Add a continuous line strip connecting consecutive points.
        
        Parameters:
            points: list of [x,y,z] coordinates defining the vertices of the line strip
            color: optional (r,g,b) color tuple. If None, a unique color will be generated
        
        Returns:
            tuple: (vertices array, edges array) of the added geometry
        
        Notes:
            Creates a sequence of connected line segments where each point is connected
            to the next point in the sequence.
        
        Example:
            points = [[0,0,0], [1,1,0], [2,1,0], [2,2,0]]
            exporter.add_linestrip(points, color=(1,0,0))  # Red connected line
        """
        if len(points) < 2:
            return None
        
        # Create edges connecting consecutive points
        edges = [[i, i+1] for i in range(len(points)-1)]
        
        return self.add_lines(points, edges, color)
    
    def add_lines(self, vertices, edges, color=None):
        """
        Add line segments to the GLTF file.
        
        Parameters:
            vertices: list of [x,y,z] coordinates defining the line endpoints
            edges: list of [i,j] indices defining lines (2 vertices per line)
            color: optional (r,g,b) color tuple. If None, a unique color will be generated
        
        Returns:
            tuple: (vertices array, edges array) of the added geometry
        
        Example:
            vertices = [[0,0,0], [1,1,1]]
            edges = [[0,1]]
            exporter.add_lines(vertices, edges, color=(0,1,0))  # Green line
        """
        vertices = np.array(vertices, dtype=np.float32)
        edges = np.array(edges, dtype=np.uint32).flatten()
        
        vertex_view_idx = self._create_buffer_view(vertices, ARRAY_BUFFER)
        edge_view_idx = self._create_buffer_view(edges, ELEMENT_ARRAY_BUFFER)
        
        vertex_accessor_idx = self._create_accessor(
            vertex_view_idx,
            FLOAT,
            len(vertices),
            "VEC3",
            vertices.min(axis=0).tolist(),
            vertices.max(axis=0).tolist()
        )
        
        edge_accessor_idx = self._create_accessor(
            edge_view_idx,
            UNSIGNED_INT,
            len(edges),
            "SCALAR"
        )
        
        material_idx = self._create_material(color, unlit=True)
        
        primitive = Primitive(
            attributes={"POSITION": vertex_accessor_idx},
            indices=edge_accessor_idx,
            material=material_idx,
            mode=LINES
        )
        
        self.gltf.meshes[0].primitives.append(primitive)
        return vertices, edges

    def add_points(self, vertices, color=None):
        """
        Add points to the GLTF file as simple dots.
        
        Parameters:
            vertices: list of [x,y,z] coordinates defining point positions
            color: optional (r,g,b) color tuple. If None, a unique color will be generated
        
        Returns:
            array: vertices array of the added points
        
        Example:
            points = [[0,0,0], [1,1,1], [2,2,2]]
            exporter.add_points(points, color=(0,0,1))  # Blue points
        """
        vertices = np.array(vertices, dtype=np.float32)
        
        vertex_view_idx = self._create_buffer_view(vertices, ARRAY_BUFFER)
        
        vertex_accessor_idx = self._create_accessor(
            vertex_view_idx,
            FLOAT,
            len(vertices),
            "VEC3",
            vertices.min(axis=0).tolist(),
            vertices.max(axis=0).tolist()
        )
        
        material_idx = self._create_material(color, unlit=True)
        
        primitive = Primitive(
            attributes={"POSITION": vertex_accessor_idx},
            material=material_idx,
            mode=POINTS
        )
        
        self.gltf.meshes[0].primitives.append(primitive)
        return vertices

    def add_normals(self, points, directions, color=None):
        """
        Add normal vectors as lines.
        
        Parameters:
            points: list of [x,y,z] coordinates for arrow start points
            directions: list of [dx,dy,dz] vectors defining arrow directions
            color: optional (r,g,b) color tuple. If None, a unique color will be generated
        
        Returns:
            tuple: (vertices array, indices array) of the added geometry
        
        Example:
            points = [[0,0,0]]
            directions = [[0,0,1]]  # Points up in Z direction
            exporter.add_normal_arrows(points, directions, color=(1,0,1))  # Purple line
        """
        points = np.array(points, dtype=np.float32)
        directions = np.array(directions, dtype=np.float32)
        
        # Create end points for the normal vectors
        endpoints = points + directions
        vertices = np.vstack((points, endpoints))
        
        # Create edges connecting points to their normal vector endpoints
        edges = np.array([[i, i + len(points)] for i in range(len(points))])
        
        return self.add_lines(vertices, edges, color)

    def add_text(self, position, text, size=1.0, color=None, font_path=None):
        """
        Add 3D text at the specified position.
        
        Parameters:
            position: [x,y,z] coordinate for text position
            text: string to display
            size: height of the text in world units (default: 1.0)
            color: optional (r,g,b) color tuple. If None, a unique color will be generated
            font_path: optional path to a font file that Freetype can open.
        
        Returns:
            tuple: (vertices array, indices array) of the added geometry
        
        Notes:
            - Uses high-resolution textures for clear rendering
        
        Example:
            exporter.add_text([0,2,0], "Hello World", size=0.5, color=(1,1,1))  # White text
        """
        position = np.array(position, dtype=np.float32)
        
        # Create texture first to get aspect ratio
        texture_index = self._create_text_texture(text, color=color or (1, 1, 1), font_path=font_path)
        
        # Get image aspect ratio
        img = PIL.Image.open(io.BytesIO(base64.b64decode(
            self.gltf.images[texture_index]["uri"].split(",")[1]
        )))
        aspect_ratio = img.width / img.height
        
        # Create billboard quad vertices with proper aspect ratio
        # Increased base size significantly
        half_height = size / 2
        half_width = half_height * aspect_ratio
        
        vertices = np.array([
            [position[0] - half_width, position[1] - half_height, position[2]],
            [position[0] + half_width, position[1] - half_height, position[2]],
            [position[0] + half_width, position[1] + half_height, position[2]],
            [position[0] - half_width, position[1] + half_height, position[2]]
        ], dtype=np.float32)
        
        # UV coordinates
        uvs = np.array([
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 0]
        ], dtype=np.float32)
        
        # Indices for two triangles
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        
        # Create buffer views
        vertex_view_idx = self._create_buffer_view(vertices, ARRAY_BUFFER)
        uv_view_idx = self._create_buffer_view(uvs, ARRAY_BUFFER)
        index_view_idx = self._create_buffer_view(indices, ELEMENT_ARRAY_BUFFER)
        
        # Create accessors
        vertex_accessor_idx = self._create_accessor(
            vertex_view_idx,
            FLOAT,
            len(vertices),
            "VEC3",
            vertices.min(axis=0).tolist(),
            vertices.max(axis=0).tolist()
        )
        
        uv_accessor_idx = self._create_accessor(
            uv_view_idx,
            FLOAT,
            len(uvs),
            "VEC2",
            [0, 0],
            [1, 1]
        )
        
        index_accessor_idx = self._create_accessor(
            index_view_idx,
            UNSIGNED_INT,
            len(indices),
            "SCALAR"
        )
        
        # Create material with texture
        material_idx = self._create_material(color, texture_index, unlit=True)
        
        # Create primitive
        primitive = Primitive(
            attributes={
                "POSITION": vertex_accessor_idx,
                "TEXCOORD_0": uv_accessor_idx
            },
            indices=index_accessor_idx,
            material=material_idx,
            mode=TRIANGLES
        )
        
        self.gltf.meshes[0].primitives.append(primitive)
        return vertices, indices
    
    def add_spheres(self, centers, radius=0.1, color=None, segments=16, unlit=True):
        """
        Add spheres to the GLTF file at specified center points.
        
        Parameters:
            centers: list of [x,y,z] coordinates for sphere centers
            radius: radius of the spheres (default: 0.1)
            color: optional (r,g,b) color tuple. If None, a unique color will be generated
            segments: number of segments for sphere mesh (default: 16)
            unlit: if True, creates constant-color spheres; if False, uses lit materials (default: True)
        
        Returns:
            tuple: (vertices array, indices array) of the added geometry
        
        Example:
            centers = [[0,0,0], [1,1,1]]
            exporter.add_spheres(centers, radius=0.2, color=(1,0,0))  # Red spheres
        """
        
        material_index = self._create_material( color, unlit=unlit )
        
        # Check cache or create new sphere mesh
        if segments not in self._mesh_cache['sphere']:
            # Create unit sphere geometry once
            sphere_verts, sphere_normals, sphere_indices = self._create_sphere_mesh(segments)
            mesh_index = self._create_primitive_geometry( material_index, sphere_verts, sphere_indices, normals = sphere_normals )
            self._mesh_cache['sphere'][segments] = mesh_index
        
        # Get cached mesh index
        mesh_index = self._clone_mesh_with_material( self._mesh_cache['sphere'][segments], material_index )
        
        # Create a node for each sphere position with appropriate scale
        scale = [radius, radius, radius]
        for center in centers:
            self._create_node(mesh_index, translation=center, scale=scale)
        
        return mesh_index
    
    def add_cylinder_strips(self, points, radius=0.05, color=None, segments=16, add_spheres=True, unlit=True):
        """
        Add connected cylinders between consecutive points.
        
        Parameters:
            points: list of [x,y,z] coordinates defining the cylinder path
            radius: radius of the cylinders (default: 0.05)
            color: optional (r,g,b) color tuple. If None, a unique color will be generated
            segments: number of segments for cylinder cross-section (default: 16)
            add_spheres: if True, adds spheres at joints for smooth connections (default: True)
            unlit: if True, creates constant-color geometry; if False, uses lit materials (default: True)
        
        Returns:
            tuple: (vertices array, indices array) of the added geometry
        
        Example:
            points = [[0,0,0], [1,1,0], [2,1,0]]
            exporter.add_cylinder_strips(points, radius=0.1, color=(0,1,0))  # Green tube
        """
        if len(points) < 2:
            return None
        
        material_index = self._create_material( color, unlit=unlit )
        
        # Check cache or create new unit cylinder mesh
        if segments not in self._mesh_cache['cylinder']:
            # Create unit cylinder geometry once
            cyl_verts, cyl_normals, cyl_indices = self._create_cylinder_mesh(segments)
            mesh_index = self._create_primitive_geometry( material_index, cyl_verts, cyl_indices, normals = cyl_normals )
            self._mesh_cache['cylinder'][segments] = mesh_index
        
        mesh_index = self._clone_mesh_with_material( self._mesh_cache['cylinder'][segments], material_index )
        
        # Create nodes for each cylinder segment
        for i in range(len(points) - 1):
            start = np.array(points[i])
            end = np.array(points[i + 1])
            
            direction = end - start
            height = np.linalg.norm(direction)
            if height < 1e-6:
                continue
            
            direction_normalized = direction / height
            
            # Calculate rotation quaternion from up vector to direction
            up = np.array([0, 1, 0])
            if np.abs(np.dot(direction_normalized, up)) > 0.999:
                rotation = [0, 0, 0, 1] if direction_normalized[1] > 0 else [1, 0, 0, 0]
            else:
                axis = np.cross(up, direction_normalized)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.dot(up, direction_normalized))
                s = np.sin(angle/2)
                rotation = [axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle/2)]
            
            translation = start + direction/2
            scale = [radius, height, radius]
            self._create_node(mesh_index, translation=translation.tolist(), 
                            rotation=rotation, scale=scale)
        
        if add_spheres:
            self.add_spheres(points, radius=radius, color=color, segments=segments)
        
        return mesh_index
    
    def add_normal_arrows(self, points, directions, shaft_radius=0.02, head_radius=0.04, 
                         head_length_ratio=0.25, color=None, segments=16, unlit=True):
        """
        Add normal vectors as cylinders with cone tips.
        
        Parameters:
            points: list of [x,y,z] coordinates for arrow start points
            directions: list of [dx,dy,dz] vectors defining arrow directions
            shaft_radius: radius of the cylinder shaft (default: 0.02)
            head_radius: radius of the cone tip (default: 0.04)
            head_length_ratio: ratio of cone length to total length (default: 0.25)
            color: optional (r,g,b) color tuple. If None, a unique color will be generated
            segments: number of segments for cylinder/cone cross-sections (default: 16)
            unlit: if True, creates constant-color geometry; if False, uses lit materials (default: True)
        
        Returns:
            tuple: (vertices array, indices array) of the added geometry
        
        Example:
            points = [[0,0,0]]
            directions = [[0,0,1]]  # Points up in Z direction
            exporter.add_normal_arrows(points, directions, color=(1,0,1))  # Purple arrow
        """
        
        material_index = self._create_material( color, unlit=unlit )
        
        if segments not in self._mesh_cache['cylinder']:
            cyl_verts, cyl_normals, cyl_indices = self._create_cylinder_mesh(segments)
            mesh_index = self._create_primitive_geometry( material_index, cyl_verts, cyl_indices, normals=cyl_normals )
            self._mesh_cache['cylinder'][segments] = mesh_index
            
        if segments not in self._mesh_cache['cone']:
            cone_verts, cone_normals, cone_indices = self._create_cone_mesh(segments)
            mesh_index = self._create_primitive_geometry( material_index, cone_verts, cone_indices, normals=cone_normals )
            self._mesh_cache['cone'][segments] = mesh_index
        
        cyl_mesh_index = self._clone_mesh_with_material( self._mesh_cache['cylinder'][segments], material_index )
        cone_mesh_index = self._clone_mesh_with_material( self._mesh_cache['cone'][segments], material_index )
        
        for point, direction in zip(points, directions):
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            
            direction_normalized = direction / length
            shaft_length = length * (1 - head_length_ratio)
            head_length = length * head_length_ratio
            
            # Calculate rotation from up vector
            up = np.array([0, 1, 0])
            if np.abs(np.dot(direction_normalized, up)) > 0.999:
                rotation = [0, 0, 0, 1] if direction_normalized[1] > 0 else [1, 0, 0, 0]
            else:
                axis = np.cross(up, direction_normalized)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.dot(up, direction_normalized))
                s = np.sin(angle/2)
                rotation = [axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle/2)]
            
            # Add shaft
            shaft_translation = point + direction_normalized * (shaft_length/2)
            shaft_scale = [shaft_radius, shaft_length, shaft_radius]
            self._create_node(cyl_mesh_index, 
                            translation=shaft_translation.tolist(),
                            rotation=rotation,
                            scale=shaft_scale)
            
            # Add head
            head_translation = point + direction_normalized * (length - head_length/2)
            head_scale = [head_radius, head_length, head_radius]
            self._create_node(cone_mesh_index,
                            translation=head_translation.tolist(),
                            rotation=rotation,
                            scale=head_scale)
        
        return cyl_mesh_index, cone_mesh_index

    def save(self, filename):
        """
        Save the GLTF file to disk.
        
        Parameters:
            filename: output filename (should end in .gltf)
        
        Example:
            exporter.save("output.gltf")
        """
        # Create the final buffer
        self.gltf.buffers = [Buffer(
            byteLength=len(self.all_data),
            uri=f"data:application/octet-stream;base64,{base64.b64encode(self.all_data).decode('ascii')}"
        )]
        
        # Save to file
        self.gltf.save(filename)
