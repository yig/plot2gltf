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
    TRIANGLES
)

class GLTFGeometryExporter:
    def __init__(self):
        self.gltf = GLTF2()
        self.gltf.scenes = [Scene(nodes=[0])]
        self.gltf.nodes = [Node(mesh=0)]
        self.gltf.meshes = [Mesh(primitives=[])]
        self.gltf.materials = []
        self.gltf.buffers = []
        self.gltf.bufferViews = []
        self.gltf.accessors = []
        self.gltf.textures = []
        self.gltf.images = []
        self.gltf.samplers = []
        
        self.color_index = 0
        self.buffer_offset = 0
        self.all_data = bytearray()
    
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
    
    def _create_material(self, color=None, texture_index=None):
        """Create a material with the given color or texture"""
        if color is None:
            color = self._get_unique_color()
        
        pbr_metallic_roughness = {
            "baseColorFactor": [*color, 1.0] if texture_index is None else [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.5
        }
        
        if texture_index is not None:
            pbr_metallic_roughness["baseColorTexture"] = {
                "index": texture_index
            }
        
        material = Material(
            pbrMetallicRoughness=pbr_metallic_roughness,
            alphaMode="BLEND" if texture_index is not None else "OPAQUE",
            doubleSided=True  # Make text visible from both sides
        )
        
        self.gltf.materials.append(material)
        return len(self.gltf.materials) - 1

    def _create_text_texture(self, text, font_size=32, color=(1, 1, 1)):
        """Create a texture containing the given text"""
        # Use larger base font size and padding
        base_font_size = 128  # Increased from 32
        padding = base_font_size // 2
        
        # Estimate text size and create image with proper aspect ratio
        try:
            font = PIL.ImageFont.truetype("arial.ttf", base_font_size)
        except:
            font = PIL.ImageFont.load_default()
            
        # Get text size for proper aspect ratio
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create image with proper aspect ratio
        img_width = text_width + padding * 2
        img_height = text_height + padding * 2
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
        
        # Add sampler
        self.gltf.samplers.append({
            "magFilter": 9729,  # LINEAR
            "minFilter": 9729,  # LINEAR
            "wrapS": 33071,    # CLAMP_TO_EDGE
            "wrapT": 33071     # CLAMP_TO_EDGE
        })
        
        # Add texture
        self.gltf.textures.append({
            "sampler": len(self.gltf.samplers) - 1,
            "source": len(self.gltf.images) - 1
        })
        
        return len(self.gltf.textures) - 1

    def add_triangles(self, vertices, faces, color=None):
        """Add triangle mesh to the GLTF file"""
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32).flatten()  # Flatten for indices
        
        # Create buffer views
        vertex_view_idx = self._create_buffer_view(vertices, ARRAY_BUFFER)
        index_view_idx = self._create_buffer_view(faces, ELEMENT_ARRAY_BUFFER)
        
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
            len(faces),
            "SCALAR"
        )
        
        # Create material
        material_idx = self._create_material(color)
        
        # Create primitive
        primitive = Primitive(
            attributes={"POSITION": vertex_accessor_idx},
            indices=index_accessor_idx,
            material=material_idx,
            mode=TRIANGLES
        )
        
        self.gltf.meshes[0].primitives.append(primitive)
        return vertices, faces

    def add_lines(self, vertices, edges, color=None):
        """Add lines to the GLTF file"""
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
        
        material_idx = self._create_material(color)
        
        primitive = Primitive(
            attributes={"POSITION": vertex_accessor_idx},
            indices=edge_accessor_idx,
            material=material_idx,
            mode=LINES
        )
        
        self.gltf.meshes[0].primitives.append(primitive)
        return vertices, edges

    def add_points(self, vertices, color=None):
        """Add points to the GLTF file"""
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
        
        material_idx = self._create_material(color)
        
        primitive = Primitive(
            attributes={"POSITION": vertex_accessor_idx},
            material=material_idx,
            mode=POINTS
        )
        
        self.gltf.meshes[0].primitives.append(primitive)
        return vertices

    def add_normals(self, points, directions, color=None):
        """Add normal vectors to the GLTF file as lines"""
        points = np.array(points, dtype=np.float32)
        directions = np.array(directions, dtype=np.float32)
        
        # Create end points for the normal vectors
        endpoints = points + directions
        vertices = np.vstack((points, endpoints))
        
        # Create edges connecting points to their normal vector endpoints
        edges = np.array([[i, i + len(points)] for i in range(len(points))])
        
        return self.add_lines(vertices, edges, color)

    def add_text(self, position, text, size=1.0, color=None):
        """Add 3D text at the specified position"""
        position = np.array(position, dtype=np.float32)
        
        # Create texture first to get aspect ratio
        texture_index = self._create_text_texture(text, color=color or (1, 1, 1))
        
        # Get image aspect ratio
        img = PIL.Image.open(io.BytesIO(base64.b64decode(
            self.gltf.images[texture_index]["uri"].split(",")[1]
        )))
        aspect_ratio = img.width / img.height
        
        # Create billboard quad vertices with proper aspect ratio
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
        material_idx = self._create_material(color, texture_index)
        
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
    
    def _create_sphere_mesh(self, radius=1.0, segments=16):
        """Create a sphere mesh centered at origin"""
        vertices = []
        indices = []
        
        # Create rings of vertices
        for i in range(segments + 1):
            lat = np.pi * (-0.5 + float(i) / segments)
            for j in range(segments):
                lon = 2 * np.pi * float(j) / segments
                x = np.cos(lat) * np.cos(lon)
                y = np.cos(lat) * np.sin(lon)
                z = np.sin(lat)
                vertices.append([x * radius, y * radius, z * radius])
        
        # Create triangles
        for i in range(segments):
            for j in range(segments):
                first = i * segments + j
                second = first + 1
                third = (i + 1) * segments + j
                fourth = third + 1
                
                # Two triangles per quad
                indices.extend([first, second, third])
                indices.extend([second, fourth, third])
        
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)
    
    def _create_cylinder_mesh(self, radius=1.0, height=1.0, segments=16):
        """Create a cylinder mesh centered at origin, extending along Y axis"""
        vertices = []
        indices = []
        
        # Create circles of vertices for top and bottom
        for y in [-height/2, height/2]:
            for i in range(segments):
                angle = 2 * np.pi * float(i) / segments
                x = np.cos(angle) * radius
                z = np.sin(angle) * radius
                vertices.append([x, y, z])
        
        # Create triangles for the sides
        for i in range(segments):
            i1 = i
            i2 = (i + 1) % segments
            i3 = i + segments
            i4 = ((i + 1) % segments) + segments
            
            indices.extend([i1, i2, i3])
            indices.extend([i2, i4, i3])
        
        # Add end caps
        center_bottom = len(vertices)
        vertices.append([0, -height/2, 0])
        center_top = len(vertices)
        vertices.append([0, height/2, 0])
        
        for i in range(segments):
            next_i = (i + 1) % segments
            # Bottom cap
            indices.extend([center_bottom, i, next_i])
            # Top cap
            indices.extend([center_top, i + segments, (next_i) + segments])
        
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)
    
    def _create_cone_mesh(self, radius=1.0, height=1.0, segments=16):
        """Create a cone mesh centered at origin, pointing along Y axis"""
        vertices = []
        indices = []
        
        # Create circle of vertices for base
        for i in range(segments):
            angle = 2 * np.pi * float(i) / segments
            x = np.cos(angle) * radius
            z = np.sin(angle) * radius
            vertices.append([x, -height/2, z])
        
        # Add tip vertex
        tip_index = len(vertices)
        vertices.append([0, height/2, 0])
        
        # Add center of base
        base_center_index = len(vertices)
        vertices.append([0, -height/2, 0])
        
        # Create triangles for the sides
        for i in range(segments):
            next_i = (i + 1) % segments
            # Side triangle
            indices.extend([i, next_i, tip_index])
            # Base triangle
            indices.extend([base_center_index, next_i, i])
        
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)
    
    def add_spheres(self, centers, radius=0.1, color=None, segments=16):
        """Add spheres at the given center points"""
        sphere_verts, sphere_indices = self._create_sphere_mesh(radius, segments)
        all_vertices = []
        all_indices = []
        
        for center in centers:
            # Translate sphere vertices to center position
            center_verts = sphere_verts + center
            base_index = len(all_vertices)
            
            all_vertices.extend(center_verts)
            all_indices.extend(sphere_indices + base_index)
        
        return self.add_triangles(all_vertices, np.array(all_indices).reshape(-1, 3), color)
    
    def add_cylinder_strips(self, points, radius=0.05, color=None, segments=16):
        """Add connected cylinders between consecutive points"""
        if len(points) < 2:
            return
        
        all_vertices = []
        all_indices = []
        
        for i in range(len(points) - 1):
            start = np.array(points[i])
            end = np.array(points[i + 1])
            
            # Calculate cylinder height and orientation
            direction = end - start
            height = np.linalg.norm(direction)
            if height < 1e-6:  # Skip if points are too close
                continue
            
            # Create transformation matrix to orient cylinder
            up = np.array([0, 1, 0])
            if np.abs(np.dot(direction / height, up)) > 0.999:
                # If direction is nearly parallel to up vector, use X axis for rotation
                rotation_axis = np.array([1, 0, 0])
            else:
                rotation_axis = np.cross(up, direction)
                rotation_axis /= np.linalg.norm(rotation_axis)
            
            angle = np.arccos(np.dot(direction / height, up))
            c = np.cos(angle)
            s = np.sin(angle)
            t = 1 - c
            
            # Rotation matrix around arbitrary axis
            R = np.array([
                [t*rotation_axis[0]**2 + c, t*rotation_axis[0]*rotation_axis[1] - s*rotation_axis[2], t*rotation_axis[0]*rotation_axis[2] + s*rotation_axis[1]],
                [t*rotation_axis[0]*rotation_axis[1] + s*rotation_axis[2], t*rotation_axis[1]**2 + c, t*rotation_axis[1]*rotation_axis[2] - s*rotation_axis[0]],
                [t*rotation_axis[0]*rotation_axis[2] - s*rotation_axis[1], t*rotation_axis[1]*rotation_axis[2] + s*rotation_axis[0], t*rotation_axis[2]**2 + c]
            ])
            
            # Create and transform cylinder
            cyl_verts, cyl_indices = self._create_cylinder_mesh(radius, height, segments)
            transformed_verts = (cyl_verts @ R.T) + (start + direction/2)
            
            # Add to collection
            base_index = len(all_vertices)
            all_vertices.extend(transformed_verts)
            all_indices.extend(cyl_indices + base_index)
        
        return self.add_triangles(all_vertices, np.array(all_indices).reshape(-1, 3), color)
    
    def add_normal_arrows(self, points, directions, shaft_radius=0.02, head_radius=0.04, 
                         head_length_ratio=0.25, color=None, segments=16):
        """Add normal vectors as cylinders with cone tips"""
        all_vertices = []
        all_indices = []
        
        for point, direction in zip(points, directions):
            point = np.array(point)
            direction = np.array(direction)
            
            # Normalize direction and calculate lengths
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
                
            direction = direction / length
            head_length = length * head_length_ratio
            shaft_length = length - head_length
            
            # Create and orient shaft (cylinder)
            up = np.array([0, 1, 0])
            if np.abs(np.dot(direction, up)) > 0.999:
                rotation_axis = np.array([1, 0, 0])
            else:
                rotation_axis = np.cross(up, direction)
                rotation_axis /= np.linalg.norm(rotation_axis)
            
            angle = np.arccos(np.dot(direction, up))
            c = np.cos(angle)
            s = np.sin(angle)
            t = 1 - c
            
            R = np.array([
                [t*rotation_axis[0]**2 + c, t*rotation_axis[0]*rotation_axis[1] - s*rotation_axis[2], t*rotation_axis[0]*rotation_axis[2] + s*rotation_axis[1]],
                [t*rotation_axis[0]*rotation_axis[1] + s*rotation_axis[2], t*rotation_axis[1]**2 + c, t*rotation_axis[1]*rotation_axis[2] - s*rotation_axis[0]],
                [t*rotation_axis[0]*rotation_axis[2] - s*rotation_axis[1], t*rotation_axis[1]*rotation_axis[2] + s*rotation_axis[0], t*rotation_axis[2]**2 + c]
            ])
            
            # Create shaft
            shaft_verts, shaft_indices = self._create_cylinder_mesh(shaft_radius, shaft_length, segments)
            transformed_shaft = (shaft_verts @ R.T) + (point + direction * shaft_length/2)
            
            # Create head (cone)
            head_verts, head_indices = self._create_cone_mesh(head_radius, head_length, segments)
            transformed_head = (head_verts @ R.T) + (point + direction * (shaft_length + head_length/2))
            
            # Add to collection
            base_index = len(all_vertices)
            all_vertices.extend(transformed_shaft)
            all_indices.extend(shaft_indices + base_index)
            
            base_index = len(all_vertices)
            all_vertices.extend(transformed_head)
            all_indices.extend(head_indices + base_index)
        
        return self.add_triangles(all_vertices, np.array(all_indices).reshape(-1, 3), color)

    def save(self, filename):
        """Save the GLTF file"""
        # Create the final buffer
        self.gltf.buffers = [Buffer(
            byteLength=len(self.all_data),
            uri=f"data:application/octet-stream;base64,{base64.b64encode(self.all_data).decode('ascii')}"
        )]
        
        # Save to file
        self.gltf.save(filename)
