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
            alphaMode="BLEND" if texture_index is not None else "OPAQUE"
        )
        
        self.gltf.materials.append(material)
        return len(self.gltf.materials) - 1

    def _create_text_texture(self, text, font_size=32, color=(1, 1, 1)):
        """Create a texture containing the given text"""
        # Create image with alpha channel
        padding = font_size // 2
        img_size = (font_size * len(text) + padding * 2, font_size + padding * 2)
        image = PIL.Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(image)
        
        # Load font (using default font)
        try:
            font = PIL.ImageFont.truetype("arial.ttf", font_size)
        except:
            font = PIL.ImageFont.load_default()
        
        # Draw text
        rgb_color = tuple(int(c * 255) for c in color)
        draw.text((padding, padding), text, font=font, fill=(*rgb_color, 255))
        
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
        
        # Create billboard quad vertices
        half_size = size / 2
        vertices = np.array([
            [position[0] - half_size, position[1] - half_size, position[2]],
            [position[0] + half_size, position[1] - half_size, position[2]],
            [position[0] + half_size, position[1] + half_size, position[2]],
            [position[0] - half_size, position[1] + half_size, position[2]]
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
        
        # Create texture with text
        texture_index = self._create_text_texture(text, font_size=32, color=color or (1, 1, 1))
        
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

    def save(self, filename):
        """Save the GLTF file"""
        # Create the final buffer
        self.gltf.buffers = [Buffer(
            byteLength=len(self.all_data),
            uri=f"data:application/octet-stream;base64,{base64.b64encode(self.all_data).decode('ascii')}"
        )]
        
        # Save to file
        self.gltf.save(filename)
