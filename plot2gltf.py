import numpy as np
from pygltflib import GLTF2, Buffer, BufferView, Accessor, Mesh, Primitive, Node, Scene, Material
from pygltflib import BufferTarget, ComponentType, AccessorType
import colorsys
import base64

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
        
        self.color_index = 0
        self.buffer_offset = 0
    
    def _get_unique_color(self):
        """Generate a unique color using HSV color space"""
        hue = (self.color_index * 0.618033988749895) % 1.0  # golden ratio conjugate
        self.color_index += 1
        return colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    
    def _create_buffer_view(self, data, target):
        """Create a buffer view for the given data"""
        byte_length = data.nbytes
        buffer_view = BufferView(
            buffer=0,
            byteOffset=self.buffer_offset,
            byteLength=byte_length,
            target=target
        )
        self.buffer_offset += byte_length
        return buffer_view
    
    def _create_accessor(self, buffer_view_index, component_type, count, accessor_type):
        """Create an accessor for the given buffer view"""
        return Accessor(
            bufferView=buffer_view_index,
            componentType=component_type,
            count=count,
            type=accessor_type
        )
    
    def _create_material(self, color=None):
        """Create a material with the given color or generate a unique one"""
        if color is None:
            color = self._get_unique_color()
        
        material = Material(
            pbrMetallicRoughness={
                "baseColorFactor": [*color, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.5
            }
        )
        self.gltf.materials.append(material)
        return len(self.gltf.materials) - 1

    def add_triangles(self, vertices, faces, color=None):
        """Add triangle mesh to the GLTF file"""
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32)
        
        # Create buffer views
        vertex_buffer_view = self._create_buffer_view(vertices, BufferTarget.ARRAY_BUFFER)
        index_buffer_view = self._create_buffer_view(faces, BufferTarget.ELEMENT_ARRAY_BUFFER)
        
        # Create accessors
        vertex_accessor = self._create_accessor(
            len(self.gltf.bufferViews),
            ComponentType.FLOAT,
            len(vertices),
            AccessorType.VEC3
        )
        index_accessor = self._create_accessor(
            len(self.gltf.bufferViews) + 1,
            ComponentType.UNSIGNED_INT,
            len(faces) * 3,
            AccessorType.SCALAR
        )
        
        # Create material
        material_index = self._create_material(color)
        
        # Create primitive
        primitive = Primitive(
            attributes={"POSITION": len(self.gltf.accessors)},
            indices=len(self.gltf.accessors) + 1,
            material=material_index,
            mode=4  # TRIANGLES
        )
        
        # Update GLTF
        self.gltf.bufferViews.extend([vertex_buffer_view, index_buffer_view])
        self.gltf.accessors.extend([vertex_accessor, index_accessor])
        self.gltf.meshes[0].primitives.append(primitive)
        
        return vertices, faces
    
    def add_lines(self, vertices, edges, color=None):
        """Add lines to the GLTF file"""
        vertices = np.array(vertices, dtype=np.float32)
        edges = np.array(edges, dtype=np.uint32)
        
        vertex_buffer_view = self._create_buffer_view(vertices, BufferTarget.ARRAY_BUFFER)
        index_buffer_view = self._create_buffer_view(edges, BufferTarget.ELEMENT_ARRAY_BUFFER)
        
        vertex_accessor = self._create_accessor(
            len(self.gltf.bufferViews),
            ComponentType.FLOAT,
            len(vertices),
            AccessorType.VEC3
        )
        index_accessor = self._create_accessor(
            len(self.gltf.bufferViews) + 1,
            ComponentType.UNSIGNED_INT,
            len(edges) * 2,
            AccessorType.SCALAR
        )
        
        material_index = self._create_material(color)
        
        primitive = Primitive(
            attributes={"POSITION": len(self.gltf.accessors)},
            indices=len(self.gltf.accessors) + 1,
            material=material_index,
            mode=1  # LINES
        )
        
        self.gltf.bufferViews.extend([vertex_buffer_view, index_buffer_view])
        self.gltf.accessors.extend([vertex_accessor, index_accessor])
        self.gltf.meshes[0].primitives.append(primitive)
        
        return vertices, edges
    
    def add_points(self, vertices, color=None):
        """Add points to the GLTF file"""
        vertices = np.array(vertices, dtype=np.float32)
        
        vertex_buffer_view = self._create_buffer_view(vertices, BufferTarget.ARRAY_BUFFER)
        
        vertex_accessor = self._create_accessor(
            len(self.gltf.bufferViews),
            ComponentType.FLOAT,
            len(vertices),
            AccessorType.VEC3
        )
        
        material_index = self._create_material(color)
        
        primitive = Primitive(
            attributes={"POSITION": len(self.gltf.accessors)},
            material=material_index,
            mode=0  # POINTS
        )
        
        self.gltf.bufferViews.append(vertex_buffer_view)
        self.gltf.accessors.append(vertex_accessor)
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
    
    def save(self, filename):
        """Save the GLTF file"""
        # Combine all data into a single buffer
        all_data = b''
        for buffer_view in self.gltf.bufferViews:
            accessor = self.gltf.accessors[len(all_data)]
            if accessor.type == AccessorType.VEC3:
                data = np.array(accessor.data, dtype=np.float32)
            else:
                data = np.array(accessor.data, dtype=np.uint32)
            all_data += data.tobytes()
        
        # Create the buffer
        self.gltf.buffers = [Buffer(
            byteLength=len(all_data),
            uri="data:application/octet-stream;base64," + base64.b64encode(all_data).decode('ascii')
        )]
        
        # Save to file
        self.gltf.save(filename)
