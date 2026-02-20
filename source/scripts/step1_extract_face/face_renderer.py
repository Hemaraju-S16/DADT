import bpy
import os
import math
import sys

def setup_and_render(image_path, output_path, ortho_scale=1.12):
    scene = bpy.context.scene
    camera_name = "face_cam"
    
    # 1. Camera Check
    if camera_name not in bpy.data.objects:
        print(f"Error: Camera '{camera_name}' not found!")
        return
    
    cam_obj = bpy.data.objects[camera_name]
    scene.camera = cam_obj

    # 2. Create the Plane
    bpy.ops.mesh.primitive_plane_add(size=ortho_scale, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane_mesh = plane.data
    
    # Rotate 90 degrees on X to face the camera
    plane.rotation_euler = (math.radians(90), 0, 0)

    # 3. Apply Shadeless Emission Material
    # Standard BSDF needs lights; Emission works in background/darkness
    mat = bpy.data.materials.new(name="Temp_Image_Mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear() # Clear default Principled BSDF
    
    node_tex = nodes.new('ShaderNodeTexImage')
    node_emit = nodes.new('ShaderNodeEmission')
    node_out = nodes.new('ShaderNodeOutputMaterial')
    
    if os.path.isfile(image_path):
        node_tex.image = bpy.data.images.load(image_path)
    else:
        print(f"Image not found: {image_path}")
        return
    
    # Link: Image -> Emission -> Output
    mat.node_tree.links.new(node_tex.outputs['Color'], node_emit.inputs['Color'])
    mat.node_tree.links.new(node_emit.outputs['Emission'], node_out.inputs['Surface'])
    plane.data.materials.append(mat)

    # 4. Configure Render Settings (EEVEE)
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.filepath = output_path
    
    # Ensure background doesn't interfere
    scene.render.film_transparent = True 

    # 5. Render (Standard render works in background mode)
    bpy.ops.render.render(write_still=True)
    
    # 6. Cleanup
    bpy.data.objects.remove(plane, do_unlink=True)
    bpy.data.meshes.remove(plane_mesh, do_unlink=True)
    bpy.data.materials.remove(mat, do_unlink=True)
    
    print(f"SUCCESS: Render saved to {output_path}")
    
    


# --- ARGUMENT PARSING ---
if __name__ == "__main__":
    # Standard Blender arg parsing: ignore everything before "--"
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        if len(args) >= 2:
            img_in = args[0]
            img_out = args[1]
            setup_and_render(img_in, img_out)
        else:
            print("Usage: blender file.blend --python script.py -- <input_path> <output_path>")