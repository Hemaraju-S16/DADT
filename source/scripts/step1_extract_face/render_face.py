import subprocess
import os

def execute_blender_render(input_path, output_path):
    # 1. Get the Absolute Path of this script's directory
    # If script is in: /source/scripts/step1_extract_face/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up to the Project Root (Deterministic_Analytical_Detail_Transfer)
    # We go up 3 levels from scripts/step1_extract_face/ to reach the root
    project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
    
    # 3. Define the Files
    blender_exe = "blender" 
    blend_file = os.path.join(project_root, "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/blender_files/face_renderer.blend")
    script_file = os.path.join(project_root, "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/scripts/step1_extract_face/face_renderer.py")

    # Debug Prints: Check these in your terminal!
    print(f"--- PATH CHECK ---")
    print(f"Project Root: {project_root}")
    print(f"Blend File:   {blend_file} ({'FOUND' if os.path.exists(blend_file) else 'NOT FOUND'})")
    print(f"Python Script: {script_file} ({'FOUND' if os.path.exists(script_file) else 'NOT FOUND'})")
    print(f"------------------")

    if not os.path.exists(blend_file):
        return

    # 4. Command Construction
    # Using absolute paths for everything to avoid Blender getting lost
    command = [
        blender_exe,
        "-b", blend_file,
        "-P", script_file,
        "--", 
        os.path.abspath(input_path), 
        os.path.abspath(output_path)
    ]

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Blender Output:\n", process.stdout)
    except subprocess.CalledProcessError as e:
        print("!!! BLENDER CRASHED !!!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

if __name__ == "__main__":
    # Use paths relative to where you run the script, or absolute paths
    img_in = "intermediate_inputs/step_1_inputs/input_768_gray_bg.jpg"
    img_out = "intermediate_outputs/step_1_outputs/face_cam_render.png"

    execute_blender_render(img_in, img_out)