
from PIL import Image

def extract_frames_from_gif(gif_path, output_folder):
    # Open the GIF image
    gif = Image.open(gif_path)

    # Iterate over each frame in the GIF
    for frame_index in range(gif.n_frames):
        # Seek to the current frame
        gif.seek(frame_index)

        # Extract the current frame as a PIL Image object
        frame = gif.copy()

        # Generate a unique filename for the frame
        frame_filename = f"frame_{frame_index}.png"

        # Save the frame as PNG image to the output folder
        frame.save(f"{output_folder}/{frame_filename}", "PNG")

    # Close the GIF image
    gif.close()

# Example usage
gif_path = "C:/Users/lsion/Desktop/SC_New_1_Robot.gif"  # Replace with the path to your GIF file
output_folder = "C:/Users/lsion/Desktop/output"   # Replace with the desired output folder

extract_frames_from_gif(gif_path, output_folder)