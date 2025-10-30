# common/generate_animation.py
import argparse
import re
from pathlib import Path

import imageio


def natural_key(file: Path):
    """
    This function is used to sort filenames in a natural order.

    @param file: Path to the file.
    @return: list of integers and strings extracted from the filename.

    """
    # Extract numeric parts from the filename
    return [
        int(text) if text.isdigit() else text for text in re.split(r"(\d+)", file.stem)
    ]


def generate_animation(
    duration: float = 10,
    folder_name: str = "figures",
    save_folder: str = "animations",
    delete_figures: bool = False,
    loop: int = 0,
):
    """
    Generate an animation from images in a folder.
    This function creates a GIF animation from images in a specified folder.


    @param duration: float, duration in seconds
    @param folder_name: path to the image folder.
    @param save_folder: path to save the animation.
    @param delete_figures: boolean. Deletes source images after saving.
    @param loop: Number of loops. Defaults to 0 (infinite).
    @return:
    """
    folder_path = Path(folder_name)
    save_path = Path(save_folder) / "animation1.gif"

    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    filenames = sorted(folder_path.iterdir(), key=natural_key)

    images = []
    for file in filenames:
        if file.is_file():
            image = imageio.v2.imread(file)
            images.append(image)

    imageio.mimsave(save_path, images, duration=duration, loop=loop)

    if delete_figures:
        for file in filenames:
            try:
                if file.is_file():
                    file.unlink()
            except Exception as e:
                print(f"Error deleting {file}: {e}")


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate an animation from figures.")
    parser.add_argument(
        "--duration", type=int, default=5, help="Duration of the animation in seconds"
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        default="figures",
        help="Folder containing figure files",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="animations",
        help="Folder to save the animation",
    )
    parser.add_argument(
        "--delete_figures",
        action="store_true",
        help="Delete figure files after creating the animation",
    )
    parser.add_argument(
        "--loop", type=int, default=0, help="Number of loops (0 for infinite)"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    generate_animation(
        duration=args.duration,
        folder_name=args.folder_name,
        save_folder=args.save_folder,
        delete_figures=args.delete_figures,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
