# common/generate_animation.py
import argparse
import re
from pathlib import Path

import imageio


def natural_key(file: Path) -> list[int | str]:
    """
    Generate a natural sorting key from a filename.

    Splits the filename into alternating numeric and non-numeric parts so that
    files like 'file2', 'file10' sort in intuitive order.

    Parameters
    ----------
    file : Path
        The file whose stem will be processed.

    Returns
    -------
    List[Union[int, str]]
        A list containing integers for numeric parts and strings for text parts.
    """
    # Extract numeric parts from the filename
    parts = re.split(r"(\d+)", file.stem)
    return [int(part) if part.isdigit() else part for part in parts]


def generate_animation(
    duration: float = 10,
    source_folder: str | Path = "figures",
    save_folder: str | Path = "animations",
    delete_figures: bool = False,
    loop: int = 0,
) -> Path:
    """
    Create a GIF animation from all images in a folder.

    Parameters
    ----------
    duration : float
        Duration (in seconds) between frames in the GIF.
    source_folder : str
        Folder containing the source image files.
    save_folder : str
        Folder where the GIF will be written.
    delete_figures : bool
        If True, deletes all images from the folder after saving the GIF.
    loop : int
        Number of times the GIF will loop. ``0`` means infinite loop.

    Returns
    -------
    Path
        Path to the created GIF file.
    """
    source_folder = Path(source_folder)
    save_folder = Path(save_folder)

    save_path = save_folder / "animation1.gif"

    save_folder.mkdir(parents=True, exist_ok=True)

    filenames = sorted(source_folder.iterdir(), key=natural_key)

    # Read images and append to list
    images = []
    for file in filenames:
        if file.is_file():
            try:
                image = imageio.v2.imread(file)
                images.append(image)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if not images:
        raise ValueError(f"No valid images found in {source_folder}")

    # Save animation
    imageio.mimsave(save_path, images, duration=duration, loop=loop)

    # Optionally, delete the original frames
    if delete_figures:
        for file in filenames:
            try:
                if file.is_file():
                    file.unlink()
            except Exception as e:
                print(f"Error deleting {file}: {e}")

    return save_path


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the animation generator.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Generate an animation from figures.")
    parser.add_argument(
        "--duration", type=int, default=5, help="Duration of the animation in seconds"
    )

    parser.add_argument(
        "--source_folder",
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

    return parser.parse_args()


def main():
    """CLI entrypoint to generate animations."""

    args = parse_arguments()

    generate_animation(
        duration=args.duration,
        source_folder=args.source_folder,
        save_folder=args.save_folder,
        delete_figures=args.delete_figures,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
