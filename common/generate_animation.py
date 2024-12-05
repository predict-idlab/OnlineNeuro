import imageio
import argparse
from pathlib import Path
import re


def natural_key(file: Path):
    # Extract numeric parts from the filename
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', file.stem)]


def generate_animation(duration: float = 10, folder_name: str = 'figures',
                       save_folder: str = "animations",
                       delete_figures: bool = False,
                       loop: int = 0
                       ):
    """
    @param duration: float, duration in seconds
    @param folder_name: path to the image folder. We assume images are enumerated naturally.
    @param save_folder: path to save the animation.
    @param delete_figures: boolean. Whether to delete the source images after saving the animation.
    @param loop: Defaults to 0, infinitely. How many loops in the animation
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
    parser = argparse.ArgumentParser(description='Generate an animation from figures.')
    parser.add_argument('--duration', type=int, default=5, help='Duration of the animation in seconds')
    parser.add_argument('--folder_name', type=str, default='figures', help='Folder containing figure files')
    parser.add_argument('--save_folder', type=str, default='animations', help='Folder to save the animation')
    parser.add_argument('--delete_figures', action='store_true', help='Delete figure files after creating the animation')
    parser.add_argument('--loop', type=int, default=0, help='Number of times the GIF should loop (0 for infinite)')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    generate_animation(duration=args.duration,
                       folder_name=args.folder_name,
                       save_folder=args.save_folder,
                       delete_figures=args.delete_figures,
                       loop=args.loop)


if __name__ == '__main__':
    main()
