import imageio
import os
import argparse


def generate_animation(duration=3, folder_name='figures',
                       save_folder="./animations/",
                       delete_figures=False,
                       loop=0
                       ):
    #TODO implement delete_figures, in case we want to get rid of per-trial images and just keep the animation
    #TODO implement target save_folder

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    filenames = os.listdir(f'./{folder_name}/')
    filenames.sort()

    images = []
    for file in filenames:
        image = imageio.imread(f'./{folder_name}/{file}')
        images.append(image)
    #images = list(map(lambda filename: imageio.imread(filename), filenames))
    imageio.mimsave(os.path.join(f'{save_folder}animation1.gif'), images, duration=duration, loop=0) # modify the frame duration as needed

    if delete_figures:
        for file in filenames:
            try:
                file_path = f'./{folder_name}/{file}'
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Generate an animation from figures.')
    parser.add_argument('--duration', type=int, default=5, help='Duration of the animation in seconds')
    parser.add_argument('--folder_name', type=str, default='figures', help='Folder containing figure files')
    parser.add_argument('--save_folder', type=str, default='./animations/', help='Folder to save the animation')
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