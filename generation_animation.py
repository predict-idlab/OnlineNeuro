import imageio
import os


def generate_animation(iter_values, duration=3, folder_name='figures',
                        save_folder=None,
                       delete_figures=False
                       ):
    #TODO implement delete_figures, in case we want to get rid of per-trial images and just keep the animation
    #TODO implement target save_folder

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    #filenames = [f'.{folder_name}/plot_{i+1:2d}.png' for i in range(iter_values)]
    filenames = os.listdir(f'./{folder_name}/')
    filenames.sort()
    # with imageio.get_writer('./figures/animation.gif', loop=0, mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

    images = []
    for file in filenames:
        image = imageio.imread(f'./{folder_name}/{file}')
        images.append(image)
    #images = list(map(lambda filename: imageio.imread(filename), filenames))
    imageio.mimsave(os.path.join('./figures/animation1.gif'), images, duration=duration) # modify the frame duration as needed
