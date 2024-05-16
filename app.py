import matplotlib.pyplot as plt
from flask import Flask, send_file

app = Flask(__name__)
# TODO currently the plots are simply saved locally. It would be helpful to start via threading an App (Flask or such)
# that displays in real time some plots of the optimization process.
# Or at least get things on tensorboard wandb to get an idea of what's happening.


# Generate your plot
def generate_plot():
    # Example plot
    plt.plot([1, 2, 3, 4])
    plt.ylabel('Some numbers')
    plt.savefig('plot.png')  # Save the plot as an image
    plt.close()

@app.route('/plot.png')
def plot_png():
    generate_plot()  # Regenerate the plot each time it's requested
    return send_file('plot.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)