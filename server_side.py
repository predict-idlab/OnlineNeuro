import os
import socket
import json
import threading
import sys
import pandas as pd
import numpy as np

import plotly
import matlab.engine
import tensorflow as tf
import warnings
import time

import trieste
from trieste.data import Dataset
from bayessian_optimizer import BayesianOptimizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import BayesianActiveLearningByDisagreement, \
    ExpectedFeasibility, NegativePredictiveMean, IntegratedVarianceReduction, \
    PredictiveVariance, ExpectedHypervolumeImprovement
# https://github.com/secondmind-labs/trieste/blob/c6a039aa9ecf413c7bcb400ff565cd283c5a16f5/trieste/acquisition/function/__init__.py
from online_learning import build_model
from trieste.acquisition.multi_objective.pareto import (
    Pareto,
    get_reference_point,
)

# AugmentedExpectedImprovement
# ExpectedImprovement
# ProbabilityOfImprovement
# NegativeLowerConfidenceBound
# NegativePredictiveMean
# ProbabilityOfFeasibility

# FastConstraintsFeasibility
# ExpectedConstrainedImprovement
# MonteCarloExpectedImprovement
# MonteCarloAugmentedExpectedImprovement
# MonteCarloExpectedImprovement
# BatchMonteCarloExpectedImprovement
# BatchExpectedImprovement
# MultipleOptimismNegativeLowerConfidenceBound

# MakePositive
from trieste.acquisition.rule import OBJECTIVE
from threading import Thread

from plotting import update_plot
#from flask import Flask, send_file
# app = Flask(__name__)
#
# @app.route('/plot.png')
# def plot_png(file):
#     return send_file('plot.png', mimetype='image/png')


def run_matlab_main(**kwargs):
    # TODO this function can be modified so that arguments are passed to the matlab engine

    # Start MATLAB engine
    print("Launching Matlab from Python side")
    eng = matlab.engine.start_matlab()

    # Change directory to current directory
    current_directory = os.path.dirname(os.path.abspath(__file__))

    eng.cd(current_directory)  # Change path to your directory where main.m resides

    # Call MATLAB function main
    confirm_bool = bool(kwargs['matlab_initiate'])
    eng.main(confirm_bool, nargout=0)  # nargout=0 means we're not expecting any output from the MATLAB function

    # Stop MATLAB engine
    eng.quit()


def main(matlab_call=False, *args, **kwargs) -> None:
    # Load port configuration
    print(f"matlab_call flag: {matlab_call}")
    with open('config.json', 'r') as f:
        config = json.load(f)

    # TODO verify here that the problem type matches model type
    # i.e. prevent multiobjective/regression to be solved with classification, etc...

    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Establishing connection at port {config['ip']} with port {config['port']}")
    # Bind the socket to the address and port
    server_socket.bind((config['ip'], config['port']))

    # Listen for incoming connections
    server_socket.listen(1)
    print("Waiting for a connection...")

    #Start Matlab process via engine and threading
    # if matlab_call:
    #     # If threading, Python launches the Matlab main, else, main needs to be manually launched
    #     t = threading.Thread(target=run_matlab_main, kwargs={"matlab_initiate": False})
    #     t.start()

    # Accept a connection
    client_socket, client_address = server_socket.accept()
    print("Connection from:", client_address)

    # Receive data
    data = client_socket.recv(1024).decode()
    data = json.loads(data)
    print(f"Received data:", data)

    # Respond back
    response = {"message": "Hello from Python",
                "randomNumber": data['dummyNumber']}

    response_json = json.dumps(response) + "\n"
    client_socket.sendall(response_json.encode())
    print("Data sent back to matlab", response_json.encode())

    ## First batch of data
    # Passes 'name of fun'
    # num of features, num of targets, and UB, LB.
    received_data = client_socket.recv(1024).decode()
    exp_config = json.loads(received_data)
    print("Received from MATLAB:", received_data)

    ## Create the first fit, using experiment configuration
    # Send this values to query from matlab
    search_space = trieste.space.Box(lower=exp_config['lower_bound'],
                                     upper=exp_config['upper_bound'])

    #TODO Optional. Add other sampling methods such as LHS (Halton is in Trieste but not needed).

    initial_qp = search_space.sample_sobol(num_samples=config['experiment']['init_samples'])

    response = {'message': 'first queried points using Sobol method',
                'query_points': initial_qp.numpy().tolist()}

    response_json = json.dumps(response) + "\n"
    client_socket.sendall(response_json.encode())

    received_data = client_socket.recv(1024).decode()
    received_data = json.loads(received_data)
    if 'tot_pckgs' in received_data:
        rest_of_msg = []
        for _ in range(received_data['tot_pckgs'] - 1):
            msg = client_socket.recv(1024).decode()
            msg = json.loads(msg)
            rest_of_msg.append(msg)

        for msg in rest_of_msg:
            for k, v in msg.items():
                if k != 'tot_pckgs':
                    received_data[k] += v

    print("First data package:", received_data)

    query_points = np.stack(initial_qp.numpy())
    observations = np.array(received_data['init_response'])
    init_data_size = len(query_points)

    if observations.ndim <= 1:
        observations = observations.reshape(-1, 1)

    #Important note. For some tensorflow reason observations need to be float even for classification problems.
    #Important note. Tf variance can only accept real numbers
    init_dataset = Dataset(query_points=tf.cast(query_points, tf.float64),
                           observations=tf.cast(observations, tf.float64))

    if config['experiment']['noise_free']:
        config['kernel_variance'] = 1e-6
    else:
        config['kernel_variance'] = None

    print("Init dataset")
    print(init_dataset)
    model = build_model(init_dataset, search_space, config)
    # TODO extend to batch_sampling at a later stage
    if config['experiment']['batch_sampling']:
        if 'num_query_points' in config['experiment']:
            num_query_points = config['experiment']['num_query_points']
        else:
            #Default
            num_query_points = 3
    else:
        num_query_points = 1

    if config['problem'] == 'multiobjective':
        acq = ExpectedHypervolumeImprovement()
        print("Multiobjective model using HyperVolumes")
    elif config['experiment']['classification']:
        acq = BayesianActiveLearningByDisagreement()
        print("using Classification BALD")
    else:
        #Sampl close to a given threshold
        #acq = ExpectedFeasibility(threshold=-0.5)
        #Equivalent to Maximizing a function
        #acq = NegativePredictiveMean()
        #Minimize global variance
        acq = PredictiveVariance()
        #acq = IntegratedVarianceReduction()

    rule = EfficientGlobalOptimization(builder=acq,
                                       num_query_points=num_query_points
                                       )

    os.makedirs(f"{config['save_path']}/{exp_config['name']}/", exist_ok=True)
    bo = BayesianOptimizer(observer=exp_config['name'],
                           search_space=search_space,
                           track_state=True,
                           track_path=f"{config['save_path']}/{exp_config['name']}/",
                           acquisition_rule=rule)

    terminate_flag = False
    qp = bo.request_query_points(datasets=init_dataset,
                                 models=model,
                                 fit_initial_model=True,
                                 fit_model=True)

    if qp is None:
        terminate_flag = True
        warnings.warn("Terminated before optimization started \n no query points were retrieved")

    qp_list = qp.numpy().tolist()

    with_plots = True
    count = 0

    # TODO send stop signals from both ends upon certain conditions (i.e. no improvement, reached goal,
    # buget limit, etc...)

    # TODO, for toy-problems, add the option of generating performance results (i.e. Accuracy/RMSE over
    # updates

    # TODO, for real-world problems, generate the GridSearch solution, store (within GIT?) And use to
    # generate example figures of online learning efficiency

    # TODO, plotting functions for Pareto

    # TODO, plotting functions for 3D search shapes (see Notebook in NeuoAAA repo)

    while not terminate_flag:
        #Counter is used to save figures with different names. If kept constant it overwrites the figure.
        count += 1

        # Check if termination signal received from MATLAB
        # Respond back
        message = {"query_points": qp_list,
                   "terminate_flag": terminate_flag}

        response_json = json.dumps(message) + "\n"
        client_socket.sendall(response_json.encode())

        if "terminate_flag" in received_data:
            terminate_flag = received_data['terminate_flag']
            print("Termination signal received from MATLAB \n Saving results... \n Closing connection...")

        received_data = client_socket.recv(1024).decode()
        received_data = json.loads(received_data)

        if with_plots:
            plot_data = bo.result.try_get_final_datasets()[OBJECTIVE]
            update_plot(bo,
                        #eval_points=None,
                        train_data=(plot_data.query_points[:init_data_size],
                                    plot_data.observations[:init_data_size]),
                        sampled_data=(plot_data.query_points[init_data_size:],
                                      plot_data.observations[init_data_size:]),
                        ground_truth=None,
                        init_fun=None,
                        count=count
                        )
            time.sleep(1)

        observations = np.array(received_data['observations'])
        if observations.ndim <= 1:
            # TODO Need to fix this to allow batch sampling, but also multivariate modeling.
            # Right not this is basically hardcoded
            if config['problem'] == 'multiobjective':
                observations = observations.reshape(-1, 2)
            else:
                observations = observations.reshape(-1, 1)

        bo.optimize_step(query_points=qp,
                         observer_output=observations)

        final_data = bo.result.try_get_final_datasets()[OBJECTIVE]
        final_models = bo.result.try_get_final_models()[OBJECTIVE]

        qp = bo.request_query_points(datasets=final_data,
                                     models=final_models,
                                     fit_initial_model=False,
                                     fit_model=True)

        qp_list = qp.numpy().tolist()

    # Close the connection
    client_socket.close()
    server_socket.close()


if __name__ == "__main__":
    if "--matlab_call" in sys.argv:
        matlab_call = True
    else:
        matlab_call = False

    main(matlab_call=matlab_call)