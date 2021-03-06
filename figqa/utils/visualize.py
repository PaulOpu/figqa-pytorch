import os.path as pth
import json
import numpy as np
import visdom
import time
import datetime

from torch.utils.tensorboard import SummaryWriter


class VisdomVisualize():
    def __init__(self, server='localhost', port=8894, env_name='main',
                 config_file='.visdom_config.json'):
        '''
        Initialize a visdom server on server:port

        Override port and server using the local configuration from
        the json file at $config_file (containing a dict with optional
        keys 'server' and 'port').

        Credit: based on a visdom wrapper by Nirbhay Modhe
        '''
        print("Initializing visdom env [%s]"%env_name)
        if pth.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                if 'server' in config:
                    server = config['server']
                if 'port' in config:
                    port = int(config['port'])
        i = 1
        while pth.exists("/workspace/figqa-pytorch/data/logs/rn"+str(i)):
            i += 1 
        
        self.viz = visdom.Visdom(
            port=port,
            env=env_name,
            server=server,
            #username="",
            #password=""
            log_to_filename="/workspace/figqa-pytorch/data/logs/rn"+str(i)
            #base_url="/tools/8894",
            #use_incoming_socket=True
        )
        self.wins = {}

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_writer = SummaryWriter(
            log_dir=f"/workspace/figqa-pytorch/data/logs/{env_name}_{current_time}",
            comment=env_name)

    @property
    def env(self):
        return self.viz.env

    @env.setter
    def env(self, env_name):
        self.viz.env = env_name

    def append_data(self, x, y, key, line_name, xlabel="Iterations",
                    ytype="linear"):
        '''
        Add or update a plot on the visdom server self.viz

        Plots and lines are created if they don't exist, otherwise
        they are updated.

        Arguments:
            x: Scalar -> X-coordinate on plot
            y: Scalar -> Y Value at x
            key: Name of plot/graph
            line_name: Name of line within plot/graph
            xlabel: Label for x-axis (default: # Iterations)
        '''
        if key in self.wins.keys():
            self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                win=self.wins[key],
                name=line_name,
                update="append"
            )
        else:
            self.wins[key] = self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                opts=dict(
                    xlabel=xlabel,
                    ylabel=key,
                    ytype=ytype,
                    title=key,
                    marginleft=30,
                    marginright=30,
                    marginbottom=30,
                    margintop=30,
                    legend=[line_name]
                )
            )

    def append_histogram(self, x, y, chart):
        '''

        Arguments:
            x: 
            y: 
            chart:
        '''
        self.tensorboard_writer.add_histogram(chart, y, x)
        self.tensorboard_writer.close()  


        """ if key in self.wins.keys():
            self.viz.boxplot(
                X=np.array([y]),
                win=self.wins[key],
                opts=dict(legend=[x]),
                update="append"
            )
        else:
            self.wins[key] = self.viz.boxplot(
                X=y,
                opts=dict(
                    xlabel=xlabel,
                    ylabel=key,
                    ytype=ytype,
                    title=key,
                    marginleft=30,
                    marginright=30,
                    marginbottom=30,
                    margintop=30,
                    legend=[x]
                )
            ) """
