import os

def get_dir():
    directory   = "/home/maximilian/Documents/studium/research_projects/forschungsprojekt_kopenhagen/code/two_layer_project/data/descend_and_dry_out/"
    if not os.path.exists(directory):
        directory   = '/home/nim60855/Documents/two_layer_project/data/descend_and_dry_out/'
    return directory
