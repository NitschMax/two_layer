import os

def get_dir():
    directory   = "/home/maximilian/Documents/studium/research_projects/forschungsprojekt_kopenhagen/code/two_layer_project/data/dry_out/"
    if not os.path.exists(directory):
        directory   = '/home/gorm/Documents/Max_Nitsch/two_layer_project/data/'
    return directory
