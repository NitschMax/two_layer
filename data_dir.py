import os

def get_dir():
    directory   = "/home/maximilian/Documents/studium/research_projects/forschungsprojekt_kopenhagen/code/sandpile_project/data/"
    if not os.path.exists(directory):
        directory   = 0
    return directory
