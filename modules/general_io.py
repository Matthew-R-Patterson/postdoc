"""
Collection of functions for file input and output 

"""
# general python modules
import os
import sys
import iris


# my own modules
cwd = os.getcwd()
repo_dir = '/'
for directory in cwd.split('/')[1:]:
    repo_dir = os.path.join(repo_dir, directory)
    if directory == 'postdoc':
        break

modules_dir = os.path.join(repo_dir, 'modules')
sys.path.append(modules_dir)



