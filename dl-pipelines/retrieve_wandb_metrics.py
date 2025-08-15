#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Docstring for collect_metrics.py
Docstrings: http://www.python.org/dev/peps/pep-0257/
"""
from __future__ import absolute_import, division, print_function
from warnings import warn

import sys, os

try:
    sys.path.append(os.getenv('pyutil'))
    from pylib import *
except Exception as e:
    warn(f'Unable to import pylib: {e}: '+sys.exc_info()[0]) 

import wandb
api = wandb.Api()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(os.path.split(__file__)[1])

    # keyword arguments
    parser.add_argument('-v', '--verbose', action='store_true',help='Verbose output')
    parser.add_argument('--entity', type=str, default='sourcefinder', help='entity (default=srcfinder)')    
    parser.add_argument('--proj', type=str, help='wandb proj id')
    parser.add_argument('--runid', type=str, help='wandb run id')

    # positional arguments 
    parser.add_argument('--path', type=str, help='path')

    args = parser.parse_args()
    
    verbose = args.verbose 
    entity = args.entity
    
    path = args.path or '.'
    jsonf = pathjoin(path,'argparse_args.json')
    if pathexists(jsonf):
        outf = pathjoin(path,"metrics.csv")
        with open(jsonf,'r') as fid:
            train_args = json.load(fid)

        proj = '_'.join(train_args['expname'].split('_')[3:])
        #rname = train_args['wandb_run_name']
        runid = train_args['wandb_run_id']
    else:
        proj = args.proj
        runid = args.runid
        outf = pathjoin(path,f"proj-{proj}_runid-{runid}_metrics.csv")

    # run is specified by <entity>/<project>/<run_id>
    run = api.run(pathjoin(entity,proj,runid))

    # save the metrics for the run to a csv file
    rdf = run.history()
    rdf.to_csv(outf,index=False)

    print('rdf:\n%s'%str((rdf)))

    
    print(f'saved: {outf}')
