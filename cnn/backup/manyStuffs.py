import os

jobNames = ['dropout_kernel', 'dropout_maxpooling', 'kernelsize', 'pf_pk', 
                'pf_vk', 'vf_pk', 'vf_vk', 'uniform_dropout', 
                 'uniform_layers', 'uniform_smaller']


def writeSlFiles():
    for job in jobNames:
        filename = 'run_cnn_' + job + '.sl'
        with open(filename, 'w+') as file:
            file.write('#!/bin/bash\n')
            file.write('#SBATCH -q regular\n')
            file.write('#SBATCH -N 2\n')
            file.write('#SBATCH -t 3:00:00\n')
            file.write('#SBATCH -L SCRATCH\n')
            file.write('#SBATCH -C haswell\n')
            file.write('cd ${SLURM_SUBMIT_DIR}\n')
            file.write('sh run_cnn_' + job + '.sh')


def writeShFiles():
    for job in jobNames:
        filename = 'run_cnn_' + job + '.sh'
        with open(filename, 'w+') as file:
            file.write('#!/bin/bash\n')
            file.write('module load python/2.7-anaconda-4.4\n')
            file.write('source activate convNet\n')
            pythonFile = ('~/520project/transient-detection/'+
                        'cnn/jobs/run_cnn_' + job +'.py')
            dataFile = 'medium_3_channel'
            file.write('python ' + pythonFile + ' ' + 
                            dataFile + ' ' + job + '\n')



writeSlFiles()                            
