import os
import glob

jobNames = ['dropout_kernel', 'dropout_maxpooling', 'kernelsize', 'pf_pk', 
                'pf_vk', 'vf_pk', 'vf_vk', 'uniform_dropout', 
                 'uniform_layers', 'uniform_smaller']
def submitJobs():
    for job in jobNames:
        os.system('sbatch ' + 'run_cnn_' + job + '.sl')
        
# submitJobs()

def replaceString():
    for job in jobNames:
        scriptContents = ''
        filename = 'run_cnn_' + job + '.sh'
        with open(filename, 'r') as file:
            scriptContents = file.read()
        # borg1 = 'checkpointer'
        # borg2 = '# checkpointer'
        # scriptContents = scriptContents.replace(borg1, borg2)
        borg3 = 'medium_3_channel'
        borg4 = 'medium_diff'
        scriptContents = scriptContents.replace(borg3, borg4)

        with open(filename, 'w') as file:
            file.write(scriptContents)

def findString():
    writeFile = "diff_channel_results"
    lookup1 = "Missed detection rate: "
    lookup2 = "False positive rate: "
    lookup3 = "/global/homes/l/liuto/520project/transient-detection/"
    with open(writeFile + '.txt', 'w+') as results:
        for slurmFile in glob.glob("slurm*.out"):
            with open(slurmFile, 'r') as file:
                print(slurmFile)
                for num, line in enumerate(file,1):
                    if lookup1 in line:
                        results.write(line)
                    if lookup2 in line:
                        results.write(line)
                    if lookup3 in line:
                        results.write(line.split(' ')[3]+'\n')
    print("done")
                        
findString()

# replaceString()

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
