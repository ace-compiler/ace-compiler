#!/usr/bin/python3

import os
import sys
import argparse
import matplotlib.pyplot as plt
import subprocess
import datetime

MODELS = ['ResNet-20', 'ResNet-32', 'ResNet-32*', 'ResNet-44', 'ResNet-56', 'ResNet-110']
INDEXES = ['resnet20_cifar10', 'resnet32_cifar10', 'resnet32_cifar100', 'resnet44_cifar10', 'resnet56_cifar10', 'resnet110_cifar10']

def write_log(info, log):
    print(info[:-1])
    log.write(info)
    log.flush()
    return

def time_and_memory(outputs):
    result = outputs.strip('"').split(' ')
    return result[0], result[1]

def generate_accuracy(raw_acc, ace_acc, log):
    '''
    Generate accuracy table (Table 10 in paper)
    '''
    acc_gain = []
    for idx in range(len(ace_acc)):
        gain = (ace_acc[idx] - raw_acc[idx]) * 100 / raw_acc[idx]
        gain = "%.1f" % round(gain, 1)
        acc_gain.append(gain + '%')
    rows = []
    for idx in range(len(ace_acc)):
        rows.append([MODELS[idx], str(raw_acc[idx]) + '%', str(ace_acc[idx]) + '%', acc_gain[idx]])

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.table(cellText=rows, \
             colLabels=['Model', 'Unencrypted', 'Encrypted', 'Accuracy Gain'], \
             cellLoc='center', loc='center')
    fig.tight_layout()
    plt.savefig('Table10.pdf')
    plt.close()
    info = 'Table10.pdf generated!\n'
    write_log(info, log)
    return

def run_raw_accuracy(image_num, log, debug):
    '''
    Run unencrypted accuracy test for all models with given image numbers
    '''
    info = '-------- Unencrypted Accuracy for %d Images --------\n' % image_num
    write_log(info, log)
    acc_res = []
    cifar_path = '/app/cifar'
    if not os.path.exists(cifar_path):
        print(cifar_path, 'does not exist!')
        sys.exit(-1)
    cifar10_script = '/app/scripts/infer_model.py'
    if not os.path.exists(cifar10_script):
        print(script, 'does not exist!')
        sys.exit(-1)
    cifar100_script = '/app/scripts/infer_model_cifar100.py'
    if not os.path.exists(cifar100_script):
        print(script, 'does not exist!')
        sys.exit(-1)
    onnx_path = '/app/model'
    if not os.path.exists(onnx_path):
        print(onnx_path, 'does not exist!')
        sys.exit(-1)
    model_files = [f for f in os.listdir(onnx_path) if os.path.isfile(os.path.join(onnx_path, f))]
    total_time = 0.0
    max_memory = 0.0
    for rn in INDEXES:
        script = cifar10_script
        if rn.find('cifar100') != -1:
            script = cifar100_script
        onnx_file = None
        for f in model_files:
            if f.find(rn) != -1:
                onnx_file = os.path.join(onnx_path, f)
                break
        if not os.path.exists(onnx_file):
            print(onnx_file, 'does not exists!')
            sys.exit(-1)
        cmds = ['time', '-f', '\"%e %M\"', 'python3', script, '--loader_onnx', onnx_file, '--datasets', cifar_path, '--images', str(image_num)]
        if debug:
            print(' '.join(cmds))
        ret = subprocess.run(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        accuracy = 0.0
        info = '%s: Inference Failed!\n' % rn
        if ret.returncode == 0:
            rtime, rmemory = time_and_memory(ret.stderr.decode().splitlines()[0])
            time = float(rtime)
            memory = float(rmemory) / 1000000
            total_time += time
            if max_memory < memory:
                max_memory = memory
            accuracy = float(ret.stdout.decode().splitlines()[-1].split(':')[1].strip()) * 100
            info = '%s: Unencrypted Accuracy = %.1f%%\n' % (rn, accuracy)
        acc_res.append(accuracy)
        write_log(info, log)
    info = 'Total time = %.2f(s), Max memory usage = %.1f(Gb)\n' % (total_time, max_memory)
    write_log(info, log)
    return acc_res

def run_ace_accuracy(image_num, log, debug):
    '''
    Run ACE accuracy test for all models with given image numbers
    '''
    info = '-------- Encrypted Accuracy via ACE for %d Images --------\n' % image_num
    write_log(info, log)
    acc_res = []
    script = '/app/scripts/accuracy.sh'
    if not os.path.exists(script):
        print(script, 'does not exist!')
        sys.exit(-1)
    total_time = 0.0
    max_memory = 0.0
    for rn in INDEXES:
        cmds = ['time', '-f', '\"%e %M\"', script, rn, '0', str(image_num)]
        if debug:
            print(' '.join(cmds))
        ret = subprocess.run(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        accuracy = 0.0
        err_log = '/app/release_openmp/' + rn + '.acc.0.' + str(image_num-1) + '.log'
        info = '%s: Inference Failed! Refer to \'%s\' for details.\n' % (rn, err_log)
        if ret.returncode == 0:
            rtime, rmemory = time_and_memory(ret.stderr.decode().splitlines()[0])
            time = float(rtime)
            memory = float(rmemory) / 1000000
            total_time += time
            if max_memory < memory:
                max_memory = memory
            msg = ret.stdout.decode().splitlines()
            for line in msg:
                if line.find('[RESULT]') == -1:
                    continue
                item = line.split(' ')
                accuracy = float(item[6].strip().split(',')[0].strip()) * 100
            info = '%s: Accuracy = %.1f%%, Time = %.2f(s), Memory = %.1f(GB)\n' % (rn, accuracy, time, memory)
        acc_res.append(accuracy)
        write_log(info, log)
    info = 'Total time = %.2f(s), Max memory usage = %.1f(Gb)\n' % (total_time, max_memory)
    write_log(info, log)
    return acc_res

def main():
    parser = argparse.ArgumentParser(description='Run accuracy data for ACE Framework')
    parser.add_argument('-n', '--num', type=int, default=10, help='Number of images to run for each model, ranges: [0, 10000]')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='Print out debug info')
    args = parser.parse_args()
    debug = args.debug
    image_num = args.num
    date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    log_file_name = date_time + '_acc_' + str(image_num) + '.log'
    log = open(os.path.join(os.getcwd(), log_file_name), 'w')
    raw_acc = run_raw_accuracy(image_num, log, debug)
    ace_acc = run_ace_accuracy(image_num, log, debug)
    info = '-------- Accuracy Test Done --------\n'
    write_log(info, log)
    generate_accuracy(raw_acc, ace_acc, log)
    log.close()
    return

if __name__ == "__main__":
    main()
