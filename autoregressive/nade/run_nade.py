# Copyright 2011 Hugo Larochelle. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
# 
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY Hugo Larochelle ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo Larochelle.

import sys,os,fcntl

if __name__== '__main__':
    if len(sys.argv) != 8:
        print 'Usage: run_nade dataset learning_rate decrease_constant hidden_size seed order_seed untied_weights'
        sys.exit()
    
    dataset = sys.argv[1]
    learning_rate = float(sys.argv[2])
    decrease_constant = float(sys.argv[3])
    hidden_size = int(sys.argv[4])
    seed = int(sys.argv[5])
    order_seed = int(sys.argv[6])
    untied_weights = bool(int(sys.argv[7]))

else:
    dataset = 'ocr_letters'
    learning_rate = 0.001
    learning_rate = 0.
    hidden_size = 100
    seed = 1234
    order_seed = 1234
    untied_weights = False

import nade
import copy, time
import numpy as np
import mlpython.mlproblems.generic as mlpb

print 'Loading dataset'
datadir = os.path.abspath(os.path.curdir) + '/data/'

datasets = ['adult',
            'binarized_mnist',
            'connect4',
            'dna',
            'mushrooms',
            'nips',
            'ocr_letters',
            'rcv1',
            'web']

if dataset not in datasets:
    raise ValueError('dataset '+dataset+' unknown')

exec 'import mlpython.datasets.'+dataset+' as mldataset'
exec 'datadir = datadir + \''+dataset+'/\''
all_data = mldataset.load(datadir,load_to_memory=True)

train_data, train_metadata = all_data['train']
if dataset == 'binarized_mnist' or dataset == 'nips': 
    trainset = mlpb.MLProblem(train_data,train_metadata)
else:
    trainset = mlpb.SubsetFieldsProblem(train_data,train_metadata)

trainset.setup()

valid_data, valid_metadata = all_data['valid']
validset = trainset.apply_on(valid_data,valid_metadata)

test_data, test_metadata = all_data['test']
testset = trainset.apply_on(test_data,test_metadata)

# Preparing result file
result_file = 'results_' + dataset + '_nade.txt'
header_line = 'lr\tdc\tnhid\tseed\torder_seed\tuntied\tbest_it\ttrain\ttr_std\tvalid\tva_std\ttest\tte_std\n'
if not os.path.exists(result_file):
    file = open(result_file, "w")
    file.write(header_line)
    file.close()

print 'Training NADE'

rng_order = np.random.mtrand.RandomState(order_seed)
input_order = range(train_metadata['input_size'])
rng_order.shuffle(input_order)

model = nade.NADE(n_stages = 1,
                  learning_rate = learning_rate, 
                  decrease_constant = decrease_constant,
                  hidden_size = hidden_size,
                  seed = seed,
                  input_order = input_order,
                  untied_weights = untied_weights)

best_val_error = np.inf
best_test_error = np.inf
best_train_error = np.inf
best_val_error_std = np.inf
best_test_error_std = np.inf
best_train_error_std = np.inf
best_it = 0

look_ahead = 10
n_incr_error = 0

for stage in range(1,501):
    if not n_incr_error < look_ahead:
        break
    model.n_stages = stage
    this_time = time.time()
    print 'Training epoch',stage,'...',
    sys.stdout.flush()
    model.train(trainset)
    print 'finished after',time.time()-this_time,'seconds'
    sys.stdout.flush()

    print 'Evaluating on validation set ...',
    sys.stdout.flush()
    this_time = time.time()
    outputs, costs = model.test(validset)
    error = np.mean(costs,axis=0)[0]
    error_std = np.std(costs,axis=0,ddof=1)[0]/np.sqrt(len(costs))
    print 'finished after',time.time()-this_time,'seconds'
    print 'NLL: ' + str(error)
    sys.stdout.flush()
    if error < best_val_error:
        best_val_error = error
        best_val_error_std = error_std
        best_it = stage
        n_incr_error = 0
        best_model = copy.deepcopy(model)
    else:
        n_incr_error += 1


print 'Evaluating best model on train and test set'
outputs, costs = best_model.test(trainset)
best_train_error = np.mean(costs,axis=0)[0]
best_train_error_std = np.std(costs,axis=0,ddof=1)[0]/np.sqrt(len(costs))
outputs, costs = best_model.test(testset)
best_test_error = np.mean(costs,axis=0)[0]
best_test_error_std = np.std(costs,axis=0,ddof=1)[0]/np.sqrt(len(costs))

model_info = [str(learning_rate),str(decrease_constant),str(hidden_size),str(seed),str(order_seed),str(untied_weights),str(best_it),str(best_train_error),str(best_train_error_std),str(best_val_error),str(best_val_error_std),str(best_test_error),str(best_test_error_std)]
line = '\t'.join(model_info)+'\n'
file = open(result_file, "a")
fcntl.flock(file.fileno(), fcntl.LOCK_EX)
file.write(line)
file.close() # unlocks the file
