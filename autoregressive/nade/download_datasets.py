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
    if len(sys.argv) != 1:
        print 'Usage: download_datasets'
        sys.exit()

import mlpython.datasets.adult as adult
import mlpython.datasets.binarized_mnist as binarized_mnist
import mlpython.datasets.connect4 as connect4
import mlpython.datasets.dna as dna
import mlpython.datasets.mushrooms as mushrooms
import mlpython.datasets.nips as nips
import mlpython.datasets.ocr_letters as ocr_letters
import mlpython.datasets.rcv1 as rcv1
import mlpython.datasets.web as web

if not os.path.exists('data/'):
    os.makedirs('data/')

datasets = [ 'adult',
             'binarized_mnist',
             'connect4',
             'dna',
             'mushrooms',
             'nips',
             'ocr_letters',
             'rcv1',
             'web']

for dataset in datasets:
    print '* Obtaining dataset',dataset,'*'    
    dir = 'data/'+dataset
    if not os.path.exists(dir):
        os.makedirs(dir)
    eval(dataset+'.obtain(dir)')
    print ''
