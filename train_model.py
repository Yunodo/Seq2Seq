import os
import numpy as np
import trax
import jax
import jax.numpy as np

from prepare_data import text_data_generator
from model import biLSTMwithAttn
from training_loop import create_training_loop

%load_ext tensorboard

vocab_file = 'en_8k.subword' # one of: 'subword', 'sentencepiece', 'char'

BATCH_SIZE = 64
n_training_steps = 30000

input_vocab_size = 8000
target_vocab_size = 8000
d_model = 512 # depth of inner layers of LSTM


def txt_reader(filepath):
    # Reads a csv file and returns Python tuple generator
    df = pd.read_csv(filepath)
    while True:
        try:
            for row in df.itertuples(index = False):
                yield(row[0], row[1]) # (input, target)
        except StopIteration:
            df = pd.read_csv(filepath)
train_data, eval_data = txt_reader('train.csv'), txt_reader('test.csv')

data_generator = text_data_generator(vocab_file, BATCH_SIZE)
train_stream, eval_stream =  data_generator(train_data), data_generator(eval_data)


"""
Parameters for the model are automatically initialized with Trax training loop.
If there're some pretrained weights, use:
model.init_from_file(os.path.join(output_dir,'model.pkl.gz'),weights_only=True)
"""
Net = biLSTMwithAttn(input_vocab_size = input_vocab_size,
        target_vocab_size = 8000, d_model =  d_model, mode = 'train')

output_dir = "/content/drive/My Drive/paraphrase'" # weights, logs etc. stored


loop = create_training_loop(Net, train_stream,
            eval_stream, output_dir)
%tensorboard --logdir output_dir # displaying TensorBoard training results
loop.run(n_training_steps)

"""
How to resume checkpoint if training broke down:
loop.load_checkpoint(directory=output_dir, filename="model.pkl.gz")
"""
