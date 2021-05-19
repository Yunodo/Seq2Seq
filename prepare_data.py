"""
Function that modifies Python generator for data processing of text data
    f: Python generator -> Python generator
        args:
            - vocab_file: either of 'subword', 'sentencepiece' or 'char'
            - BATCH_SIZE
"""
import trax
import trax.data as d

def text_data_generator(vocab_file = 'en_8k.subword', BATCH_SIZE):

    def Flip():
        # Modifies Python generator to (input, flipped input, target)
        # Used in bidirectional lstm models
        def helper(g):
          for temp in g:
            yield temp[0], np.flip(temp[0], axis = 1), temp[1]
        return lambda g: helper(g)

    data_generator = d.Serial(
      d.Tokenize(vocab_file=vocab_file),
      d.FilterByLength(max_length=120, length_keys=[0]),
      d.BucketByLength(boundaries=[120], batch_sizes=[BATCH_SIZE, 1], length_keys=[0]),
      Flip()
      )
    return data_generator
