"""
Creates an instance of Trax training loop
"""

import trax
from trax.supervised import training
from metrics import SequenceLoss, SequenceAccuracy

def create_training_loop(model, train_data, eval_data, output_dir):

    train_task = training.TrainTask(
        labeled_data=train_data,
        loss_layer=SequenceLoss(),
        optimizer=trax.optimizers.Adam(0.001),
        n_steps_per_checkpoint=200,
    )

    eval_task = training.EvalTask(
        labeled_data=eval_data,
        metrics=[SequenceLoss(), SequenceAccuracy()],
    )

    return training.Loop(model, train_task, eval_tasks=[eval_task],
                    output_dir = output_dir)
