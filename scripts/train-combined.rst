.. _train-tensorflow-overview:

Get Started with Distributed Training using TensorFlow/Keras
============================================================

Ray Train's `TensorFlow <https://www.tensorflow.org/>`__ integration enables you
to scale your TensorFlow and Keras training functions to many machines and GPUs.

On a technical level, Ray Train schedules your training workers
and configures ``TF_CONFIG`` for you, allowing you to run
your ``MultiWorkerMirroredStrategy`` training script. See `Distributed
training with TensorFlow <https://www.tensorflow.org/guide/distributed_training>`_
for more information.

Most of the examples in this guide use TensorFlow with Keras, but
Ray Train also works with vanilla TensorFlow.


Quickstart
-----------
.. literalinclude:: ./doc_code/tf_starter.py
  :language: python
  :start-after: __tf_train_start__
  :end-before: __tf_train_end__


Update your training function
-----------------------------

First, update your :ref:`training function <train-overview-training-function>` to support distributed
training.


.. note::
   The current TensorFlow implementation supports
   ``MultiWorkerMirroredStrategy`` (and ``MirroredStrategy``). If there are
   other strategies you wish to see supported by Ray Train, submit a `feature request on GitHub <https://github.com/ray-project/ray/issues>`_.

These instructions closely follow TensorFlow's `Multi-worker training
with Keras <https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras>`_
tutorial. One key difference is that Ray Train handles the environment
variable set up for you.

**Step 1:** Wrap your model in ``MultiWorkerMirroredStrategy``.

The `MultiWorkerMirroredStrategy <https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy>`_
enables synchronous distributed training. You *must* build and compile the ``Model`` within the scope of the strategy.

.. testcode::
    :skipif: True

    with tf.distribute.MultiWorkerMirroredStrategy().scope():
        model = ... # build model
        model.compile()

**Step 2:** Update your ``Dataset`` batch size to the *global* batch
size.

Set ``batch_size`` appropriately because `batch <https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch>`_
splits evenly across worker processes.

.. code-block:: diff

    -batch_size = worker_batch_size
    +batch_size = worker_batch_size * train.get_context().get_world_size()


.. warning::
    Ray doesn't automatically set any environment variables or configuration
    related to local parallelism or threading
    :ref:`aside from "OMP_NUM_THREADS" <omp-num-thread-note>`.
    If you want greater control over TensorFlow threading, use
    the ``tf.config.threading`` module (eg.
    ``tf.config.threading.set_inter_op_parallelism_threads(num_cpus)``)
    at the beginning of your ``train_loop_per_worker`` function.

Create a TensorflowTrainer
--------------------------

``Trainer``\s are the primary Ray Train classes for managing state and
execute training. For distributed Tensorflow,
use a :class:`~ray.train.tensorflow.TensorflowTrainer`
that you can setup like this:

.. testcode::
    :hide:

    train_func = lambda: None

.. testcode::

    from ray.train import ScalingConfig
    from ray.train.tensorflow import TensorflowTrainer
    # For GPU Training, set `use_gpu` to True.
    use_gpu = False
    trainer = TensorflowTrainer(
        train_func,
        scaling_config=ScalingConfig(use_gpu=use_gpu, num_workers=2)
    )

To customize the backend setup, you can pass a
:class:`~ray.train.tensorflow.TensorflowConfig`:

.. testcode::
    :skipif: True

    from ray.train import ScalingConfig
    from ray.train.tensorflow import TensorflowTrainer, TensorflowConfig

    trainer = TensorflowTrainer(
        train_func,
        tensorflow_backend=TensorflowConfig(...),
        scaling_config=ScalingConfig(num_workers=2),
    )


For more configurability, see the :py:class:`~ray.train.data_parallel_trainer.DataParallelTrainer` API.


Run a training function
-----------------------

With a distributed training function and a Ray Train ``Trainer``, you are now
ready to start training.

.. testcode::
    :skipif: True

    trainer.fit()

Load and preprocess data
------------------------

TensorFlow by default uses its own internal dataset sharding policy, as described
`in the guide <https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#dataset_sharding>`__.
If your TensorFlow dataset is compatible with distributed loading, you don't need to
change anything.

If you require more advanced preprocessing, you may want to consider using Ray Data
for distributed data ingest. See :ref:`Ray Data with Ray Train <data-ingest-torch>`.

The main difference is that you may want to convert your Ray Data dataset shard to
a TensorFlow dataset in your training function so that you can use the Keras
API for model training.

`See this example <https://github.com/ray-project/ray/blob/master/python/ray/train/examples/tf/tune_tensorflow_autoencoder_example.py>`__
for distributed data loading. The relevant parts are:

.. testcode::

    import tensorflow as tf
    from ray import train
    from ray.train.tensorflow import prepare_dataset_shard

    def train_func(config: dict):
        # ...

        # Get dataset shard from Ray Train
        dataset_shard = train.get_context().get_dataset_shard("train")

        # Define a helper function to build a TensorFlow dataset
        def to_tf_dataset(dataset, batch_size):
            def to_tensor_iterator():
                for batch in dataset.iter_tf_batches(
                    batch_size=batch_size, dtypes=tf.float32
                ):
                    yield batch["image"], batch["label"]

            output_signature = (
                tf.TensorSpec(shape=(None, 784), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 784), dtype=tf.float32),
            )
            tf_dataset = tf.data.Dataset.from_generator(
                to_tensor_iterator, output_signature=output_signature
            )
            # Call prepare_dataset_shard to disable automatic sharding
            # (since the dataset is already sharded)
            return prepare_dataset_shard(tf_dataset)

        for epoch in range(epochs):
            # Call our helper function to build the dataset
            tf_dataset = to_tf_dataset(
                dataset=dataset_shard,
                batch_size=64,
            )
            history = multi_worker_model.fit(tf_dataset)



Report results
--------------
During training, the training loop should report intermediate results and checkpoints
to Ray Train. This reporting logs the results to the console output and appends them to
local log files. The logging also triggers :ref:`checkpoint bookkeeping <train-dl-configure-checkpoints>`.

The easiest way to report your results with Keras is by using the
:class:`~ray.train.tensorflow.keras.ReportCheckpointCallback`:

.. testcode::

    from ray.train.tensorflow.keras import ReportCheckpointCallback

    def train_func(config: dict):
        # ...
        for epoch in range(epochs):
            model.fit(dataset, callbacks=[ReportCheckpointCallback()])


This callback automatically forwards all results and checkpoints from the
Keras training function to Ray Train.


Aggregate results
~~~~~~~~~~~~~~~~~

TensorFlow Keras automatically aggregates metrics from all workers. If you wish to have more
control over that, consider implementing a `custom training loop <https://www.tensorflow.org/tutorials/distribute/custom_training>`__.


Save and load checkpoints
-------------------------

You can save :class:`Checkpoints <ray.train.Checkpoint>` by calling ``train.report(metrics, checkpoint=Checkpoint(...))`` in the
training function. This call saves the checkpoint state from the distributed
workers on the ``Trainer``, where you executed your python script.

You can access the latest saved checkpoint through the ``checkpoint`` attribute of
the :py:class:`~ray.train.Result`, and access the best saved checkpoints with the ``best_checkpoints``
attribute.

These concrete examples demonstrate how Ray Train appropriately saves checkpoints, model weights but not models, in distributed training.


.. testcode::

    import json
    import os
    import tempfile

    from ray import train
    from ray.train import Checkpoint, ScalingConfig
    from ray.train.tensorflow import TensorflowTrainer

    import numpy as np

    def train_func(config):
        import tensorflow as tf
        n = 100
        # create a toy dataset
        # data   : X - dim = (n, 4)
        # target : Y - dim = (n, 1)
        X = np.random.normal(0, 1, size=(n, 4))
        Y = np.random.uniform(0, 1, size=(n, 1))

        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with strategy.scope():
            # toy neural network : 1-layer
            model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="linear", input_shape=(4,))])
            model.compile(optimizer="Adam", loss="mean_squared_error", metrics=["mse"])

        for epoch in range(config["num_epochs"]):
            history = model.fit(X, Y, batch_size=20)

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                model.save(os.path.join(temp_checkpoint_dir, "model.keras"))
                checkpoint_dict = os.path.join(temp_checkpoint_dir, "checkpoint.json")
                with open(checkpoint_dict, "w") as f:
                    json.dump({"epoch": epoch}, f)
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                train.report({"loss": history.history["loss"][0]}, checkpoint=checkpoint)

    trainer = TensorflowTrainer(
        train_func,
        train_loop_config={"num_epochs": 5},
        scaling_config=ScalingConfig(num_workers=2),
    )
    result = trainer.fit()
    print(result.checkpoint)

By default, checkpoints persist to local disk in the :ref:`log
directory <train-log-dir>` of each run.

Load checkpoints
~~~~~~~~~~~~~~~~

.. testcode::

    import os
    import tempfile

    from ray import train
    from ray.train import Checkpoint, ScalingConfig
    from ray.train.tensorflow import TensorflowTrainer

    import numpy as np

    def train_func(config):
        import tensorflow as tf
        n = 100
        # create a toy dataset
        # data   : X - dim = (n, 4)
        # target : Y - dim = (n, 1)
        X = np.random.normal(0, 1, size=(n, 4))
        Y = np.random.uniform(0, 1, size=(n, 1))

        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with strategy.scope():
            # toy neural network : 1-layer
            checkpoint = train.get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    model = tf.keras.models.load_model(
                        os.path.join(checkpoint_dir, "model.keras")
                    )
            else:
                model = tf.keras.Sequential(
                    [tf.keras.layers.Dense(1, activation="linear", input_shape=(4,))]
                )
            model.compile(optimizer="Adam", loss="mean_squared_error", metrics=["mse"])

        for epoch in range(config["num_epochs"]):
            history = model.fit(X, Y, batch_size=20)

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                model.save(os.path.join(temp_checkpoint_dir, "model.keras"))
                extra_json = os.path.join(temp_checkpoint_dir, "checkpoint.json")
                with open(extra_json, "w") as f:
                    json.dump({"epoch": epoch}, f)
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                train.report({"loss": history.history["loss"][0]}, checkpoint=checkpoint)

    trainer = TensorflowTrainer(
        train_func,
        train_loop_config={"num_epochs": 5},
        scaling_config=ScalingConfig(num_workers=2),
    )
    result = trainer.fit()
    print(result.checkpoint)

    # Start a new run from a loaded checkpoint
    trainer = TensorflowTrainer(
        train_func,
        train_loop_config={"num_epochs": 5},
        scaling_config=ScalingConfig(num_workers=2),
        resume_from_checkpoint=result.checkpoint,
    )
    result = trainer.fit()


Further reading
---------------
See :ref:`User Guides <train-user-guides>` to explore more topics:

- :ref:`Experiment tracking <train-experiment-tracking-native>`
- :ref:`Fault tolerance and training on spot instances <train-fault-tolerance>`
- :ref:`Hyperparameter optimization <train-tune>`


.. _train-pytorch-transformers:

Get Started with Distributed Training using Hugging Face Transformers
=====================================================================

This tutorial walks through the process of converting an existing Hugging Face Transformers script to use Ray Train.

Learn how to:

1. Configure a :ref:`training function <train-overview-training-function>` to report metrics and save checkpoints.
2. Configure :ref:`scaling <train-overview-scaling-config>` and CPU or GPU resource requirements for your training job.
3. Launch your distributed training job with a :class:`~ray.train.torch.TorchTrainer`.

Quickstart
----------

For reference, the final code follows:

.. testcode::
    :skipif: True

    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    def train_func():
        # Your Transformers training code here.

    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()

1. `train_func` is the Python code that executes on each distributed training worker.
2. :class:`~ray.train.ScalingConfig` defines the number of distributed training workers and whether to use GPUs.
3. :class:`~ray.train.torch.TorchTrainer` launches the distributed training job.

Compare a Hugging Face Transformers training script with and without Ray Train.

.. tab-set::

    .. tab-item:: Hugging Face Transformers

        .. This snippet isn't tested because it doesn't use any Ray code.

        .. testcode::
            :skipif: True

            # Adapted from Hugging Face tutorial: https://huggingface.co/docs/transformers/training

            import numpy as np
            import evaluate
            from datasets import load_dataset
            from transformers import (
                Trainer,
                TrainingArguments,
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )

            # Datasets
            dataset = load_dataset("yelp_review_full")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True)

            small_train_dataset = dataset["train"].select(range(1000)).map(tokenize_function, batched=True)
            small_eval_dataset = dataset["test"].select(range(1000)).map(tokenize_function, batched=True)

            # Model
            model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-cased", num_labels=5
            )

            # Metrics
            metric = evaluate.load("accuracy")

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                return metric.compute(predictions=predictions, references=labels)

            # Hugging Face Trainer
            training_args = TrainingArguments(
                output_dir="test_trainer", evaluation_strategy="epoch", report_to="none"
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=small_train_dataset,
                eval_dataset=small_eval_dataset,
                compute_metrics=compute_metrics,
            )

            # Start Training
            trainer.train()



    .. tab-item:: Hugging Face Transformers + Ray Train

        .. code-block:: python
            :emphasize-lines: 13-15, 21, 67-68, 72, 80-87

            import os

            import numpy as np
            import evaluate
            from datasets import load_dataset
            from transformers import (
                Trainer,
                TrainingArguments,
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )

            import ray.train.huggingface.transformers
            from ray.train import ScalingConfig
            from ray.train.torch import TorchTrainer


            # [1] Encapsulate data preprocessing, training, and evaluation
            # logic in a training function
            # ============================================================
            def train_func():
                # Datasets
                dataset = load_dataset("yelp_review_full")
                tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

                def tokenize_function(examples):
                    return tokenizer(examples["text"], padding="max_length", truncation=True)

                small_train_dataset = (
                    dataset["train"].select(range(1000)).map(tokenize_function, batched=True)
                )
                small_eval_dataset = (
                    dataset["test"].select(range(1000)).map(tokenize_function, batched=True)
                )

                # Model
                model = AutoModelForSequenceClassification.from_pretrained(
                    "bert-base-cased", num_labels=5
                )

                # Evaluation Metrics
                metric = evaluate.load("accuracy")

                def compute_metrics(eval_pred):
                    logits, labels = eval_pred
                    predictions = np.argmax(logits, axis=-1)
                    return metric.compute(predictions=predictions, references=labels)

                # Hugging Face Trainer
                training_args = TrainingArguments(
                    output_dir="test_trainer",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    report_to="none",
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=small_train_dataset,
                    eval_dataset=small_eval_dataset,
                    compute_metrics=compute_metrics,
                )

                # [2] Report Metrics and Checkpoints to Ray Train
                # ===============================================
                callback = ray.train.huggingface.transformers.RayTrainReportCallback()
                trainer.add_callback(callback)

                # [3] Prepare Transformers Trainer
                # ================================
                trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

                # Start Training
                trainer.train()


            # [4] Define a Ray TorchTrainer to launch `train_func` on all workers
            # ===================================================================
            ray_trainer = TorchTrainer(
                train_func,
                scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
                # [4a] If running in a multi-node cluster, this is where you
                # should configure the run's persistent storage that is accessible
                # across all worker nodes.
                # run_config=ray.train.RunConfig(storage_path="s3://..."),
            )
            result: ray.train.Result = ray_trainer.fit()

            # [5] Load the trained model.
            with result.checkpoint.as_directory() as checkpoint_dir:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    ray.train.huggingface.transformers.RayTrainReportCallback.CHECKPOINT_NAME,
                )
                model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)


Set up a training function
--------------------------

.. include:: ./common/torch-configure-train_func.rst

Ray Train sets up the distributed process group on each worker before entering this function. 
Put all the logic into this function, including dataset construction and preprocessing,
model initialization, transformers trainer definition and more.

.. note::

    If you are using Hugging Face Datasets or Evaluate, make sure to call ``datasets.load_dataset`` and ``evaluate.load``
    inside the training function. Don't pass the loaded datasets and metrics from outside of the training
    function, because it might cause serialization errors while transferring the objects to the workers.


Report checkpoints and metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To persist your checkpoints and monitor training progress, add a
:class:`ray.train.huggingface.transformers.RayTrainReportCallback` utility callback to your Trainer.


.. code-block:: diff

     import transformers
     from ray.train.huggingface.transformers import RayTrainReportCallback

     def train_func():
         ...
         trainer = transformers.Trainer(...)
    +    trainer.add_callback(RayTrainReportCallback())
         ...


Reporting metrics and checkpoints to Ray Train ensures that you can use Ray Tune and :ref:`fault-tolerant training <train-fault-tolerance>`.
Note that the :class:`ray.train.huggingface.transformers.RayTrainReportCallback` only provides a simple implementation, and you can :ref:`further customize <train-dl-saving-checkpoints>` it.


Prepare a Transformers Trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, pass your Transformers Trainer into
:meth:`~ray.train.huggingface.transformers.prepare_trainer` to validate
your configurations and enable Ray Data Integration.


.. code-block:: diff

     import transformers
     import ray.train.huggingface.transformers

     def train_func():
         ...
         trainer = transformers.Trainer(...)
    +    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
         trainer.train()
         ...


.. include:: ./common/torch-configure-run.rst


Next steps
----------

After you have converted your Hugging Face Transformers training script to use Ray Train:

* See :ref:`User Guides <train-user-guides>` to learn more about how to perform specific tasks.
* Browse the :doc:`Examples <examples>` for end-to-end examples of how to use Ray Train.
* Dive into the :ref:`API Reference <train-api>` for more details on the classes and methods used in this tutorial.


.. _transformers-trainer-migration-guide:

TransformersTrainer Migration Guide
-----------------------------------

Ray 2.1 introduced the `TransformersTrainer`, which exposes a `trainer_init_per_worker` interface
to define `transformers.Trainer`, then runs a pre-defined training function in a black box.

Ray 2.7 introduced the newly unified :class:`~ray.train.torch.TorchTrainer` API,
which offers enhanced transparency, flexibility, and simplicity. This API aligns more
with standard Hugging Face Transformers scripts, ensuring that you have better control over your
native Transformers training code.


.. tab-set::

    .. tab-item:: (Deprecating) TransformersTrainer

        .. This snippet isn't tested because it contains skeleton code.

        .. testcode::
            :skipif: True

            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM
            from datasets import load_dataset

            import ray
            from ray.train.huggingface import TransformersTrainer
            from ray.train import ScalingConfig

            # Dataset
            def preprocess(examples):
                ...

            hf_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
            processed_ds = hf_datasets.map(preprocess, ...)

            ray_train_ds = ray.data.from_huggingface(processed_ds["train"])
            ray_eval_ds = ray.data.from_huggingface(processed_ds["validation"])

            # Define the Trainer generation function
            def trainer_init_per_worker(train_dataset, eval_dataset, **config):
                MODEL_NAME = "gpt2"
                model_config = AutoConfig.from_pretrained(MODEL_NAME)
                model = AutoModelForCausalLM.from_config(model_config)
                args = transformers.TrainingArguments(
                    output_dir=f"{MODEL_NAME}-wikitext2",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    logging_strategy="epoch",
                    learning_rate=2e-5,
                    weight_decay=0.01,
                    max_steps=100,
                )
                return transformers.Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                )

            # Build a Ray TransformersTrainer
            scaling_config = ScalingConfig(num_workers=4, use_gpu=True)
            ray_trainer = TransformersTrainer(
                trainer_init_per_worker=trainer_init_per_worker,
                scaling_config=scaling_config,
                datasets={"train": ray_train_ds, "evaluation": ray_eval_ds},
            )
            result = ray_trainer.fit()


    .. tab-item:: (New API) TorchTrainer

        .. This snippet isn't tested because it contains skeleton code.

        .. testcode::
            :skipif: True

            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM
            from datasets import load_dataset

            import ray
            from ray.train.huggingface.transformers import (
                RayTrainReportCallback,
                prepare_trainer,
            )
            from ray.train import ScalingConfig

            # Dataset
            def preprocess(examples):
                ...

            hf_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
            processed_ds = hf_datasets.map(preprocess, ...)

            ray_train_ds = ray.data.from_huggingface(processed_ds["train"])
            ray_eval_ds = ray.data.from_huggingface(processed_ds["evaluation"])

            # [1] Define the full training function
            # =====================================
            def train_func():
                MODEL_NAME = "gpt2"
                model_config = AutoConfig.from_pretrained(MODEL_NAME)
                model = AutoModelForCausalLM.from_config(model_config)

                # [2] Build Ray Data iterables
                # ============================
                train_dataset = ray.train.get_dataset_shard("train")
                eval_dataset = ray.train.get_dataset_shard("evaluation")

                train_iterable_ds = train_dataset.iter_torch_batches(batch_size=8)
                eval_iterable_ds = eval_dataset.iter_torch_batches(batch_size=8)

                args = transformers.TrainingArguments(
                    output_dir=f"{MODEL_NAME}-wikitext2",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    logging_strategy="epoch",
                    learning_rate=2e-5,
                    weight_decay=0.01,
                    max_steps=100,
                )

                trainer = transformers.Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_iterable_ds,
                    eval_dataset=eval_iterable_ds,
                )

                # [3] Inject Ray Train Report Callback
                # ====================================
                trainer.add_callback(RayTrainReportCallback())

                # [4] Prepare your trainer
                # ========================
                trainer = prepare_trainer(trainer)
                trainer.train()

            # Build a Ray TorchTrainer
            scaling_config = ScalingConfig(num_workers=4, use_gpu=True)
            ray_trainer = TorchTrainer(
                train_func,
                scaling_config=scaling_config,
                datasets={"train": ray_train_ds, "evaluation": ray_eval_ds},
            )
            result = ray_trainer.fit()


.. _train-pytorch-lightning:

Get Started with Distributed Training using PyTorch Lightning
=============================================================

This tutorial walks through the process of converting an existing PyTorch Lightning script to use Ray Train.

Learn how to:

1. Configure the Lightning Trainer so that it runs distributed with Ray and on the correct CPU or GPU device.
2. Configure :ref:`training function <train-overview-training-function>` to report metrics and save checkpoints.
3. Configure :ref:`scaling <train-overview-scaling-config>` and CPU or GPU resource requirements for a training job.
4. Launch a distributed training job with a :class:`~ray.train.torch.TorchTrainer`.

Quickstart
----------

For reference, the final code is as follows:

.. testcode::
    :skipif: True

    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    def train_func():
        # Your PyTorch Lightning training code here.

    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()

1. `train_func` is the Python code that executes on each distributed training worker.
2. :class:`~ray.train.ScalingConfig` defines the number of distributed training workers and whether to use GPUs.
3. :class:`~ray.train.torch.TorchTrainer` launches the distributed training job.

Compare a PyTorch Lightning training script with and without Ray Train.

.. tab-set::

    .. tab-item:: PyTorch Lightning

        .. This snippet isn't tested because it doesn't use any Ray code.

        .. testcode::
            :skipif: True

            import torch
            from torchvision.models import resnet18
            from torchvision.datasets import FashionMNIST
            from torchvision.transforms import ToTensor, Normalize, Compose
            from torch.utils.data import DataLoader
            import lightning.pytorch as pl

            # Model, Loss, Optimizer
            class ImageClassifier(pl.LightningModule):
                def __init__(self):
                    super(ImageClassifier, self).__init__()
                    self.model = resnet18(num_classes=10)
                    self.model.conv1 = torch.nn.Conv2d(
                        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                    )
                    self.criterion = torch.nn.CrossEntropyLoss()

                def forward(self, x):
                    return self.model(x)

                def training_step(self, batch, batch_idx):
                    x, y = batch
                    outputs = self.forward(x)
                    loss = self.criterion(outputs, y)
                    self.log("loss", loss, on_step=True, prog_bar=True)
                    return loss

                def configure_optimizers(self):
                    return torch.optim.Adam(self.model.parameters(), lr=0.001)

            # Data
            transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
            train_data = FashionMNIST(root='./data', train=True, download=True, transform=transform)
            train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

            # Training
            model = ImageClassifier()
            trainer = pl.Trainer(max_epochs=10)
            trainer.fit(model, train_dataloaders=train_dataloader)


    .. tab-item:: PyTorch Lightning + Ray Train

        .. code-block:: python
            :emphasize-lines: 11-12, 38, 52-57, 59, 63, 66-73

            import os
            import tempfile

            import torch
            from torch.utils.data import DataLoader
            from torchvision.models import resnet18
            from torchvision.datasets import FashionMNIST
            from torchvision.transforms import ToTensor, Normalize, Compose
            import lightning.pytorch as pl

            import ray.train.lightning
            from ray.train.torch import TorchTrainer

            # Model, Loss, Optimizer
            class ImageClassifier(pl.LightningModule):
                def __init__(self):
                    super(ImageClassifier, self).__init__()
                    self.model = resnet18(num_classes=10)
                    self.model.conv1 = torch.nn.Conv2d(
                        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                    )
                    self.criterion = torch.nn.CrossEntropyLoss()

                def forward(self, x):
                    return self.model(x)

                def training_step(self, batch, batch_idx):
                    x, y = batch
                    outputs = self.forward(x)
                    loss = self.criterion(outputs, y)
                    self.log("loss", loss, on_step=True, prog_bar=True)
                    return loss

                def configure_optimizers(self):
                    return torch.optim.Adam(self.model.parameters(), lr=0.001)


            def train_func():
                # Data
                transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
                data_dir = os.path.join(tempfile.gettempdir(), "data")
                train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
                train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

                # Training
                model = ImageClassifier()
                # [1] Configure PyTorch Lightning Trainer.
                trainer = pl.Trainer(
                    max_epochs=10,
                    devices="auto",
                    accelerator="auto",
                    strategy=ray.train.lightning.RayDDPStrategy(),
                    plugins=[ray.train.lightning.RayLightningEnvironment()],
                    callbacks=[ray.train.lightning.RayTrainReportCallback()],
                    # [1a] Optionally, disable the default checkpointing behavior
                    # in favor of the `RayTrainReportCallback` above.
                    enable_checkpointing=False,
                )
                trainer = ray.train.lightning.prepare_trainer(trainer)
                trainer.fit(model, train_dataloaders=train_dataloader)

            # [2] Configure scaling and resource requirements.
            scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=True)

            # [3] Launch distributed training job.
            trainer = TorchTrainer(
                train_func,
                scaling_config=scaling_config,
                # [3a] If running in a multi-node cluster, this is where you
                # should configure the run's persistent storage that is accessible
                # across all worker nodes.
                # run_config=ray.train.RunConfig(storage_path="s3://..."),
            )
            result: ray.train.Result = trainer.fit()

            # [4] Load the trained model.
            with result.checkpoint.as_directory() as checkpoint_dir:
                model = ImageClassifier.load_from_checkpoint(
                    os.path.join(
                        checkpoint_dir,
                        ray.train.lightning.RayTrainReportCallback.CHECKPOINT_NAME,
                    ),
                )


Set up a training function
--------------------------

.. include:: ./common/torch-configure-train_func.rst

Ray Train sets up your distributed process group on each worker. You only need to
make a few changes to your Lightning Trainer definition.

.. code-block:: diff

     import lightning.pytorch as pl
    -from pl.strategies import DDPStrategy
    -from pl.plugins.environments import LightningEnvironment
    +import ray.train.lightning

     def train_func():
         ...
         model = MyLightningModule(...)
         datamodule = MyLightningDataModule(...)

         trainer = pl.Trainer(
    -        devices=[0, 1, 2, 3],
    -        strategy=DDPStrategy(),
    -        plugins=[LightningEnvironment()],
    +        devices="auto",
    +        accelerator="auto",
    +        strategy=ray.train.lightning.RayDDPStrategy(),
    +        plugins=[ray.train.lightning.RayLightningEnvironment()]
         )
    +    trainer = ray.train.lightning.prepare_trainer(trainer)

         trainer.fit(model, datamodule=datamodule)

The following sections discuss each change.

Configure the distributed strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ray Train offers several sub-classed distributed strategies for Lightning.
These strategies retain the same argument list as their base strategy classes.
Internally, they configure the root device and the distributed
sampler arguments.

- :class:`~ray.train.lightning.RayDDPStrategy`
- :class:`~ray.train.lightning.RayFSDPStrategy`
- :class:`~ray.train.lightning.RayDeepSpeedStrategy`


.. code-block:: diff

     import lightning.pytorch as pl
    -from pl.strategies import DDPStrategy
    +import ray.train.lightning

     def train_func():
         ...
         trainer = pl.Trainer(
             ...
    -        strategy=DDPStrategy(),
    +        strategy=ray.train.lightning.RayDDPStrategy(),
             ...
         )
         ...

Configure the Ray cluster environment plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ray Train also provides a :class:`~ray.train.lightning.RayLightningEnvironment` class
as a specification for the Ray Cluster. This utility class configures the worker's
local, global, and node rank and world size.


.. code-block:: diff

     import lightning.pytorch as pl
    -from pl.plugins.environments import LightningEnvironment
    +import ray.train.lightning

     def train_func():
         ...
         trainer = pl.Trainer(
             ...
    -        plugins=[LightningEnvironment()],
    +        plugins=[ray.train.lightning.RayLightningEnvironment()],
             ...
         )
         ...


Configure parallel devices
^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition, Ray TorchTrainer has already configured the correct
``CUDA_VISIBLE_DEVICES`` for you. One should always use all available
GPUs by setting ``devices="auto"`` and ``acelerator="auto"``.


.. code-block:: diff

     import lightning.pytorch as pl

     def train_func():
         ...
         trainer = pl.Trainer(
             ...
    -        devices=[0,1,2,3],
    +        devices="auto",
    +        accelerator="auto",
             ...
         )
         ...



Report checkpoints and metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To persist your checkpoints and monitor training progress, add a
:class:`ray.train.lightning.RayTrainReportCallback` utility callback to your Trainer.


.. code-block:: diff

     import lightning.pytorch as pl
     from ray.train.lightning import RayTrainReportCallback

     def train_func():
         ...
         trainer = pl.Trainer(
             ...
    -        callbacks=[...],
    +        callbacks=[..., RayTrainReportCallback()],
         )
         ...


Reporting metrics and checkpoints to Ray Train enables you to support :ref:`fault-tolerant training <train-fault-tolerance>` and :ref:`hyperparameter optimization <train-tune>`.
Note that the :class:`ray.train.lightning.RayTrainReportCallback` class only provides a simple implementation, and can be :ref:`further customized <train-dl-saving-checkpoints>`.

Prepare your Lightning Trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, pass your Lightning Trainer into
:meth:`~ray.train.lightning.prepare_trainer` to validate
your configurations.


.. code-block:: diff

     import lightning.pytorch as pl
     import ray.train.lightning

     def train_func():
         ...
         trainer = pl.Trainer(...)
    +    trainer = ray.train.lightning.prepare_trainer(trainer)
         ...


.. include:: ./common/torch-configure-run.rst


Next steps
----------

After you have converted your PyTorch Lightning training script to use Ray Train:

* See :ref:`User Guides <train-user-guides>` to learn more about how to perform specific tasks.
* Browse the :doc:`Examples <examples>` for end-to-end examples of how to use Ray Train.
* Consult the :ref:`API Reference <train-api>` for more details on the classes and methods from this tutorial.

Version Compatibility
---------------------

Ray Train is tested with `pytorch_lightning` versions `1.6.5` and `2.1.2`. For full compatibility, use ``pytorch_lightning>=1.6.5`` .
Earlier versions aren't prohibited but may result in unexpected issues. If you run into any compatibility issues, consider upgrading your PyTorch Lightning version or
`file an issue <https://github.com/ray-project/ray/issues>`_.

.. note::

    If you are using Lightning 2.x, please use the import path `lightning.pytorch.xxx` instead of `pytorch_lightning.xxx`.

.. _lightning-trainer-migration-guide:

LightningTrainer Migration Guide
--------------------------------

Ray 2.4 introduced the `LightningTrainer`, and exposed a
`LightningConfigBuilder` to define configurations for `pl.LightningModule`
and `pl.Trainer`.

It then instantiates the model and trainer objects and runs a pre-defined
training function in a black box.

This version of the LightningTrainer API was constraining and limited
your ability to manage the training functionality.

Ray 2.7 introduced the newly unified :class:`~ray.train.torch.TorchTrainer` API, which offers
enhanced transparency, flexibility, and simplicity. This API is more aligned
with standard PyTorch Lightning scripts, ensuring users have better
control over their native Lightning code.


.. tab-set::

    .. tab-item:: (Deprecating) LightningTrainer

        .. This snippet isn't tested because it raises a hard deprecation warning.

        .. testcode::
            :skipif: True

            from ray.train.lightning import LightningConfigBuilder, LightningTrainer

            config_builder = LightningConfigBuilder()
            # [1] Collect model configs
            config_builder.module(cls=MyLightningModule, lr=1e-3, feature_dim=128)

            # [2] Collect checkpointing configs
            config_builder.checkpointing(monitor="val_accuracy", mode="max", save_top_k=3)

            # [3] Collect pl.Trainer configs
            config_builder.trainer(
                max_epochs=10,
                accelerator="gpu",
                log_every_n_steps=100,
            )

            # [4] Build datasets on the head node
            datamodule = MyLightningDataModule(batch_size=32)
            config_builder.fit_params(datamodule=datamodule)

            # [5] Execute the internal training function in a black box
            ray_trainer = LightningTrainer(
                lightning_config=config_builder.build(),
                scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
                run_config=RunConfig(
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=3,
                        checkpoint_score_attribute="val_accuracy",
                        checkpoint_score_order="max",
                    ),
                )
            )
            result = ray_trainer.fit()

            # [6] Load the trained model from an opaque Lightning-specific checkpoint.
            lightning_checkpoint = result.checkpoint
            model = lightning_checkpoint.get_model(MyLightningModule)



    .. tab-item:: (New API) TorchTrainer

        .. This snippet isn't tested because it runs with 4 GPUs, and CI is only run with 1.

        .. testcode::
            :skipif: True

            import os

            import lightning.pytorch as pl

            import ray.train
            from ray.train.torch import TorchTrainer
            from ray.train.lightning import (
                RayDDPStrategy,
                RayLightningEnvironment,
                RayTrainReportCallback,
                prepare_trainer
            )

            def train_func():
                # [1] Create a Lightning model
                model = MyLightningModule(lr=1e-3, feature_dim=128)

                # [2] Report Checkpoint with callback
                ckpt_report_callback = RayTrainReportCallback()

                # [3] Create a Lighting Trainer
                trainer = pl.Trainer(
                    max_epochs=10,
                    log_every_n_steps=100,
                    # New configurations below
                    devices="auto",
                    accelerator="auto",
                    strategy=RayDDPStrategy(),
                    plugins=[RayLightningEnvironment()],
                    callbacks=[ckpt_report_callback],
                )

                # Validate your Lightning trainer configuration
                trainer = prepare_trainer(trainer)

                # [4] Build your datasets on each worker
                datamodule = MyLightningDataModule(batch_size=32)
                trainer.fit(model, datamodule=datamodule)

            # [5] Explicitly define and run the training function
            ray_trainer = TorchTrainer(
                train_func,
                scaling_config=ray.train.ScalingConfig(num_workers=4, use_gpu=True),
                run_config=ray.train.RunConfig(
                    checkpoint_config=ray.train.CheckpointConfig(
                        num_to_keep=3,
                        checkpoint_score_attribute="val_accuracy",
                        checkpoint_score_order="max",
                    ),
                )
            )
            result = ray_trainer.fit()

            # [6] Load the trained model from a simplified checkpoint interface.
            checkpoint: ray.train.Checkpoint = result.checkpoint
            with checkpoint.as_directory() as checkpoint_dir:
                print("Checkpoint contents:", os.listdir(checkpoint_dir))
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.ckpt")
                model = MyLightningModule.load_from_checkpoint(checkpoint_path)


.. _train-hf-accelerate:

Get Started with Distributed Training using Hugging Face Accelerate
===================================================================

The :class:`~ray.train.torch.TorchTrainer` can help you easily launch your `Accelerate <https://huggingface.co/docs/accelerate>`_  training across a distributed Ray cluster.

You only need to run your existing training code with a TorchTrainer. You can expect the final code to look like this:

.. testcode::
    :skipif: True

    from accelerate import Accelerator

    def train_func():
        # Instantiate the accelerator
        accelerator = Accelerator(...)

        model = ...
        optimizer = ...
        train_dataloader = ...
        eval_dataloader = ...
        lr_scheduler = ...

        # Prepare everything for distributed training
        (
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # Start training
        ...

    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(...),
        # If running in a multi-node cluster, this is where you
        # should configure the run's persistent storage that is accessible
        # across all worker nodes.
        # run_config=ray.train.RunConfig(storage_path="s3://..."),
        ...
    )
    trainer.fit()

.. tip::

    Model and data preparation for distributed training is completely handled by the `Accelerator <https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator>`_
    object and its `Accelerator.prepare() <https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.prepare>`_  method.

    Unlike with native PyTorch, **don't** call any additional Ray Train utilities
    like :meth:`~ray.train.torch.prepare_model` or :meth:`~ray.train.torch.prepare_data_loader` in your training function.

Configure Accelerate
--------------------

In Ray Train, you can set configurations through the `accelerate.Accelerator <https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator>`_
object in your training function. Below are starter examples for configuring Accelerate.

.. tab-set::

    .. tab-item:: DeepSpeed

        For example, to run DeepSpeed with Accelerate, create a `DeepSpeedPlugin <https://huggingface.co/docs/accelerate/main/en/package_reference/deepspeed>`_
        from a dictionary:

        .. testcode::
            :skipif: True

            from accelerate import Accelerator, DeepSpeedPlugin

            DEEPSPEED_CONFIG = {
                "fp16": {
                    "enabled": True
                },
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": False
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": "auto",
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto",
                    "gather_16bit_weights_on_model_save": True,
                    "round_robin_gradients": True
                },
                "gradient_accumulation_steps": "auto",
                "gradient_clipping": "auto",
                "steps_per_print": 10,
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "wall_clock_breakdown": False
            }

            def train_func():
                # Create a DeepSpeedPlugin from config dict
                ds_plugin = DeepSpeedPlugin(hf_ds_config=DEEPSPEED_CONFIG)

                # Initialize Accelerator
                accelerator = Accelerator(
                    ...,
                    deepspeed_plugin=ds_plugin,
                )

                # Start training
                ...

            from ray.train.torch import TorchTrainer
            from ray.train import ScalingConfig

            trainer = TorchTrainer(
                train_func,
                scaling_config=ScalingConfig(...),
                run_config=ray.train.RunConfig(storage_path="s3://..."),
                ...
            )
            trainer.fit()

    .. tab-item:: FSDP
        :sync: FSDP

        For PyTorch FSDP, create a `FullyShardedDataParallelPlugin <https://huggingface.co/docs/accelerate/main/en/package_reference/fsdp>`_
        and pass it to the Accelerator.

        .. testcode::
            :skipif: True

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
            from accelerate import Accelerator, FullyShardedDataParallelPlugin

            def train_func():
                fsdp_plugin = FullyShardedDataParallelPlugin(
                    state_dict_config=FullStateDictConfig(
                        offload_to_cpu=False,
                        rank0_only=False
                    ),
                    optim_state_dict_config=FullOptimStateDictConfig(
                        offload_to_cpu=False,
                        rank0_only=False
                    )
                )

                # Initialize accelerator
                accelerator = Accelerator(
                    ...,
                    fsdp_plugin=fsdp_plugin,
                )

                # Start training
                ...

            from ray.train.torch import TorchTrainer
            from ray.train import ScalingConfig

            trainer = TorchTrainer(
                train_func,
                scaling_config=ScalingConfig(...),
                run_config=ray.train.RunConfig(storage_path="s3://..."),
                ...
            )
            trainer.fit()

Note that Accelerate also provides a CLI tool, `"accelerate config"`, to generate a configuration and launch your training
job with `"accelerate launch"`. However, it's not necessary here because Ray's `TorchTrainer` already sets up the Torch
distributed environment and launches the training function on all workers.


Next, see these end-to-end examples below for more details:

.. tab-set::

    .. tab-item:: Example with Ray Data

        .. dropdown:: Show Code

            .. literalinclude:: /../../python/ray/train/examples/accelerate/accelerate_torch_trainer.py
                :language: python
                :start-after: __accelerate_torch_basic_example_start__
                :end-before: __accelerate_torch_basic_example_end__

    .. tab-item:: Example with PyTorch DataLoader

        .. dropdown:: Show Code

            .. literalinclude:: /../../python/ray/train/examples/accelerate/accelerate_torch_trainer_no_raydata.py
                :language: python
                :start-after: __accelerate_torch_basic_example_no_raydata_start__
                :end-before: __accelerate_torch_basic_example_no_raydata_end__

.. seealso::

    If you're looking for more advanced use cases, check out this Llama-2 fine-tuning example:

    - `Fine-tuning Llama-2 series models with Deepspeed, Accelerate, and Ray Train. <https://github.com/ray-project/ray/tree/master/doc/source/templates/04_finetuning_llms_with_deepspeed>`_

You may also find these user guides helpful:

- :ref:`Configuring Scale and GPUs <train_scaling_config>`
- :ref:`Configuration and Persistent Storage <train-run-config>`
- :ref:`Saving and Loading Checkpoints <train-checkpointing>`
- :ref:`How to use Ray Data with Ray Train <data-ingest-torch>`


AccelerateTrainer Migration Guide
---------------------------------

Before Ray 2.7, Ray Train's `AccelerateTrainer` API was the
recommended way to run Accelerate code. As a subclass of :class:`TorchTrainer <ray.train.torch.TorchTrainer>`,
the AccelerateTrainer takes in a configuration file generated by ``accelerate config`` and applies it to all workers.
Aside from that, the functionality of ``AccelerateTrainer`` is identical to ``TorchTrainer``.

However, this caused confusion around whether this was the *only* way to run Accelerate code.
Because you can express the full Accelerate functionality with the ``Accelerator`` and ``TorchTrainer`` combination, the plan is to deprecate the ``AccelerateTrainer`` in Ray 2.8,
and it's recommend to run your  Accelerate code directly with ``TorchTrainer``.


.. _train-key-concepts:

.. _train-overview:

Ray Train Overview
==================


To use Ray Train effectively, you need to understand four main concepts:

#. :ref:`Training function <train-overview-training-function>`: A Python function that contains your model training logic.
#. :ref:`Worker <train-overview-worker>`: A process that runs the training function.
#. :ref:`Scaling configuration: <train-overview-scaling-config>` A configuration of the number of workers and compute resources (for example, CPUs or GPUs).
#. :ref:`Trainer <train-overview-trainers>`: A Python class that ties together the training function, workers, and scaling configuration to execute a distributed training job.

.. figure:: images/overview.png
    :align: center

.. _train-overview-training-function:

Training function
-----------------

The training function is a user-defined Python function that contains the end-to-end model training loop logic.
When launching a distributed training job, each worker executes this training function.

Ray Train documentation uses the following conventions:

#. `train_func` is a user-defined function that contains the training code.
#. `train_func` is passed into the Trainer's `train_loop_per_worker` parameter.

.. testcode::

    def train_func():
        """User-defined training function that runs on each distributed worker process.

        This function typically contains logic for loading the model,
        loading the dataset, training the model, saving checkpoints,
        and logging metrics.
        """
        ...

.. _train-overview-worker:

Worker
------

Ray Train distributes model training compute to individual worker processes across the cluster.
Each worker is a process that executes the `train_func`.
The number of workers determines the parallelism of the training job and is configured in the :class:`~ray.train.ScalingConfig`.

.. _train-overview-scaling-config:

Scaling configuration
---------------------

The :class:`~ray.train.ScalingConfig` is the mechanism for defining the scale of the training job.
Specify two basic parameters for worker parallelism and compute resources:

* :class:`num_workers <ray.train.ScalingConfig>`: The number of workers to launch for a distributed training job.
* :class:`use_gpu <ray.train.ScalingConfig>`: Whether each worker should use a GPU or CPU.

.. testcode::

    from ray.train import ScalingConfig

    # Single worker with a CPU
    scaling_config = ScalingConfig(num_workers=1, use_gpu=False)

    # Single worker with a GPU
    scaling_config = ScalingConfig(num_workers=1, use_gpu=True)

    # Multiple workers, each with a GPU
    scaling_config = ScalingConfig(num_workers=4, use_gpu=True)

.. _train-overview-trainers:

Trainer
-------

The Trainer ties the previous three concepts together to launch distributed training jobs.
Ray Train provides :ref:`Trainer classes <train-api>` for different frameworks.
Calling the :meth:`fit() <ray.train.trainer.BaseTrainer.fit>` method executes the training job by:

#. Launching workers as defined by the :ref:`scaling_config <train-overview-scaling-config>`.
#. Setting up the framework's distributed environment on all workers.
#. Running the `train_func` on all workers.

.. testcode::
    :hide:

    def train_func():
        pass

    scaling_config = ScalingConfig(num_workers=1, use_gpu=False)

.. testcode::

    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    trainer.fit()


.. _train-user-guides:

Ray Train User Guides
=====================

.. toctree::
    :maxdepth: 2

    user-guides/data-loading-preprocessing
    user-guides/using-gpus
    user-guides/persistent-storage
    user-guides/monitoring-logging
    user-guides/checkpoints
    user-guides/experiment-tracking
    user-guides/results
    user-guides/fault-tolerance
    user-guides/reproducibility
    Hyperparameter Optimization <user-guides/hyperparameter-optimization>


.. _train-more-frameworks:

More Frameworks
===============

.. toctree::
    :hidden:

    Hugging Face Accelerate Guide <huggingface-accelerate>
    DeepSpeed Guide <deepspeed>
    TensorFlow and Keras Guide <distributed-tensorflow-keras>
    XGBoost and LightGBM Guide <distributed-xgboost-lightgbm>
    Horovod Guide <horovod>

.. grid:: 1 2 3 4
    :gutter: 1
    :class-container: container pb-3

    .. grid-item-card::
        :img-top: /images/accelerate_logo.png
        :class-img-top: mt-2 w-75 d-block mx-auto fixed-height-img
        :link: huggingface-accelerate
        :link-type: doc

        Hugging Face Accelerate

    .. grid-item-card::
        :img-top: /images/deepspeed_logo.svg
        :class-img-top: mt-2 w-75 d-block mx-auto fixed-height-img
        :link: deepspeed
        :link-type: doc

        DeepSpeed

    .. grid-item-card::
        :img-top: /images/tf_logo.png
        :class-img-top: mt-2 w-75 d-block mx-auto fixed-height-img
        :link: distributed-tensorflow-keras
        :link-type: doc

        TensorFlow and Keras

    .. grid-item-card::
        :img-top: /images/xgboost_logo.png
        :class-img-top: mt-2 w-75 d-block mx-auto fixed-height-img
        :link: distributed-xgboost-lightgbm
        :link-type: doc

        XGBoost and LightGBM

    .. grid-item-card::
        :img-top: /images/horovod.png
        :class-img-top: mt-2 w-75 d-block mx-auto fixed-height-img
        :link: horovod
        :link-type: doc


        Horovod



.. _train-horovod:

Get Started with Distributed Training using Horovod
===================================================

Ray Train configures the Horovod environment and Rendezvous
server for you, allowing you to run your ``DistributedOptimizer`` training
script. See the `Horovod documentation <https://horovod.readthedocs.io/en/stable/index.html>`_
for more information.

Quickstart
-----------
.. literalinclude:: ./doc_code/hvd_trainer.py
  :language: python



Update your training function
-----------------------------

First, update your :ref:`training function <train-overview-training-function>` to support distributed
training.

If you have a training function that already runs with the `Horovod Ray
Executor <https://horovod.readthedocs.io/en/stable/ray_include.html#horovod-ray-executor>`_,
you shouldn't need to make any additional changes.

To onboard onto Horovod, visit the `Horovod guide
<https://horovod.readthedocs.io/en/stable/index.html#get-started>`_.


Create a HorovodTrainer
-----------------------

``Trainer``\s are the primary Ray Train classes to use to manage state and
execute training. For Horovod, use a :class:`~ray.train.horovod.HorovodTrainer`
that you can setup like this:

.. testcode::
    :hide:

    train_func = lambda: None

.. testcode::

    from ray.train import ScalingConfig
    from ray.train.horovod import HorovodTrainer
    # For GPU Training, set `use_gpu` to True.
    use_gpu = False
    trainer = HorovodTrainer(
        train_func,
        scaling_config=ScalingConfig(use_gpu=use_gpu, num_workers=2)
    )

When training with Horovod, always use a HorovodTrainer,
irrespective of the training framework, for example, PyTorch or TensorFlow.

To customize the backend setup, you can pass a
:class:`~ray.train.horovod.HorovodConfig`:

.. testcode::
    :skipif: True

    from ray.train import ScalingConfig
    from ray.train.horovod import HorovodTrainer, HorovodConfig

    trainer = HorovodTrainer(
        train_func,
        tensorflow_backend=HorovodConfig(...),
        scaling_config=ScalingConfig(num_workers=2),
    )

For more configurability, see the :py:class:`~ray.train.data_parallel_trainer.DataParallelTrainer` API.

Run a training function
-----------------------

With a distributed training function and a Ray Train ``Trainer``, you are now
ready to start training.

.. testcode::
    :skipif: True

    trainer.fit()


Further reading
---------------

Ray Train's :class:`~ray.train.horovod.HorovodTrainer` replaces the distributed
communication backend of the native libraries with its own implementation.
Thus, the remaining integration points remain the same. If you're using Horovod
with :ref:`PyTorch <train-pytorch>` or :ref:`Tensorflow <train-tensorflow-overview>`,
refer to the respective guides for further configuration
and information.

If you are implementing your own Horovod-based training routine without using any of
the training libraries, read through the
:ref:`User Guides <train-user-guides>`, as you can apply much of the content
to generic use cases and adapt them easily.




.. _train-docs:

Ray Train: Scalable Model Training
==================================

.. toctree::
    :hidden:

    Overview <overview>
    PyTorch Guide <getting-started-pytorch>
    PyTorch Lightning Guide <getting-started-pytorch-lightning>
    Hugging Face Transformers Guide <getting-started-transformers>
    more-frameworks
    User Guides <user-guides>
    Examples <examples>
    Benchmarks <benchmarks>
    api/api


.. div:: sd-d-flex-row sd-align-major-center sd-align-minor-center

    .. div:: sd-w-50

        .. raw:: html
           :file: images/logo.svg


Ray Train is a scalable machine learning library for distributed training and fine-tuning.

Ray Train allows you to scale model training code from a single machine to a cluster of machines in the cloud, and abstracts away the complexities of distributed computing.
Whether you have large models or large datasets, Ray Train is the simplest solution for distributed training.

Ray Train provides support for many frameworks:

.. list-table::
   :widths: 1 1
   :header-rows: 1

   * - PyTorch Ecosystem
     - More Frameworks
   * - PyTorch
     - TensorFlow
   * - PyTorch Lightning
     - Keras
   * - Hugging Face Transformers
     - Horovod
   * - Hugging Face Accelerate
     - XGBoost
   * - DeepSpeed
     - LightGBM

Install Ray Train
-----------------

To install Ray Train, run:

.. code-block:: console

    $ pip install -U "ray[train]"

To learn more about installing Ray and its libraries, see
:ref:`Installing Ray <installation>`.

Get started
-----------

.. grid:: 1 2 2 2
    :gutter: 1
    :class-container: container pb-6

    .. grid-item-card::

        **Overview**
        ^^^

        Understand the key concepts for distributed training with Ray Train.

        +++
        .. button-ref:: train-overview
            :color: primary
            :outline:
            :expand:

            Learn the basics

    .. grid-item-card::

        **PyTorch**
        ^^^

        Get started on distributed model training with Ray Train and PyTorch.

        +++
        .. button-ref:: train-pytorch
            :color: primary
            :outline:
            :expand:

            Try Ray Train with PyTorch

    .. grid-item-card::

        **PyTorch Lightning**
        ^^^

        Get started on distributed model training with Ray Train and Lightning.

        +++
        .. button-ref:: train-pytorch-lightning
            :color: primary
            :outline:
            :expand:

            Try Ray Train with Lightning

    .. grid-item-card::

        **Hugging Face Transformers**
        ^^^

        Get started on distributed model training with Ray Train and Transformers.

        +++
        .. button-ref:: train-pytorch-transformers
            :color: primary
            :outline:
            :expand:

            Try Ray Train with Transformers

Learn more
----------

.. grid:: 1 2 2 2
    :gutter: 1
    :class-container: container pb-6

    .. grid-item-card::

        **More Frameworks**
        ^^^

        Don't see your framework? See these guides.

        +++
        .. button-ref:: train-more-frameworks
            :color: primary
            :outline:
            :expand:

            Try Ray Train with other frameworks

    .. grid-item-card::

        **User Guides**
        ^^^

        Get how-to instructions for common training tasks with Ray Train.

        +++
        .. button-ref:: train-user-guides
            :color: primary
            :outline:
            :expand:

            Read how-to guides

    .. grid-item-card::

        **Examples**
        ^^^

        Browse end-to-end code examples for different use cases.

        +++
        .. button-ref:: examples
            :color: primary
            :outline:
            :expand:
            :ref-type: doc

            Learn through examples

    .. grid-item-card::

        **API**
        ^^^

        Consult the API Reference for full descriptions of the Ray Train API.

        +++
        .. button-ref:: train-api
            :color: primary
            :outline:
            :expand:

            Read the API Reference


.. _train-deepspeed:

Get Started with DeepSpeed
==========================

The :class:`~ray.train.torch.TorchTrainer` can help you easily launch your `DeepSpeed <https://www.deepspeed.ai/>`_  training across a distributed Ray cluster.

Code example
------------

You only need to run your existing training code with a TorchTrainer. You can expect the final code to look like this:

.. testcode::
    :skipif: True

    import deepspeed
    from deepspeed.accelerator import get_accelerator

    def train_func():
        # Instantiate your model and dataset
        model = ...
        train_dataset = ...
        eval_dataset = ...
        deepspeed_config = {...} # Your Deepspeed config

        # Prepare everything for distributed training
        model, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=tokenized_datasets["train"],
            collate_fn=collate_fn,
            config=deepspeed_config,
        )

        # Define the GPU device for the current worker
        device = get_accelerator().device_name(model.local_rank)

        # Start training
        ...

    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(...),
        # If running in a multi-node cluster, this is where you
        # should configure the run's persistent storage that is accessible
        # across all worker nodes.
        # run_config=ray.train.RunConfig(storage_path="s3://..."),
        ...
    )
    result = trainer.fit()


Below is a simple example of ZeRO-3 training with DeepSpeed only.

.. tab-set::

    .. tab-item:: Example with Ray Data

        .. dropdown:: Show Code

            .. literalinclude:: /../../python/ray/train/examples/deepspeed/deepspeed_torch_trainer.py
                :language: python
                :start-after: __deepspeed_torch_basic_example_start__
                :end-before: __deepspeed_torch_basic_example_end__

    .. tab-item:: Example with PyTorch DataLoader

        .. dropdown:: Show Code

            .. literalinclude:: /../../python/ray/train/examples/deepspeed/deepspeed_torch_trainer_no_raydata.py
                :language: python
                :start-after: __deepspeed_torch_basic_example_no_raydata_start__
                :end-before: __deepspeed_torch_basic_example_no_raydata_end__

.. tip::

    To run DeepSpeed with pure PyTorch, you **don't need to** provide any additional Ray Train utilities
    like :meth:`~ray.train.torch.prepare_model` or :meth:`~ray.train.torch.prepare_data_loader` in your training function. Instead,
    keep using `deepspeed.initialize() <https://deepspeed.readthedocs.io/en/latest/initialize.html>`_ as usual to prepare everything
    for distributed training.

Run DeepSpeed with other frameworks
-----------------------------------

Many deep learning frameworks have integrated with DeepSpeed, including Lightning, Transformers, Accelerate, and more. You can run all these combinations in Ray Train.

Check the below examples for more details:

.. list-table::
   :header-rows: 1

   * - Framework
     - Example
   * - Accelerate (:ref:`User Guide <train-hf-accelerate>`)
     - `Fine-tune Llama-2 series models with Deepspeed, Accelerate, and Ray Train. <https://github.com/ray-project/ray/tree/master/doc/source/templates/04_finetuning_llms_with_deepspeed>`_
   * - Transformers (:ref:`User Guide <train-pytorch-transformers>`)
     - :doc:`Fine-tune GPT-J-6b with DeepSpeed and Hugging Face Transformers <examples/deepspeed/gptj_deepspeed_fine_tuning>`
   * - Lightning (:ref:`User Guide <train-pytorch-lightning>`)
     - :doc:`Fine-tune vicuna-13b with DeepSpeed and PyTorch Lightning <examples/lightning/vicuna_13b_lightning_deepspeed_finetune>`


.. _train-gbdt-guide:

Get Started with Distributed Training using XGBoost and LightGBM
================================================================

Ray Train has built-in support for XGBoost and LightGBM.


Quickstart
-----------
.. tab-set::

    .. tab-item:: XGBoost

        .. literalinclude:: doc_code/gbdt_user_guide.py
           :language: python
           :start-after: __xgboost_start__
           :end-before: __xgboost_end__

    .. tab-item:: LightGBM

        .. literalinclude:: doc_code/gbdt_user_guide.py
           :language: python
           :start-after: __lightgbm_start__
           :end-before: __lightgbm_end__


Basic training with tree-based models in Train
----------------------------------------------

Just as in the original `xgboost.train() <https://xgboost.readthedocs.io/en/stable/parameter.html>`__ and
`lightgbm.train() <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`__ functions, the
training parameters are passed as the ``params`` dictionary.

.. tab-set::

    .. tab-item:: XGBoost

        .. literalinclude:: doc_code/gbdt_user_guide.py
            :language: python
            :start-after: __xgboost_start__
            :end-before: __xgboost_end__

    .. tab-item:: LightGBM

        .. literalinclude:: doc_code/gbdt_user_guide.py
            :language: python
            :start-after: __lightgbm_start__
            :end-before: __lightgbm_end__


Trainer constructors pass Ray-specific parameters.


.. _train-gbdt-checkpoints:

Save and load XGBoost and LightGBM checkpoints
----------------------------------------------

When you train a new tree on every boosting round,
you can save a checkpoint to snapshot the training progress so far.
:class:`~ray.train.xgboost.XGBoostTrainer` and :class:`~ray.train.lightgbm.LightGBMTrainer`
both implement checkpointing out of the box. These checkpoints can be loaded into memory
using static methods :meth:`XGBoostTrainer.get_model <ray.train.xgboost.XGBoostTrainer.get_model>` and
:meth:`LightGBMTrainer.get_model <ray.train.lightgbm.LightGBMTrainer.get_model>`.

The only required change is to configure :class:`~ray.train.CheckpointConfig` to set
the checkpointing frequency. For example, the following configuration
saves a checkpoint on every boosting round and only keeps the latest checkpoint:

.. literalinclude:: doc_code/key_concepts.py
    :language: python
    :start-after: __checkpoint_config_ckpt_freq_start__
    :end-before: __checkpoint_config_ckpt_freq_end__

.. tip::

    Once you enable checkpointing, you can follow :ref:`this guide <train-fault-tolerance>`
    to enable fault tolerance.


How to scale out training?
--------------------------

The benefit of using Ray Train is that you can seamlessly scale up your training by
adjusting the :class:`ScalingConfig <ray.train.ScalingConfig>`.

.. note::
    Ray Train doesn't modify or otherwise alter the working
    of the underlying XGBoost or LightGBM distributed training algorithms.
    Ray only provides orchestration, data ingest and fault tolerance.
    For more information on GBDT distributed training, refer to
    `XGBoost documentation <https://xgboost.readthedocs.io>`__ and
    `LightGBM documentation <https://lightgbm.readthedocs.io/>`__.


Following are some examples of common use-cases:

.. tab-set::

    .. tab-item:: Multi-node CPU

        Setup: 4 nodes with 8 CPUs each.

        Use-case: To utilize all resources in multi-node training.

        .. literalinclude:: doc_code/gbdt_user_guide.py
            :language: python
            :start-after: __scaling_cpu_start__
            :end-before: __scaling_cpu_end__


    .. tab-item:: Single-node multi-GPU

        Setup: 1 node with 8 CPUs and 4 GPUs.

        Use-case: If you have a single node with multiple GPUs, you need to use
        distributed training to leverage all GPUs.

        .. literalinclude:: doc_code/gbdt_user_guide.py
            :language: python
            :start-after: __scaling_gpu_start__
            :end-before: __scaling_gpu_end__

    .. tab-item:: Multi-node multi-GPU

        Setup: 4 node with 8 CPUs and 4 GPUs each.

        Use-case: If you have a multiple nodes with multiple GPUs, you need to
        schedule one worker per GPU.

        .. literalinclude:: doc_code/gbdt_user_guide.py
            :language: python
            :start-after: __scaling_gpumulti_start__
            :end-before: __scaling_gpumulti_end__

        Note that you just have to adjust the number of workers. Ray handles everything else
        automatically.


.. warning::

    Specifying a *shared storage location* (such as cloud storage or NFS) is
    *optional* for single-node clusters, but it is **required for multi-node clusters.**
    Using a local path will :ref:`raise an error <multinode-local-storage-warning>`
    during checkpointing for multi-node clusters.

    .. testcode:: python
        :skipif: True

        trainer = XGBoostTrainer(
            ..., run_config=ray.train.RunConfig(storage_path="s3://...")
        )


How many remote actors should you use?
--------------------------------------

This depends on your workload and your cluster setup.
Generally there is no inherent benefit of running more than
one remote actor per node for CPU-only training. This is because
XGBoost can already leverage multiple CPUs with threading.

However, in some cases, you should consider some starting
more than one actor per node:

* For **multi GPU training**, each GPU should have a separate
  remote actor. Thus, if your machine has 24 CPUs and 4 GPUs,
  you want to start 4 remote actors with 6 CPUs and 1 GPU
  each
* In a **heterogeneous cluster** , you might want to find the
  `greatest common divisor <https://en.wikipedia.org/wiki/Greatest_common_divisor>`_
  for the number of CPUs.
  For example, for a cluster with three nodes of 4, 8, and 12 CPUs, respectively,
  you should set the number of actors to 6 and the CPUs per
  actor to 4.

How to use GPUs for training?
-----------------------------

Ray Train enables multi-GPU training for XGBoost and LightGBM. The core backends
automatically leverage NCCL2 for cross-device communication.
All you have to do is to start one actor per GPU and set GPU-compatible parameters.
For example, XGBoost's ``tree_method`` to ``gpu_hist``. See XGBoost
documentation for more details.

For instance, if you have 2 machines with 4 GPUs each, you want
to start 8 workers, and set ``use_gpu=True``. There is usually
no benefit in allocating less (for example, 0.5) or more than one GPU per actor.

You should divide the CPUs evenly across actors per machine, so if your
machines have 16 CPUs in addition to the 4 GPUs, each actor should have
4 CPUs to use.


.. literalinclude:: doc_code/gbdt_user_guide.py
    :language: python
    :start-after: __gpu_xgboost_start__
    :end-before: __gpu_xgboost_end__


.. _data-ingest-gbdt:

How to preprocess data for training?
------------------------------------

Particularly for tabular data, Ray Data comes with out-of-the-box :ref:`preprocessors <preprocessor-ref>` that implement common feature preprocessing operations.
You can use this with Ray Train Trainers by applying them on the dataset before passing the dataset into a Trainer. For example:


.. literalinclude:: ../data/doc_code/preprocessors.py
    :language: python
    :start-after: __trainer_start__
    :end-before: __trainer_end__


How to optimize XGBoost memory usage?
-------------------------------------

XGBoost uses a compute-optimized datastructure, the ``DMatrix``,
to hold training data. When converting a dataset to a ``DMatrix``,
XGBoost creates intermediate copies and ends up
holding a complete copy of the full data. XGBoost converts the data
into the local data format. On a 64-bit system the format is 64-bit floats.
Depending on the system and original dataset dtype, this matrix can
thus occupy more memory than the original dataset.

The **peak memory usage** for CPU-based training is at least
**3x** the dataset size, assuming dtype ``float32`` on a 64-bit system,
plus about **400,000 KiB** for other resources,
like operating system requirements and storing of intermediate
results.

**Example**


* Machine type: AWS m5.xlarge (4 vCPUs, 16 GiB RAM)
* Usable RAM: ~15,350,000 KiB
* Dataset: 1,250,000 rows with 1024 features, dtype float32.
  Total size: 5,000,000 KiB
* XGBoost DMatrix size: ~10,000,000 KiB

This dataset fits exactly on this node for training.

Note that the DMatrix size might be lower on a 32 bit system.

**GPUs**

Generally, the same memory requirements exist for GPU-based
training. Additionally, the GPU must have enough memory
to hold the dataset.

In the preceding example, the GPU must have at least
10,000,000 KiB (about 9.6 GiB) memory. However,
empirical data shows that using a ``DeviceQuantileDMatrix``
seems to result in more peak GPU memory usage, possibly
for intermediate storage when loading data (about 10%).

**Best practices**

In order to reduce peak memory usage, consider the following
suggestions:


* Store data as ``float32`` or less. You often don't need
  more precision is often, and keeping data in a smaller format
  helps reduce peak memory usage for initial data loading.
* Pass the ``dtype`` when loading data from CSV. Otherwise,
  floating point values are loaded as ``np.float64``
  per default, increasing peak memory usage by 33%.


.. _train-benchmarks:

Ray Train Benchmarks
====================

Below we document key performance benchmarks for common Ray Train tasks and workflows.

.. _pytorch_gpu_training_benchmark:

GPU image training
------------------

This task uses the TorchTrainer module to train different amounts of data
using an Pytorch ResNet model.

We test out the performance across different cluster sizes and data sizes.

- `GPU image training script`_
- `GPU training small cluster configuration`_
- `GPU training large cluster configuration`_

.. note::

    For multi-host distributed training, on AWS we need to ensure ec2 instances are in the same VPC and
    all ports are open in the secure group.


.. list-table::

    * - **Cluster Setup**
      - **Data Size**
      - **Performance**
      - **Command**
    * - 1 g3.8xlarge node (1 worker)
      - 1 GB (1623 images)
      - 79.76 s (2 epochs, 40.7 images/sec)
      - `python pytorch_training_e2e.py --data-size-gb=1`
    * - 1 g3.8xlarge node (1 worker)
      - 20 GB (32460 images)
      - 1388.33 s (2 epochs, 46.76 images/sec)
      - `python pytorch_training_e2e.py --data-size-gb=20`
    * - 4 g3.16xlarge nodes (16 workers)
      - 100 GB (162300 images)
      - 434.95 s (2 epochs, 746.29 images/sec)
      - `python pytorch_training_e2e.py --data-size-gb=100 --num-workers=16`

.. _pytorch-training-parity:

Pytorch Training Parity
-----------------------

This task checks the performance parity between native Pytorch Distributed and
Ray Train's distributed TorchTrainer.

We demonstrate that the performance is similar (within 2.5\%) between the two frameworks.
Performance may vary greatly across different model, hardware, and cluster configurations.

The reported times are for the raw training times. There is an unreported constant setup
overhead of a few seconds for both methods that is negligible for longer training runs.

- `Pytorch comparison training script`_
- `Pytorch comparison CPU cluster configuration`_
- `Pytorch comparison GPU cluster configuration`_

.. list-table::

    * - **Cluster Setup**
      - **Dataset**
      - **Performance**
      - **Command**
    * - 4 m5.2xlarge nodes (4 workers)
      - FashionMNIST
      - 196.64 s (vs 194.90 s Pytorch)
      - `python workloads/torch_benchmark.py run --num-runs 3 --num-epochs 20 --num-workers 4 --cpus-per-worker 8`
    * - 4 m5.2xlarge nodes (16 workers)
      - FashionMNIST
      - 430.88 s (vs 475.97 s Pytorch)
      - `python workloads/torch_benchmark.py run --num-runs 3 --num-epochs 20 --num-workers 16 --cpus-per-worker 2`
    * - 4 g4dn.12xlarge node (16 workers)
      - FashionMNIST
      - 149.80 s (vs 146.46 s Pytorch)
      - `python workloads/torch_benchmark.py run --num-runs 3 --num-epochs 20 --num-workers 16 --cpus-per-worker 4 --use-gpu`


.. _tf-training-parity:

Tensorflow Training Parity
--------------------------

This task checks the performance parity between native Tensorflow Distributed and
Ray Train's distributed TensorflowTrainer.

We demonstrate that the performance is similar (within 1\%) between the two frameworks.
Performance may vary greatly across different model, hardware, and cluster configurations.

The reported times are for the raw training times. There is an unreported constant setup
overhead of a few seconds for both methods that is negligible for longer training runs.

.. note:: The batch size and number of epochs is different for the GPU benchmark, resulting in a longer runtime.

- `Tensorflow comparison training script`_
- `Tensorflow comparison CPU cluster configuration`_
- `Tensorflow comparison GPU cluster configuration`_

.. list-table::

    * - **Cluster Setup**
      - **Dataset**
      - **Performance**
      - **Command**
    * - 4 m5.2xlarge nodes (4 workers)
      - FashionMNIST
      - 78.81 s (vs 79.67 s Tensorflow)
      - `python workloads/tensorflow_benchmark.py run --num-runs 3 --num-epochs 20 --num-workers 4 --cpus-per-worker 8`
    * - 4 m5.2xlarge nodes (16 workers)
      - FashionMNIST
      - 64.57 s (vs 67.45 s Tensorflow)
      - `python workloads/tensorflow_benchmark.py run --num-runs 3 --num-epochs 20 --num-workers 16 --cpus-per-worker 2`
    * - 4 g4dn.12xlarge node (16 workers)
      - FashionMNIST
      - 465.16 s (vs 461.74 s Tensorflow)
      - `python workloads/tensorflow_benchmark.py run --num-runs 3 --num-epochs 200 --num-workers 16 --cpus-per-worker 4 --batch-size 64 --use-gpu`

.. _xgboost-benchmark:

XGBoost training
----------------

This task uses the XGBoostTrainer module to train on different sizes of data
with different amounts of parallelism to show near-linear scaling from distributed
data parallelism.

XGBoost parameters were kept as defaults for ``xgboost==1.7.6`` this task.


- `XGBoost Training Script`_
- `XGBoost Cluster Configuration`_

.. list-table::

    * - **Cluster Setup**
      - **Number of distributed training workers**
      - **Data Size**
      - **Performance**
      - **Command**
    * - 1 m5.4xlarge node with 16 CPUs
      - 1 training worker using 12 CPUs, leaving 4 CPUs for Ray Data tasks
      - 10 GB (26M rows)
      - 310.22 s
      - `python train_batch_inference_benchmark.py "xgboost" --size=10GB`
    * - 10 m5.4xlarge nodes
      - 10 training workers (one per node), using 10x12 CPUs, leaving 10x4 CPUs for Ray Data tasks
      - 100 GB (260M rows)
      - 326.86 s
      - `python train_batch_inference_benchmark.py "xgboost" --size=100GB`

.. _`GPU image training script`: https://github.com/ray-project/ray/blob/cec82a1ced631525a4d115e4dc0c283fa4275a7f/release/air_tests/air_benchmarks/workloads/pytorch_training_e2e.py#L95-L106
.. _`GPU training small cluster configuration`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/compute_gpu_1_aws.yaml#L6-L24
.. _`GPU training large cluster configuration`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/compute_gpu_4x4_aws.yaml#L5-L25
.. _`Pytorch comparison training script`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/workloads/torch_benchmark.py
.. _`Pytorch comparison CPU cluster configuration`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/compute_cpu_4_aws.yaml
.. _`Pytorch comparison GPU cluster configuration`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/compute_gpu_4x4_aws.yaml
.. _`Tensorflow comparison training script`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/workloads/tensorflow_benchmark.py
.. _`Tensorflow comparison CPU cluster configuration`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/compute_cpu_4_aws.yaml
.. _`Tensorflow comparison GPU cluster configuration`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/compute_gpu_4x4_aws.yaml
.. _`XGBoost Training Script`: https://github.com/ray-project/ray/blob/9ac58f4efc83253fe63e280106f959fe317b1104/release/train_tests/xgboost_lightgbm/train_batch_inference_benchmark.py
.. _`XGBoost Cluster Configuration`: https://github.com/ray-project/ray/tree/9ac58f4efc83253fe63e280106f959fe317b1104/release/train_tests/xgboost_lightgbm


.. _train-pytorch:

Get Started with Distributed Training using PyTorch
===================================================

This tutorial walks through the process of converting an existing PyTorch script to use Ray Train.

Learn how to:

1. Configure a model to run distributed and on the correct CPU/GPU device.
2. Configure a dataloader to shard data across the :ref:`workers <train-overview-worker>` and place data on the correct CPU or GPU device.
3. Configure a :ref:`training function <train-overview-training-function>` to report metrics and save checkpoints.
4. Configure :ref:`scaling <train-overview-scaling-config>` and CPU or GPU resource requirements for a training job.
5. Launch a distributed training job with a :class:`~ray.train.torch.TorchTrainer` class.

Quickstart
----------

For reference, the final code will look something like the following:

.. testcode::
    :skipif: True

    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    def train_func():
        # Your PyTorch training code here.
        ...

    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()

1. `train_func` is the Python code that executes on each distributed training worker.
2. :class:`~ray.train.ScalingConfig` defines the number of distributed training workers and whether to use GPUs.
3. :class:`~ray.train.torch.TorchTrainer` launches the distributed training job.

Compare a PyTorch training script with and without Ray Train.

.. tab-set::

    .. tab-item:: PyTorch

        .. This snippet isn't tested because it doesn't use any Ray code.

        .. testcode::
            :skipif: True

            import os
            import tempfile

            import torch
            from torch.nn import CrossEntropyLoss
            from torch.optim import Adam
            from torch.utils.data import DataLoader
            from torchvision.models import resnet18
            from torchvision.datasets import FashionMNIST
            from torchvision.transforms import ToTensor, Normalize, Compose

            # Model, Loss, Optimizer
            model = resnet18(num_classes=10)
            model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            model.to("cuda")
            criterion = CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=0.001)

            # Data
            transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
            train_data = FashionMNIST(root='./data', train=True, download=True, transform=transform)
            train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

            # Training
            for epoch in range(10):
                for images, labels in train_loader:
                    images, labels = images.to("cuda"), labels.to("cuda")
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                metrics = {"loss": loss.item(), "epoch": epoch}
                checkpoint_dir = tempfile.mkdtemp()
                checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(metrics)



    .. tab-item:: PyTorch + Ray Train

        .. code-block:: python
            :emphasize-lines: 12, 14, 21, 55-58, 59, 63, 66-68, 72-73, 76

            import os
            import tempfile

            import torch
            from torch.nn import CrossEntropyLoss
            from torch.optim import Adam
            from torch.utils.data import DataLoader
            from torchvision.models import resnet18
            from torchvision.datasets import FashionMNIST
            from torchvision.transforms import ToTensor, Normalize, Compose

            import ray.train.torch

            def train_func():
                # Model, Loss, Optimizer
                model = resnet18(num_classes=10)
                model.conv1 = torch.nn.Conv2d(
                    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
                # [1] Prepare model.
                model = ray.train.torch.prepare_model(model)
                # model.to("cuda")  # This is done by `prepare_model`
                criterion = CrossEntropyLoss()
                optimizer = Adam(model.parameters(), lr=0.001)

                # Data
                transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
                data_dir = os.path.join(tempfile.gettempdir(), "data")
                train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
                train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
                # [2] Prepare dataloader.
                train_loader = ray.train.torch.prepare_data_loader(train_loader)

                # Training
                for epoch in range(10):
                    if ray.train.get_context().get_world_size() > 1:
                        train_loader.sampler.set_epoch(epoch)

                    for images, labels in train_loader:
                        # This is done by `prepare_data_loader`!
                        # images, labels = images.to("cuda"), labels.to("cuda")
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # [3] Report metrics and checkpoint.
                    metrics = {"loss": loss.item(), "epoch": epoch}
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        torch.save(
                            model.module.state_dict(),
                            os.path.join(temp_checkpoint_dir, "model.pt")
                        )
                        ray.train.report(
                            metrics,
                            checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
                        )
                    if ray.train.get_context().get_world_rank() == 0:
                        print(metrics)

            # [4] Configure scaling and resource requirements.
            scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=True)

            # [5] Launch distributed training job.
            trainer = ray.train.torch.TorchTrainer(
                train_func,
                scaling_config=scaling_config,
                # [5a] If running in a multi-node cluster, this is where you
                # should configure the run's persistent storage that is accessible
                # across all worker nodes.
                # run_config=ray.train.RunConfig(storage_path="s3://..."),
            )
            result = trainer.fit()

            # [6] Load the trained model.
            with result.checkpoint.as_directory() as checkpoint_dir:
                model_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))
                model = resnet18(num_classes=10)
                model.conv1 = torch.nn.Conv2d(
                    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
                model.load_state_dict(model_state_dict)


Set up a training function
--------------------------

.. include:: ./common/torch-configure-train_func.rst

Set up a model
^^^^^^^^^^^^^^

Use the :func:`ray.train.torch.prepare_model` utility function to:

1. Move your model to the correct device.
2. Wrap it in ``DistributedDataParallel``.

.. code-block:: diff

    -from torch.nn.parallel import DistributedDataParallel
    +import ray.train.torch

     def train_func():

         ...

         # Create model.
         model = ...

         # Set up distributed training and device placement.
    -    device_id = ... # Your logic to get the right device.
    -    model = model.to(device_id or "cpu")
    -    model = DistributedDataParallel(model, device_ids=[device_id])
    +    model = ray.train.torch.prepare_model(model)

         ...

Set up a dataset
^^^^^^^^^^^^^^^^

.. TODO: Update this to use Ray Data.

Use the :func:`ray.train.torch.prepare_data_loader` utility function, which:

1. Adds a :class:`~torch.utils.data.distributed.DistributedSampler` to your :class:`~torch.utils.data.DataLoader`.
2. Moves the batches to the right device.

Note that this step isn't necessary if you're passing in Ray Data to your Trainer.
See :ref:`data-ingest-torch`.

.. code-block:: diff

     from torch.utils.data import DataLoader
    +import ray.train.torch

     def train_func():

         ...

         dataset = ...

         data_loader = DataLoader(dataset, batch_size=worker_batch_size, shuffle=True)
    +    data_loader = ray.train.torch.prepare_data_loader(data_loader)

         for epoch in range(10):
    +        if ray.train.get_context().get_world_size() > 1:
    +            data_loader.sampler.set_epoch(epoch)

             for X, y in data_loader:
    -            X = X.to_device(device)
    -            y = y.to_device(device)

         ...

.. tip::
    Keep in mind that ``DataLoader`` takes in a ``batch_size`` which is the batch size for each worker.
    The global batch size can be calculated from the worker batch size (and vice-versa) with the following equation:

    .. testcode::
        :skipif: True

        global_batch_size = worker_batch_size * ray.train.get_context().get_world_size()

.. note::
    If you already manually set up your ``DataLoader`` with a ``DistributedSampler``,
    :meth:`~ray.train.torch.prepare_data_loader` will not add another one, and will
    respect the configuration of the existing sampler.

.. note::
    :class:`~torch.utils.data.distributed.DistributedSampler` does not work with a
    ``DataLoader`` that wraps :class:`~torch.utils.data.IterableDataset`.
    If you want to work with an dataset iterator,
    consider using :ref:`Ray Data <data>` instead of PyTorch DataLoader since it
    provides performant streaming data ingestion for large scale datasets.

    See :ref:`data-ingest-torch` for more details.

Report checkpoints and metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To monitor progress, you can report intermediate metrics and checkpoints using the :func:`ray.train.report` utility function.

.. code-block:: diff

    +import os
    +import tempfile

    +import ray.train

     def train_func():

         ...

         with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt")
            )

    +       metrics = {"loss": loss.item()}  # Training/validation metrics.

            # Build a Ray Train checkpoint from a directory
    +       checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)

            # Ray Train will automatically save the checkpoint to persistent storage,
            # so the local `temp_checkpoint_dir` can be safely cleaned up after.
    +       ray.train.report(metrics=metrics, checkpoint=checkpoint)

         ...

For more details, see :ref:`train-monitoring-and-logging` and :ref:`train-checkpointing`.


.. include:: ./common/torch-configure-run.rst


Next steps
----------

After you have converted your PyTorch training script to use Ray Train:

* See :ref:`User Guides <train-user-guides>` to learn more about how to perform specific tasks.
* Browse the :doc:`Examples <examples>` for end-to-end examples of how to use Ray Train.
* Dive into the :ref:`API Reference <train-api>` for more details on the classes and methods used in this tutorial.


.. _train-monitoring-and-logging:

Monitoring and Logging Metrics
==============================

Ray Train provides an API for reporting intermediate
results and checkpoints from the training function (run on distributed workers) up to the
``Trainer`` (where your python script is executed) by calling ``train.report(metrics)``.
The results will be collected from the distributed workers and passed to the driver to
be logged and displayed.

.. warning::

    Only the results from rank 0 worker will be used. However, in order to ensure
    consistency, ``train.report()`` has to be called on each worker. If you
    want to aggregate results from multiple workers, see :ref:`train-aggregating-results`.

The primary use-case for reporting is for metrics (accuracy, loss, etc.) at
the end of each training epoch.

.. tab-set::

    .. tab-item:: PyTorch

        .. testcode::

            from ray import train

            def train_func():
                ...
                for i in range(num_epochs):
                    result = model.train(...)
                    train.report({"result": result})

    .. tab-item:: PyTorch Lightning

        In PyTorch Lightning, we use a callback to call ``train.report()``.

        .. testcode::
            :skipif: True

            from ray import train
            import pytorch_lightning as pl
            from pytorch_lightning.callbacks import Callback

            class MyRayTrainReportCallback(Callback):
                def on_train_epoch_end(self, trainer, pl_module):
                    metrics = trainer.callback_metrics
                    metrics = {k: v.item() for k, v in metrics.items()}

                    train.report(metrics=metrics)

            def train_func_per_worker():
                ...
                trainer = pl.Trainer(
                    # ...
                    callbacks=[MyRayTrainReportCallback()]
                )
                trainer.fit()

.. _train-aggregating-results:

How to obtain and aggregate results from different workers?
-----------------------------------------------------------

In real applications, you may want to calculate optimization metrics besides accuracy and loss: recall, precision, Fbeta, etc.
You may also want to collect metrics from multiple workers. While Ray Train currently only reports metrics from the rank 0
worker, you can use third-party libraries or distributed primitives of your machine learning framework to report
metrics from multiple workers.


.. tab-set::

    .. tab-item:: Native PyTorch

        Ray Train natively supports `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/>`_, which provides a collection of machine learning metrics for distributed, scalable PyTorch models.

        Here is an example of reporting both the aggregated R2 score and mean train and validation loss from all workers.

        .. literalinclude:: ../doc_code/torchmetrics_example.py
            :language: python
            :start-after: __start__


.. _train-experiment-tracking-native:

===================
Experiment Tracking
===================

.. note::
    This guide is relevant for all trainers in which you define a custom training loop.
    This includes :class:`TorchTrainer <ray.train.torch.TorchTrainer>` and
    :class:`TensorflowTrainer <ray.train.tensorflow.TensorflowTrainer>`.

Most experiment tracking libraries work out-of-the-box with Ray Train.
This guide provides instructions on how to set up the code so that your favorite experiment tracking libraries
can work for distributed training with Ray Train. The end of the guide has common errors to aid in debugging
the setup.

The following pseudo code demonstrates how to use the native experiment tracking library calls
inside of Ray Train:

.. testcode::
    :skipif: True

    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    def train_func():
        # Training code and native experiment tracking library calls go here.

    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()

Ray Train lets you use native experiment tracking libraries by customizing the tracking
logic inside the :ref:`train_func<train-overview-training-function>` function.
In this way, you can port your experiment tracking logic to Ray Train with minimal changes.

Getting Started
===============

Let's start by looking at some code snippets.

The following examples uses Weights & Biases (W&B) and MLflow but it's adaptable to other frameworks.

.. tab-set::

    .. tab-item:: W&B

        .. testcode::
            :skipif: True

            import ray
            from ray import train
            import wandb

            # Step 1
            # This ensures that all ray worker processes have `WANDB_API_KEY` set.
            ray.init(runtime_env={"env_vars": {"WANDB_API_KEY": "your_api_key"}})

            def train_func():
                # Step 1 and 2
                if train.get_context().get_world_rank() == 0:
                    wandb.init(
                        name=...,
                        project=...,
                        # ...
                    )

                # ...
                loss = optimize()
                metrics = {"loss": loss}

                # Step 3
                if train.get_context().get_world_rank() == 0:
                    wandb.log(metrics)

                # ...

                # Step 4
                # Make sure that all loggings are uploaded to the W&B backend.
                if train.get_context().get_world_rank() == 0:
                    wandb.finish()

    .. tab-item:: MLflow

        .. testcode::
            :skipif: True

            from ray import train
            import mlflow

            # Run the following on the head node:
            # $ databricks configure --token
            # mv ~/.databrickscfg YOUR_SHARED_STORAGE_PATH
            # This function assumes `databricks_config_file` is specified in the Trainer's `train_loop_config`.
            def train_func(config):
                # Step 1 and 2
                os.environ["DATABRICKS_CONFIG_FILE"] = config["databricks_config_file"]
                mlflow.set_tracking_uri("databricks")
                mlflow.set_experiment_id(...)
                mlflow.start_run()

                # ...

                loss = optimize()

                metrics = {"loss": loss}
                # Only report the results from the first worker to MLflow
                to avoid duplication

                # Step 3
                if train.get_context().get_world_rank() == 0:
                    mlflow.log_metrics(metrics)

.. tip::

    A major difference between distributed and non-distributed training is that in distributed training,
    multiple processes are running in parallel and under certain setups they have the same results. If all
    of them report results to the tracking backend, you may get duplicated results. To address that,
    Ray Train lets you apply logging logic to only the rank 0 worker with the following method:
    :meth:`ray.train.get_context().get_world_rank() <ray.train.context.TrainContext.get_world_rank>`.

    .. testcode::
        :skipif: True

        from ray import train
        def train_func():
            ...
            if train.get_context().get_world_rank() == 0:
                # Add your logging logic only for rank0 worker.
            ...

The interaction with the experiment tracking backend within the :ref:`train_func<train-overview-training-function>`
has 4 logical steps:

#. Set up the connection to a tracking backend
#. Configure and launch a run
#. Log metrics
#. Finish the run

More details about each step follows.

Step 1: Connect to your tracking backend
----------------------------------------

First, decide which tracking backend to use: W&B, MLflow, TensorBoard, Comet, etc.
If applicable, make sure that you properly set up credentials on each training worker.

.. tab-set::

    .. tab-item:: W&B

        W&B offers both *online* and *offline* modes.

        **Online**

        For *online* mode, because you log to W&B's tracking service, ensure that you set the credentials
        inside of :ref:`train_func<train-overview-training-function>`. See :ref:`Set up credentials<set-up-credentials>`
        for more information.

        .. testcode::
            :skipif: True

            # This is equivalent to `os.environ["WANDB_API_KEY"] = "your_api_key"`
            wandb.login(key="your_api_key")

        **Offline**

        For *offline* mode, because you log towards a local file system,
        point the offline directory to a shared storage path that all nodes can write to.
        See :ref:`Set up a shared file system<set-up-shared-file-system>` for more information.

        .. testcode::
            :skipif: True

            os.environ["WANDB_MODE"] = "offline"
            wandb.init(dir="some_shared_storage_path/wandb")

    .. tab-item:: MLflow

        MLflow offers both *local* and *remote* (for example, to Databrick's MLflow service) modes.

        **Local**

        For *local* mode, because you log to a local file
        system, point offline directory to a shared storage path. that all nodes can write
        to. See :ref:`Set up a shared file system<set-up-shared-file-system>` for more information.

        .. testcode::
            :skipif: True

            mlflow.start_run(tracking_uri="file:some_shared_storage_path/mlruns")

        **Remote, hosted by Databricks**

        Ensure that all nodes have access to the Databricks config file.
        See :ref:`Set up credentials<set-up-credentials>` for more information.

        .. testcode::
            :skipif: True

            # The MLflow client looks for a Databricks config file
            # at the location specified by `os.environ["DATABRICKS_CONFIG_FILE"]`.
            os.environ["DATABRICKS_CONFIG_FILE"] = config["databricks_config_file"]
            mlflow.set_tracking_uri("databricks")
            mlflow.start_run()

.. _set-up-credentials:

Set up credentials
~~~~~~~~~~~~~~~~~~

Refer to each tracking library's API documentation on setting up credentials.
This step usually involves setting an environment variable or accessing a config file.

The easiest way to pass an environment variable credential to training workers is through
:ref:`runtime environments <runtime-environments>`, where you initialize with the following code:

.. testcode::
    :skipif: True

    import ray
    # This makes sure that training workers have the same env var set
    ray.init(runtime_env={"env_vars": {"SOME_API_KEY": "your_api_key"}})

For accessing the config file, ensure that the config file is accessible to all nodes.
One way to do this is by setting up a shared storage. Another way is to save a copy in each node.

.. _set-up-shared-file-system:

Set up a shared file system
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up a network filesystem accessible to all nodes in the cluster.
For example, AWS EFS or Google Cloud Filestore.

Step 2: Configure and start the run
-----------------------------------

This step usually involves picking an identifier for the run and associating it with a project.
Refer to the tracking libraries' documentation for semantics.

.. To conveniently link back to Ray Train run, you may want to log the persistent storage path
.. of the run as a config.

..
    .. testcode::

       def train_func():
            if ray.train.get_context().get_world_rank() == 0:
                   wandb.init(..., config={"ray_train_persistent_storage_path": "TODO: fill in when API stablizes"})

.. tip::

    When performing **fault-tolerant training** with auto-restoration, use a
    consistent ID to configure all tracking runs that logically belong to the same training run.
    One way to acquire an unique ID is with the following method:
    :meth:`ray.train.get_context().get_trial_id() <ray.train.context.TrainContext.get_trial_id>`.

    .. testcode::
        :skipif: True

        import ray
        from ray.train import ScalingConfig, RunConfig, FailureConfig
        from ray.train.torch import TorchTrainer

        def train_func():
            if ray.train.get_context().get_world_rank() == 0:
                wandb.init(id=ray.train.get_context().get_trial_id())
            ...

        trainer = TorchTrainer(
            train_func,
            run_config=RunConfig(failure_config=FailureConfig(max_failures=3))
        )

        trainer.fit()


Step 3: Log metrics
-------------------

You can customize how to log parameters, metrics, models, or media contents, within
:ref:`train_func<train-overview-training-function>`, just as in a non-distributed training script.
You can also use native integrations that a particular tracking framework has with
specific training frameworks. For example, ``mlflow.pytorch.autolog()``,
``lightning.pytorch.loggers.MLFlowLogger``, etc.

Step 4: Finish the run
----------------------

This step ensures that all logs are synced to the tracking service. Depending on the implementation of
various tracking libraries, sometimes logs are first cached locally and only synced to the tracking
service in an asynchronous fashion.
Finishing the run makes sure that all logs are synced by the time training workers exit.

.. tab-set::

    .. tab-item:: W&B

        .. testcode::
            :skipif: True

            # https://docs.wandb.ai/ref/python/finish
            wandb.finish()

    .. tab-item:: MLflow

        .. testcode::
            :skipif: True

            # https://mlflow.org/docs/1.2.0/python_api/mlflow.html
            mlflow.end_run()

    .. tab-item:: Comet

        .. testcode::
            :skipif: True

            # https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/#experimentend
            Experiment.end()

Examples
========

The following are runnable examples for PyTorch and PyTorch Lightning.

PyTorch
-------

.. dropdown:: Log to W&B

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking//torch_exp_tracking_wandb.py
            :emphasize-lines: 15, 16, 17, 21, 22, 51, 52, 54, 55
            :language: python
            :start-after: __start__

.. dropdown:: Log to file-based MLflow

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/torch_exp_tracking_mlflow.py
        :emphasize-lines: 22, 23, 24, 25, 54, 55, 57, 58, 64
        :language: python
        :start-after: __start__
        :end-before: __end__

PyTorch Lightning
-----------------

You can use the native Logger integration in PyTorch Lightning with W&B, CometML, MLFlow,
and Tensorboard, while using Ray Train's TorchTrainer.

The following example walks you through the process. The code here is runnable.

.. dropdown:: W&B

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_model_dl.py
        :language: python
        :start-after: __model_dl_start__

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_wandb.py
        :language: python
        :start-after: __lightning_experiment_tracking_wandb_start__

.. dropdown:: MLflow

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_model_dl.py
        :language: python
        :start-after: __model_dl_start__

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_mlflow.py
        :language: python
        :start-after: __lightning_experiment_tracking_mlflow_start__
        :end-before: __lightning_experiment_tracking_mlflow_end__

.. dropdown:: Comet

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_model_dl.py
        :language: python
        :start-after: __model_dl_start__

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_comet.py
        :language: python
        :start-after: __lightning_experiment_tracking_comet_start__

.. dropdown:: TensorBoard

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_model_dl.py
        :language: python
        :start-after: __model_dl_start__

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_tensorboard.py
        :language: python
        :start-after: __lightning_experiment_tracking_tensorboard_start__
        :end-before: __lightning_experiment_tracking_tensorboard_end__

Common Errors
=============

Missing Credentials
-------------------

**I have already called `wandb login` cli, but am still getting**

.. code-block:: none

    wandb: ERROR api_key not configured (no-tty). call wandb.login(key=[your_api_key]).

This is probably due to wandb credentials are not set up correctly
on worker nodes. Make sure that you run ``wandb.login``
or pass ``WANDB_API_KEY`` to each training function.
See :ref:`Set up credentials <set-up-credentials>` for more details.

Missing Configurations
----------------------

**I have already run `databricks configure`, but am still getting**

.. code-block:: none

    databricks_cli.utils.InvalidConfigurationError: You haven't configured the CLI yet!

This is usually caused by running ``databricks configure`` which
generates ``~/.databrickscfg`` only on head node. Move this file to a shared
location or copy it to each node.
See :ref:`Set up credentials <set-up-credentials>` for more details.


.. _train_scaling_config:

Configuring Scale and GPUs
==========================
Increasing the scale of a Ray Train training run is simple and can be done in a few lines of code.
The main interface for this is the :class:`~ray.train.ScalingConfig`, 
which configures the number of workers and the resources they should use.

In this guide, a *worker* refers to a Ray Train distributed training worker,
which is a :ref:`Ray Actor <actor-key-concept>` that runs your training function.

Increasing the number of workers
--------------------------------
The main interface to control parallelism in your training code is to set the
number of workers. This can be done by passing the ``num_workers`` attribute to
the :class:`~ray.train.ScalingConfig`:

.. testcode::

    from ray.train import ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=8
    )


Using GPUs
----------
To use GPUs, pass ``use_gpu=True`` to the :class:`~ray.train.ScalingConfig`.
This will request one GPU per training worker. In the example below, training will
run on 8 GPUs (8 workers, each using one GPU).

.. testcode::

    from ray.train import ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=8,
        use_gpu=True
    )


Using GPUs in the training function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When ``use_gpu=True`` is set, Ray Train will automatically set up environment variables
in your training function so that the GPUs can be detected and used
(e.g. ``CUDA_VISIBLE_DEVICES``).

You can get the associated devices with :meth:`ray.train.torch.get_device`.

.. testcode::

    import torch
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer, get_device


    def train_func():
        assert torch.cuda.is_available()

        device = get_device()
        assert device == torch.device("cuda:0")

    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True
        )
    )
    trainer.fit()

Assigning multiple GPUs to a worker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sometimes you might want to allocate multiple GPUs for a worker. For example, 
you can specify `resources_per_worker={"GPU": 2}` in the `ScalingConfig` if you want to 
assign 2 GPUs for each worker.

You can get a list of associated devices with :meth:`ray.train.torch.get_devices`.

.. testcode::

    import torch
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer, get_device, get_devices


    def train_func():
        assert torch.cuda.is_available()

        device = get_device()
        devices = get_devices()
        assert device == torch.device("cuda:0")
        assert devices == [torch.device("cuda:0"), torch.device("cuda:1")]

    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True,
            resources_per_worker={"GPU": 2}
        )
    )
    trainer.fit()


Setting the GPU type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ray Train allows you to specify the accelerator type for each worker.
This is useful if you want to use a specific accelerator type for model training.
In a heterogeneous Ray cluster, this means that your training workers will be forced to run on the specified GPU type, 
rather than on any arbitrary GPU node. You can get a list of supported `accelerator_type` from 
:ref:`the available accelerator types <accelerator_types>`.

For example, you can specify `accelerator_type="A100"` in the :class:`~ray.train.ScalingConfig` if you want to 
assign each worker a NVIDIA A100 GPU. 

.. tip::
    Ensure that your cluster has instances with the specified accelerator type 
    or is able to autoscale to fulfill the request.

.. testcode::

    ScalingConfig(
        num_workers=1,
        use_gpu=True,
        accelerator_type="A100"
    )


(PyTorch) Setting the communication backend 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch Distributed supports multiple `backends <https://pytorch.org/docs/stable/distributed.html#backends>`__
for communicating tensors across workers. By default Ray Train will use NCCL when ``use_gpu=True`` and Gloo otherwise.

If you explictly want to override this setting, you can configure a :class:`~ray.train.torch.TorchConfig` 
and pass it into the :class:`~ray.train.torch.TorchTrainer`.

.. testcode::
    :hide:

    num_training_workers = 1

.. testcode::

    from ray.train.torch import TorchConfig, TorchTrainer

    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(
            num_workers=num_training_workers,
            use_gpu=True, # Defaults to NCCL
        ),
        torch_config=TorchConfig(backend="gloo"),
    )

(NCCL) Setting the communication network interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using NCCL for distributed training, you can configure the network interface cards
that are used for communicating between GPUs by setting the 
`NCCL_SOCKET_IFNAME <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname>`__ 
environment variable.

To ensure that the environment variable is set for all training workers, you can pass it
in a :ref:`Ray runtime environment <runtime-environments>`:

.. testcode::
    :skipif: True

    import ray

    runtime_env = {"env_vars": {"NCCL_SOCKET_IFNAME": "ens5"}}
    ray.init(runtime_env=runtime_env)

    trainer = TorchTrainer(...)

Setting the resources per worker
--------------------------------
If you want to allocate more than one CPU or GPU per training worker, or if you
defined :ref:`custom cluster resources <cluster-resources>`, set
the ``resources_per_worker`` attribute:

.. testcode::

    from ray.train import ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=8,
        resources_per_worker={
            "CPU": 4,
            "GPU": 2,
        },
        use_gpu=True,
    )


.. note::
    If you specify GPUs in ``resources_per_worker``, you also need to set
    ``use_gpu=True``.

You can also instruct Ray Train to use fractional GPUs. In that case, multiple workers
will be assigned the same CUDA device.

.. testcode::

    from ray.train import ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=8,
        resources_per_worker={
            "CPU": 4,
            "GPU": 0.5,
        },
        use_gpu=True,
    )



.. _train_trainer_resources:

Trainer resources
-----------------
So far we've configured resources for each training worker. Technically, each
training worker is a :ref:`Ray Actor <actor-guide>`. Ray Train also schedules
an actor for the :class:`Trainer <ray.train.trainer.BaseTrainer>` object when
you call :meth:`Trainer.fit() <ray.train.trainer.BaseTrainer.fit>`.

This object often only manages lightweight communication between the training workers.
You can still specify its resources, which can be useful if you implemented your own
Trainer that does heavier processing.

.. testcode::

    from ray.train import ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=8,
        trainer_resources={
            "CPU": 4,
            "GPU": 1,
        }
    )

Per default, a trainer uses 1 CPU. If you have a cluster with 8 CPUs and want
to start 4 training workers a 2 CPUs, this will not work, as the total number
of required CPUs will be 9 (4 * 2 + 1). In that case, you can specify the trainer
resources to use 0 CPUs:

.. testcode::

    from ray.train import ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=4,
        resources_per_worker={
            "CPU": 2,
        },
        trainer_resources={
            "CPU": 0,
        }
    )


.. _persistent-storage-guide:

.. _train-log-dir:

Configuring Persistent Storage
==============================

A Ray Train run produces a history of :ref:`reported metrics <train-monitoring-and-logging>`,
:ref:`checkpoints <train-checkpointing>`, and :ref:`other artifacts <train-artifacts>`.
You can configure these to be saved to a persistent storage location.

.. figure:: ../images/persistent_storage_checkpoint.png
    :align: center
    :width: 600px

    An example of multiple workers spread across multiple nodes uploading checkpoints to persistent storage.

**Ray Train expects all workers to be able to write files to the same persistent storage location.**
Therefore, Ray Train requires some form of external persistent storage such as
cloud storage (e.g., S3, GCS) or a shared filesystem (e.g., AWS EFS, Google Filestore, HDFS)
for multi-node training.

Here are some capabilities that persistent storage enables:

- **Checkpointing and fault tolerance**: Saving checkpoints to a persistent storage location
  allows you to resume training from the last checkpoint in case of a node failure.
  See :ref:`train-checkpointing` for a detailed guide on how to set up checkpointing.
- **Post-experiment analysis**: A consolidated location storing data from all trials is useful for post-experiment analysis
  such as accessing the best checkpoints and hyperparameter configs after the cluster has already been terminated.
- **Bridge training/fine-tuning with downstream serving and batch inference tasks**: You can easily access the models
  and artifacts to share them with others or use them in downstream tasks.


Cloud storage (AWS S3, Google Cloud Storage)
--------------------------------------------

.. tip::

    Cloud storage is the recommended persistent storage option.

Use cloud storage by specifying a bucket URI as the :class:`RunConfig(storage_path) <ray.train.RunConfig>`:

.. testcode::
    :skipif: True

    from ray import train
    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(
        ...,
        run_config=train.RunConfig(
            storage_path="s3://bucket-name/sub-path/",
            name="experiment_name",
        )
    )


Ensure that all nodes in the Ray cluster have access to cloud storage, so outputs from workers can be uploaded to a shared cloud bucket.
In this example, all files are uploaded to shared storage at ``s3://bucket-name/sub-path/experiment_name`` for further processing.


Shared filesystem (NFS, HDFS)
-----------------------------

Use by specifying the shared storage path as the :class:`RunConfig(storage_path) <ray.train.RunConfig>`:

.. testcode::
    :skipif: True

    from ray import train
    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(
        ...,
        run_config=train.RunConfig(
            storage_path="/mnt/cluster_storage",
            # HDFS example:
            # storage_path=f"hdfs://{hostname}:{port}/subpath",
            name="experiment_name",
        )
    )

Ensure that all nodes in the Ray cluster have access to the shared filesystem, e.g. AWS EFS, Google Cloud Filestore, or HDFS,
so that outputs can be saved to there.
In this example, all files are saved to ``/mnt/cluster_storage/experiment_name`` for further processing.


Local storage
-------------

Using local storage for a single-node cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're just running an experiment on a single node (e.g., on a laptop), Ray Train will use the
local filesystem as the storage location for checkpoints and other artifacts.
Results are saved to ``~/ray_results`` in a sub-directory with a unique auto-generated name by default,
unless you customize this with ``storage_path`` and ``name`` in :class:`~ray.train.RunConfig`.


.. testcode::
    :skipif: True

    from ray import train
    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(
        ...,
        run_config=train.RunConfig(
            storage_path="/tmp/custom/storage/path",
            name="experiment_name",
        )
    )


In this example, all experiment results can found locally at ``/tmp/custom/storage/path/experiment_name`` for further processing.


.. _multinode-local-storage-warning:

Using local storage for a multi-node cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

    When running on multiple nodes, using the local filesystem of the head node as the persistent storage location is no longer supported.

    If you save checkpoints with :meth:`ray.train.report(..., checkpoint=...) <ray.train.report>`
    and run on a multi-node cluster, Ray Train will raise an error if NFS or cloud storage is not setup.
    This is because Ray Train expects all workers to be able to write the checkpoint to
    the same persistent storage location.

    If your training loop does not save checkpoints, the reported metrics will still
    be aggregated to the local storage path on the head node.

    See `this issue <https://github.com/ray-project/ray/issues/37177>`_ for more information.


.. _custom-storage-filesystem:

Custom storage
--------------

If the cases above don't suit your needs, Ray Train can support custom filesystems and perform custom logic.
Ray Train standardizes on the ``pyarrow.fs.FileSystem`` interface to interact with storage
(`see the API reference here <https://arrow.apache.org/docs/python/generated/pyarrow.fs.FileSystem.html>`_).

By default, passing ``storage_path=s3://bucket-name/sub-path/`` will use pyarrow's
`default S3 filesystem implementation <https://arrow.apache.org/docs/python/generated/pyarrow.fs.S3FileSystem.html>`_
to upload files. (`See the other default implementations. <https://arrow.apache.org/docs/python/api/filesystems.html#filesystem-implementations>`_)

Implement custom storage upload and download logic by providing an implementation of
``pyarrow.fs.FileSystem`` to :class:`RunConfig(storage_filesystem) <ray.train.RunConfig>`.

.. warning::

    When providing a custom filesystem, the associated ``storage_path`` is expected
    to be a qualified filesystem path *without the protocol prefix*.

    For example, if you provide a custom S3 filesystem for ``s3://bucket-name/sub-path/``,
    then the ``storage_path`` should be ``bucket-name/sub-path/`` with the ``s3://`` stripped.
    See the example below for example usage.

.. testcode::
    :skipif: True

    import pyarrow.fs

    from ray import train
    from ray.train.torch import TorchTrainer

    fs = pyarrow.fs.S3FileSystem(
        endpoint_override="http://localhost:9000",
        access_key=...,
        secret_key=...
    )

    trainer = TorchTrainer(
        ...,
        run_config=train.RunConfig(
            storage_filesystem=fs,
            storage_path="bucket-name/sub-path",
            name="experiment_name",
        )
    )


``fsspec`` filesystems
~~~~~~~~~~~~~~~~~~~~~~~

`fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_ offers many filesystem implementations,
such as ``s3fs``, ``gcsfs``, etc.

You can use any of these implementations by wrapping the ``fsspec`` filesystem with a ``pyarrow.fs`` utility:

.. testcode::
    :skipif: True

    # Make sure to install: `pip install -U s3fs`
    import s3fs
    import pyarrow.fs

    s3_fs = s3fs.S3FileSystem(
        key='miniokey...',
        secret='asecretkey...',
        endpoint_url='https://...'
    )
    custom_fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3_fs))

    run_config = RunConfig(storage_path="minio_bucket", storage_filesystem=custom_fs)

.. seealso::

    See the API references to the ``pyarrow.fs`` wrapper utilities:

    * https://arrow.apache.org/docs/python/generated/pyarrow.fs.PyFileSystem.html
    * https://arrow.apache.org/docs/python/generated/pyarrow.fs.FSSpecHandler.html



MinIO and other S3-compatible storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can follow the :ref:`examples shown above <custom-storage-filesystem>` to configure
a custom S3 filesystem to work with MinIO.

Note that including these as query parameters in the ``storage_path`` URI directly is another option:

.. testcode::
    :skipif: True

    from ray import train
    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(
        ...,
        run_config=train.RunConfig(
            storage_path="s3://bucket-name/sub-path?endpoint_override=http://localhost:9000",
            name="experiment_name",
        )
    )


Overview of Ray Train outputs
-----------------------------

So far, we covered how to configure the storage location for Ray Train outputs.
Let's walk through a concrete example to see what exactly these outputs are,
and how they're structured in storage.

.. seealso::

    This example includes checkpointing, which is covered in detail in :ref:`train-checkpointing`.

.. testcode::
    :skipif: True

    import os
    import tempfile

    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import TorchTrainer

    def train_fn(config):
        for i in range(10):
            # Training logic here

            metrics = {"loss": ...}

            # Save arbitrary artifacts to the working directory
            rank = train.get_context().get_world_rank()
            with open(f"artifact-rank={rank}-iter={i}.txt", "w") as f:
                f.write("data")

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(..., os.path.join(temp_checkpoint_dir, "checkpoint.pt"))
                train.report(
                    metrics,
                    checkpoint=Checkpoint.from_directory(temp_checkpoint_dir)
                )

    trainer = TorchTrainer(
        train_fn,
        scaling_config=train.ScalingConfig(num_workers=2),
        run_config=train.RunConfig(
            storage_path="s3://bucket-name/sub-path/",
            name="experiment_name",
            sync_config=train.SyncConfig(sync_artifacts=True),
        )
    )
    result: train.Result = trainer.fit()
    last_checkpoint: Checkpoint = result.checkpoint

Here's a rundown of all files that will be persisted to storage:

.. code-block:: text

    s3://bucket-name/sub-path (RunConfig.storage_path)
    └── experiment_name (RunConfig.name)          <- The "experiment directory"
        ├── experiment_state-*.json
        ├── basic-variant-state-*.json
        ├── trainer.pkl
        ├── tuner.pkl
        └── TorchTrainer_46367_00000_0_...        <- The "trial directory"
            ├── events.out.tfevents...            <- Tensorboard logs of reported metrics
            ├── result.json                       <- JSON log file of reported metrics
            ├── checkpoint_000000/                <- Checkpoints
            ├── checkpoint_000001/
            ├── ...
            ├── artifact-rank=0-iter=0.txt        <- Worker artifacts (see the next section)
            ├── artifact-rank=1-iter=0.txt
            └── ...

The :class:`~ray.train.Result` and :class:`~ray.train.Checkpoint` objects returned by
``trainer.fit`` are the easiest way to access the data in these files:

.. testcode::
    :skipif: True

    result.filesystem, result.path
    # S3FileSystem, "bucket-name/sub-path/experiment_name/TorchTrainer_46367_00000_0_..."

    result.checkpoint.filesystem, result.checkpoint.path
    # S3FileSystem, "bucket-name/sub-path/experiment_name/TorchTrainer_46367_00000_0_.../checkpoint_000009"


See :ref:`train-inspect-results` for a full guide on interacting with training :class:`Results <ray.train.Result>`.


.. _train-artifacts:

Persisting training artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the example above, we saved some artifacts within the training loop to the worker's
*current working directory*.
If you were training a stable diffusion model, you could save
some sample generated images every so often as a training artifact.

By default, Ray Train changes the current working directory of each worker to be inside the run's
:ref:`local staging directory <train-local-staging-dir>`.
This way, all distributed training workers share the same absolute path as the working directory.
See :ref:`below <train-working-directory>` for how to disable this default behavior,
which is useful if you want your training workers to keep their original working directories.

If :class:`RunConfig(SyncConfig(sync_artifacts=True)) <ray.train.SyncConfig>`, then
all artifacts saved in this directory will be persisted to storage.

The frequency of artifact syncing can be configured via :class:`SyncConfig <ray.train.SyncConfig>`.
Note that this behavior is off by default.

.. figure:: ../images/persistent_storage_artifacts.png
    :align: center
    :width: 600px

    Multiple workers spread across multiple nodes save artifacts to their local
    working directory, which is then persisted to storage.

.. warning::

    Artifacts saved by *every worker* will be synced to storage. If you have multiple workers
    co-located on the same node, make sure that workers don't delete files within their
    shared working directory.

    A best practice is to only write artifacts from a single worker unless you
    really need artifacts from multiple.

    .. testcode::
        :skipif: True

        from ray import train

        if train.get_context().get_world_rank() == 0:
            # Only the global rank 0 worker saves artifacts.
            ...

        if train.get_context().get_local_rank() == 0:
            # Every local rank 0 worker saves artifacts.
            ...


.. _train-storage-advanced:

Advanced configuration
----------------------

.. _train-local-staging-dir:

Setting the local staging directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

    Prior to 2.10, the ``RAY_AIR_LOCAL_CACHE_DIR`` environment variable and ``RunConfig(local_dir)``
    were ways to configure the local staging directory to be outside of the home directory (``~/ray_results``).

    **These configurations are no longer used to configure the local staging directory.
    Please instead use** ``RunConfig(storage_path)`` **to configure where your
    run's outputs go.**


Apart from files such as checkpoints written directly to the ``storage_path``,
Ray Train also writes some logfiles and metadata files to an intermediate
*local staging directory* before they get persisted (copied/uploaded) to the ``storage_path``.
The current working directory of each worker is set within this local staging directory.

By default, the local staging directory is a sub-directory of the Ray session
directory (e.g., ``/tmp/ray/session_latest``), which is also where other temporary Ray files are dumped.

Customize the location of the staging directory by :ref:`setting the location of the
temporary Ray session directory <temp-dir-log-files>`.

Here's an example of what the local staging directory looks like:

.. code-block:: text

    /tmp/ray/session_latest/artifacts/<ray-train-job-timestamp>/
    └── experiment_name
        ├── driver_artifacts    <- These are all uploaded to storage periodically
        │   ├── Experiment state snapshot files needed for resuming training
        │   └── Metrics logfiles
        └── working_dirs        <- These are uploaded to storage if `SyncConfig(sync_artifacts=True)`
            └── Current working directory of training workers, which contains worker artifacts

.. warning::

    You should not need to look into the local staging directory.
    The ``storage_path`` should be the only path that you need to interact with.

    The structure of the local staging directory is subject to change
    in future versions of Ray Train -- do not rely on these local staging files in your application.


.. _train-working-directory:

Keep the original current working directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To disable the default behavior of Ray Train changing the current working directory,
set the ``RAY_CHDIR_TO_TRIAL_DIR=0`` environment variable.

This is useful if you want your training workers to access relative paths from the
directory you launched the training script from.

.. tip::

    When running in a distributed cluster, you will need to make sure that all workers
    have a mirrored working directory to access the same relative paths.

    One way to achieve this is setting the
    :ref:`working directory in the Ray runtime environment <workflow-local-files>`.

.. testcode::

    import os

    import ray
    import ray.train
    from ray.train.torch import TorchTrainer

    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    # Write some file in the current working directory
    with open("./data.txt", "w") as f:
        f.write("some data")

    # Set the working directory in the Ray runtime environment
    ray.init(runtime_env={"working_dir": "."})

    def train_fn_per_worker(config):
        # Check that each worker can access the working directory
        # NOTE: The working directory is copied to each worker and is read only.
        assert os.path.exists("./data.txt"), os.getcwd()

        # To use artifact syncing with `SyncConfig(sync_artifacts=True)`,
        # write artifacts here, instead of the current working directory:
        ray.train.get_context().get_trial_dir()

    trainer = TorchTrainer(
        train_fn_per_worker,
        scaling_config=ray.train.ScalingConfig(num_workers=2),
        run_config=ray.train.RunConfig(
            # storage_path=...,
            sync_config=ray.train.SyncConfig(sync_artifacts=True),
        ),
    )
    trainer.fit()


.. _train-ray-storage:

Automatically setting up persistent storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can control where to store training results with the ``RAY_STORAGE``
environment variable.

For instance, if you set ``RAY_STORAGE="s3://my_bucket/train_results"``, your
results will automatically persisted there.

If you manually set a :attr:`RunConfig.storage_path <ray.train.RunConfig.storage_path>`,
it will take precedence over this environment variable.


.. _train-checkpointing:

Saving and Loading Checkpoints
==============================

Ray Train provides a way to snapshot training progress with :class:`Checkpoints <ray.train.Checkpoint>`.

This is useful for:

1. **Storing the best-performing model weights:** Save your model to persistent storage, and use it for downstream serving/inference.
2. **Fault tolerance:** Handle node failures in a long-running training job on a cluster of pre-emptible machines/pods.
3. **Distributed checkpointing:** When doing *model-parallel training*, Ray Train checkpointing provides an easy way to
   :ref:`upload model shards from each worker in parallel <train-distributed-checkpointing>`,
   without needing to gather the full model to a single node.
4. **Integration with Ray Tune:** Checkpoint saving and loading is required by certain :ref:`Ray Tune schedulers <tune-schedulers>`.


.. _train-dl-saving-checkpoints:

Saving checkpoints during training
----------------------------------

The :class:`Checkpoint <ray.train.Checkpoint>` is a lightweight interface provided
by Ray Train that represents a *directory* that exists on local or remote storage.

For example, a checkpoint could point to a directory in cloud storage:
``s3://my-bucket/my-checkpoint-dir``.
A locally available checkpoint points to a location on the local filesystem:
``/tmp/my-checkpoint-dir``.

Here's how you save a checkpoint in the training loop:

1. Write your model checkpoint to a local directory.

   - Since a :class:`Checkpoint <ray.train.Checkpoint>` just points to a directory, the contents are completely up to you.
   - This means that you can use any serialization format you want.
   - This makes it **easy to use familiar checkpoint utilities provided by training frameworks**, such as
     ``torch.save``, ``pl.Trainer.save_checkpoint``, Accelerate's ``accelerator.save_model``,
     Transformers' ``save_pretrained``, ``tf.keras.Model.save``, etc.

2. Create a :class:`Checkpoint <ray.train.Checkpoint>` from the directory using :meth:`Checkpoint.from_directory <ray.train.Checkpoint.from_directory>`.

3. Report the checkpoint to Ray Train using :func:`ray.train.report(metrics, checkpoint=...) <ray.train.report>`.

   - The metrics reported alongside the checkpoint are used to :ref:`keep track of the best-performing checkpoints <train-dl-configure-checkpoints>`.
   - This will **upload the checkpoint to persistent storage** if configured. See :ref:`persistent-storage-guide`.


.. figure:: ../images/checkpoint_lifecycle.png

    The lifecycle of a :class:`~ray.train.Checkpoint`, from being saved locally
    to disk to being uploaded to persistent storage via ``train.report``.

As shown in the figure above, the best practice for saving checkpoints is to
first dump the checkpoint to a local temporary directory. Then, the call to ``train.report``
uploads the checkpoint to its final persistent storage location.
Then, the local temporary directory can be safely cleaned up to free up disk space
(e.g., from exiting the ``tempfile.TemporaryDirectory`` context).

.. tip::

    In standard DDP training, where each worker has a copy of the full-model, you should
    only save and report a checkpoint from a single worker to prevent redundant uploads.

    This typically looks like:

    .. literalinclude::  ../doc_code/checkpoints.py
        :language: python
        :start-after: __checkpoint_from_single_worker_start__
        :end-before: __checkpoint_from_single_worker_end__

    If using parallel training strategies such as DeepSpeed Zero-3 and FSDP, where
    each worker only has a shard of the full-model, you should save and report a checkpoint
    from each worker. See :ref:`train-distributed-checkpointing` for an example.


Here are a few examples of saving checkpoints with different training frameworks:

.. tab-set::

    .. tab-item:: Native PyTorch

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __pytorch_save_start__
            :end-before: __pytorch_save_end__

        .. tip::

            You most likely want to unwrap the DDP model before saving it to a checkpoint.
            ``model.module.state_dict()`` is the state dict without each key having a ``"module."`` prefix.


    .. tab-item:: PyTorch Lightning

        Ray Train leverages PyTorch Lightning's ``Callback`` interface to report metrics
        and checkpoints. We provide a simple callback implementation that reports
        ``on_train_epoch_end``.

        Specifically, on each train epoch end, it

        - collects all the logged metrics from ``trainer.callback_metrics``
        - saves a checkpoint via ``trainer.save_checkpoint``
        - reports to Ray Train via :func:`ray.train.report(metrics, checkpoint) <ray.train.report>`

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __lightning_save_example_start__
            :end-before: __lightning_save_example_end__

        You can always get the saved checkpoint path from :attr:`result.checkpoint <ray.train.Result.checkpoint>` and
        :attr:`result.best_checkpoints <ray.train.Result.best_checkpoints>`.

        For more advanced usage (e.g. reporting at different frequency, reporting
        customized checkpoint files), you can implement your own customized callback.
        Here is a simple example that reports a checkpoint every 3 epochs:

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __lightning_custom_save_example_start__
            :end-before: __lightning_custom_save_example_end__


    .. tab-item:: Hugging Face Transformers

        Ray Train leverages HuggingFace Transformers Trainer's ``Callback`` interface
        to report metrics and checkpoints.

        **Option 1: Use Ray Train's default report callback**

        We provide a simple callback implementation :class:`~ray.train.huggingface.transformers.RayTrainReportCallback` that
        reports on checkpoint save. You can change the checkpointing frequency by ``save_strategy`` and ``save_steps``.
        It collects the latest logged metrics and report them together with the latest saved checkpoint.

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __transformers_save_example_start__
            :end-before: __transformers_save_example_end__

        Note that :class:`~ray.train.huggingface.transformers.RayTrainReportCallback`
        binds the latest metrics and checkpoints together,
        so users can properly configure ``logging_strategy``, ``save_strategy`` and ``evaluation_strategy``
        to ensure the monitoring metric is logged at the same step as checkpoint saving.

        For example, the evaluation metrics (``eval_loss`` in this case) are logged during
        evaluation. If users want to keep the best 3 checkpoints according to ``eval_loss``, they
        should align the saving and evaluation frequency. Below are two examples of valid configurations:

        .. testcode::
            :skipif: True

            args = TrainingArguments(
                ...,
                evaluation_strategy="epoch",
                save_strategy="epoch",
            )

            args = TrainingArguments(
                ...,
                evaluation_strategy="steps",
                save_strategy="steps",
                eval_steps=50,
                save_steps=100,
            )

            # And more ...


        **Option 2: Implement your customized report callback**

        If you feel that Ray Train's default :class:`~ray.train.huggingface.transformers.RayTrainReportCallback`
        is not sufficient for your use case, you can also implement a callback yourself!
        Below is a example implementation that collects latest metrics
        and reports on checkpoint save.

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __transformers_custom_save_example_start__
            :end-before: __transformers_custom_save_example_end__


        You can customize when (``on_save``, ``on_epoch_end``, ``on_evaluate``) and
        what (customized metrics and checkpoint files) to report by implementing your own
        Transformers Trainer callback.


.. _train-distributed-checkpointing:

Saving checkpoints from multiple workers (distributed checkpointing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In model parallel training strategies where each worker only has a shard of the full-model,
you can save and report checkpoint shards in parallel from each worker.

.. figure:: ../images/persistent_storage_checkpoint.png

    Distributed checkpointing in Ray Train. Each worker uploads its own checkpoint shard
    to persistent storage independently.

Distributed checkpointing is the best practice for saving checkpoints
when doing model-parallel training (e.g., DeepSpeed, FSDP, Megatron-LM).

There are two major benefits:

1. **It is faster, resulting in less idle time.** Faster checkpointing incentivizes more frequent checkpointing!

   Each worker can upload its checkpoint shard in parallel,
   maximizing the network bandwidth of the cluster. Instead of a single node
   uploading the full model of size ``M``, the cluster distributes the load across
   ``N`` nodes, each uploading a shard of size ``M / N``.

2. **Distributed checkpointing avoids needing to gather the full model onto a single worker's CPU memory.**

   This gather operation puts a large CPU memory requirement on the worker that performs checkpointing
   and is a common source of OOM errors.


Here is an example of distributed checkpointing with PyTorch:

.. literalinclude:: ../doc_code/checkpoints.py
    :language: python
    :start-after: __distributed_checkpointing_start__
    :end-before: __distributed_checkpointing_end__


.. note::

    Checkpoint files with the same name will collide between workers.
    You can get around this by adding a rank-specific suffix to checkpoint files.

    Note that having filename collisions does not error, but it will result in the last
    uploaded version being the one that is persisted. This is fine if the file
    contents are the same across all workers.

    Model shard saving utilities provided by frameworks such as DeepSpeed will create
    rank-specific filenames already, so you usually do not need to worry about this.


.. _train-dl-configure-checkpoints:

Configure checkpointing
-----------------------

Ray Train provides some configuration options for checkpointing via :class:`~ray.train.CheckpointConfig`.
The primary configuration is keeping only the top ``K`` checkpoints with respect to a metric.
Lower-performing checkpoints are deleted to save storage space. By default, all checkpoints are kept.

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __checkpoint_config_start__
    :end-before: __checkpoint_config_end__


.. note::

    If you want to save the top ``num_to_keep`` checkpoints with respect to a metric via
    :py:class:`~ray.train.CheckpointConfig`,
    please ensure that the metric is always reported together with the checkpoints.



Using checkpoints after training
--------------------------------

The latest saved checkpoint can be accessed with :attr:`Result.checkpoint <ray.train.Result.checkpoint>`.

The full list of persisted checkpoints can be accessed with :attr:`Result.best_checkpoints <ray.train.Result.best_checkpoints>`.
If :class:`CheckpointConfig(num_to_keep) <ray.train.CheckpointConfig>` is set, this list will contain the best ``num_to_keep`` checkpoints.

See :ref:`train-inspect-results` for a full guide on inspecting training results.

:meth:`Checkpoint.as_directory <ray.train.Checkpoint.as_directory>`
and :meth:`Checkpoint.to_directory <ray.train.Checkpoint.to_directory>`
are the two main APIs to interact with Train checkpoints:

.. literalinclude:: ../doc_code/checkpoints.py
    :language: python
    :start-after: __inspect_checkpoint_example_start__
    :end-before: __inspect_checkpoint_example_end__


For Lightning and Transformers, if you are using the default `RayTrainReportCallback` for checkpoint saving in your training function, 
you can retrieve the original checkpoint files as below:

.. tab-set::

    .. tab-item:: PyTorch Lightning

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __inspect_lightning_checkpoint_example_start__
            :end-before: __inspect_lightning_checkpoint_example_end__
    
    .. tab-item:: Transformers

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __inspect_transformers_checkpoint_example_start__
            :end-before: __inspect_transformers_checkpoint_example_end__


.. _train-dl-loading-checkpoints:

Restore training state from a checkpoint
----------------------------------------

In order to enable fault tolerance, you should modify your training loop to restore
training state from a :class:`~ray.train.Checkpoint`.

The :class:`Checkpoint <ray.train.Checkpoint>` to restore from can be accessed in the
training function with :func:`ray.train.get_checkpoint <ray.train.get_checkpoint>`.

The checkpoint returned by :func:`ray.train.get_checkpoint <ray.train.get_checkpoint>` is populated in two ways:

1. It can be auto-populated as the latest reported checkpoint, e.g. during :ref:`automatic failure recovery <train-fault-tolerance>` or :ref:`on manual restoration <train-restore-guide>`.
2. It can be manually populated by passing a checkpoint to the ``resume_from_checkpoint`` argument of a Ray :class:`Trainer <ray.train.trainer.BaseTrainer>`.
   This is useful for initializing a new training run with a previous run's checkpoint.


.. tab-set::

    .. tab-item:: Native PyTorch

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __pytorch_restore_start__
            :end-before: __pytorch_restore_end__


    .. tab-item:: PyTorch Lightning

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __lightning_restore_example_start__
            :end-before: __lightning_restore_example_end__


.. note::

    In these examples, :meth:`Checkpoint.as_directory <ray.train.Checkpoint.as_directory>`
    is used to view the checkpoint contents as a local directory.

    *If the checkpoint points to a local directory*, this method just returns the
    local directory path without making a copy.

    *If the checkpoint points to a remote directory*, this method will download the
    checkpoint to a local temporary directory and return the path to the temporary directory.

    **If multiple processes on the same node call this method simultaneously,**
    only a single process will perform the download, while the others
    wait for the download to finish. Once the download finishes, all processes receive
    the same local (temporary) directory to read from.

    Once all processes have finished working with the checkpoint, the temporary directory
    is cleaned up.


.. _train-reproducibility:

Reproducibility
---------------

.. tab-set::

    .. tab-item:: PyTorch

        To limit sources of nondeterministic behavior, add
        :func:`ray.train.torch.enable_reproducibility` to the top of your training
        function.

        .. code-block:: diff

             def train_func():
            +    train.torch.enable_reproducibility()

                 model = NeuralNetwork()
                 model = train.torch.prepare_model(model)

                 ...

        .. warning:: :func:`ray.train.torch.enable_reproducibility` can't guarantee
            completely reproducible results across executions. To learn more, read
            the `PyTorch notes on randomness <https://pytorch.org/docs/stable/notes/randomness.html>`_.

..
    import ray
    from ray import tune

    def training_func(config):
        dataloader = ray.train.get_dataset()\
            .get_shard(torch.rank())\
            .iter_torch_batches(batch_size=config["batch_size"])

        for i in config["epochs"]:
            ray.train.report(...)  # use same intermediate reporting API

    # Declare the specification for training.
    trainer = Trainer(backend="torch", num_workers=12, use_gpu=True)
    dataset = ray.dataset.window()

    # Convert this to a trainable.
    trainable = trainer.to_tune_trainable(training_func, dataset=dataset)

    tuner = tune.Tuner(trainable,
        param_space={"lr": tune.uniform(), "batch_size": tune.randint(1, 2, 3)},
        tune_config=tune.TuneConfig(num_samples=12))
    results = tuner.fit()


.. _train-inspect-results:

Inspecting Training Results
===========================

The return value of your :meth:`Trainer.fit() <ray.train.trainer.BaseTrainer.fit>`
call is a :class:`~ray.train.Result` object.

The :class:`~ray.train.Result` object contains, among other information:

- The last reported metrics (e.g. the loss)
- The last reported checkpoint (to load the model)
- Error messages, if any errors occurred

Viewing metrics
---------------
You can retrieve metrics reported to Ray Train from the :class:`~ray.train.Result`
object.

Common metrics include the training or validation loss, or prediction accuracies.

The metrics retrieved from the :class:`~ray.train.Result` object
correspond to those you passed to :func:`train.report <ray.train.report>`
as an argument :ref:`in your training function <train-monitoring-and-logging>`.


Last reported metrics
~~~~~~~~~~~~~~~~~~~~~

Use :attr:`Result.metrics <ray.train.Result.metrics>` to retrieve the
latest reported metrics.

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_metrics_start__
    :end-before: __result_metrics_end__

Dataframe of all reported metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :attr:`Result.metrics_dataframe <ray.train.Result.metrics_dataframe>` to retrieve
a pandas DataFrame of all reported metrics.

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_dataframe_start__
    :end-before: __result_dataframe_end__


Retrieving checkpoints
----------------------
You can retrieve checkpoints reported to Ray Train from the :class:`~ray.train.Result`
object.

:ref:`Checkpoints <train-checkpointing>` contain all the information that is needed
to restore the training state. This usually includes the trained model.

You can use checkpoints for common downstream tasks such as
:doc:`offline batch inference with Ray Data </data/data>` or 
:doc:`online model serving with Ray Serve </serve/index>`.

The checkpoints retrieved from the :class:`~ray.train.Result` object
correspond to those you passed to :func:`train.report <ray.train.report>`
as an argument :ref:`in your training function <train-monitoring-and-logging>`.

Last saved checkpoint
~~~~~~~~~~~~~~~~~~~~~
Use :attr:`Result.checkpoint <ray.train.Result.checkpoint>` to retrieve the
last checkpoint.

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_checkpoint_start__
    :end-before: __result_checkpoint_end__


Other checkpoints
~~~~~~~~~~~~~~~~~
Sometimes you want to access an earlier checkpoint. For instance, if your loss increased
after more training due to overfitting, you may want to retrieve the checkpoint with
the lowest loss.

You can retrieve a list of all available checkpoints and their metrics with
:attr:`Result.best_checkpoints <ray.train.Result.best_checkpoints>`

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_best_checkpoint_start__
    :end-before: __result_best_checkpoint_end__

.. seealso::

    See :ref:`train-checkpointing` for more information on checkpointing.

Accessing storage location
---------------------------
If you need to retrieve the results later, you can get the storage location
of the training run with :attr:`Result.path <ray.train.Result.path>`.

This path will correspond to the :ref:`storage_path <train-log-dir>` you configured
in the :class:`~ray.train.RunConfig`. It will be a
(nested) subdirectory within that path, usually
of the form `TrainerName_date-string/TrainerName_id_00000_0_...`.

The result also contains a :class:`pyarrow.fs.FileSystem` that can be used to
access the storage location, which is useful if the path is on cloud storage.


.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_path_start__
    :end-before: __result_path_end__


You can restore a result with :meth:`Result.from_path <ray.train.Result.from_path>`:

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_restore_start__
    :end-before: __result_restore_end__



Viewing Errors
--------------
If an error occurred during training,
:attr:`Result.error <ray.train.Result.error>` will be set and contain the exception
that was raised.

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_error_start__
    :end-before: __result_error_end__


Finding results on persistent storage
-------------------------------------
All training results, including reported metrics, checkpoints, and error files,
are stored on the configured :ref:`persistent storage <train-log-dir>`.

See :ref:`our persistent storage guide <train-log-dir>` to configure this location
for your training run.


.. _data-ingest-torch:

Data Loading and Preprocessing
==============================

Ray Train integrates with :ref:`Ray Data <data>` to offer a performant and scalable streaming solution for loading and preprocessing large datasets.
Key advantages include:

- Streaming data loading and preprocessing, scalable to petabyte-scale data.
- Scaling out heavy data preprocessing to CPU nodes, to avoid bottlenecking GPU training.
- Automatic and fast failure recovery.
- Automatic on-the-fly data splitting across distributed training workers.

For more details about Ray Data, including comparisons to alternatives, see :ref:`Ray Data Overview <data_overview>`.

In this guide, we will cover how to incorporate Ray Data into your Ray Train script, and different ways to customize your data ingestion pipeline.

.. TODO: Replace this image with a better one.

.. figure:: ../images/train_ingest.png
    :align: center
    :width: 300px

Quickstart
----------
Install Ray Data and Ray Train:

.. code-block:: bash

    pip install -U "ray[data,train]"

Data ingestion can be set up with four basic steps:

1. Create a Ray Dataset from your input data.
2. Apply preprocessing operations to your Ray Dataset.
3. Input the preprocessed Dataset into the Ray Train Trainer, which internally splits the dataset equally in a streaming way across the distributed training workers.
4. Consume the Ray Dataset in your training function.

.. tab-set::

    .. tab-item:: PyTorch

        .. code-block:: python
            :emphasize-lines: 14,21,29,31-33,53

            import torch
            import ray
            from ray import train
            from ray.train import Checkpoint, ScalingConfig
            from ray.train.torch import TorchTrainer

            # Set this to True to use GPU.
            # If False, do CPU training instead of GPU training.
            use_gpu = False

            # Step 1: Create a Ray Dataset from in-memory Python lists.
            # You can also create a Ray Dataset from many other sources and file
            # formats.
            train_dataset = ray.data.from_items([{"x": [x], "y": [2 * x]} for x in range(200)])

            # Step 2: Preprocess your Ray Dataset.
            def increment(batch):
                batch["y"] = batch["y"] + 1
                return batch

            train_dataset = train_dataset.map_batches(increment)


            def train_func():
                batch_size = 16

                # Step 4: Access the dataset shard for the training worker via
                # ``get_dataset_shard``.
                train_data_shard = train.get_dataset_shard("train")
                # `iter_torch_batches` returns an iterable object that
                # yield tensor batches. Ray Data automatically moves the Tensor batches
                # to GPU if you enable GPU training.
                train_dataloader = train_data_shard.iter_torch_batches(
                    batch_size=batch_size, dtypes=torch.float32
                )

                for epoch_idx in range(1):
                    for batch in train_dataloader:
                        inputs, labels = batch["x"], batch["y"]
                        assert type(inputs) == torch.Tensor
                        assert type(labels) == torch.Tensor
                        assert inputs.shape[0] == batch_size
                        assert labels.shape[0] == batch_size
                        # Only check one batch for demo purposes.
                        # Replace the above with your actual model training code.
                        break

            # Step 3: Create a TorchTrainer. Specify the number of training workers and
            # pass in your Ray Dataset.
            # The Ray Dataset is automatically split across all training workers.
            trainer = TorchTrainer(
                train_func,
                datasets={"train": train_dataset},
                scaling_config=ScalingConfig(num_workers=2, use_gpu=use_gpu)
            )
            result = trainer.fit()

    .. tab-item:: PyTorch Lightning

        .. code-block:: python
            :emphasize-lines: 4-5,10-11,14-15,26-27,33

            from ray import train

            # Create the train and validation datasets.
            train_data = ray.data.read_csv("./train.csv")
            val_data = ray.data.read_csv("./validation.csv")

            def train_func_per_worker():
                # Access Ray datsets in your train_func via ``get_dataset_shard``.
                # Ray Data shards all datasets across workers by default.
                train_ds = train.get_dataset_shard("train")
                val_ds = train.get_dataset_shard("validation")

                # Create Ray dataset iterables via ``iter_torch_batches``.
                train_dataloader = train_ds.iter_torch_batches(batch_size=16)
                val_dataloader = val_ds.iter_torch_batches(batch_size=16)

                ...

                trainer = pl.Trainer(
                    # ...
                )

                # Feed the Ray dataset iterables to ``pl.Trainer.fit``.
                trainer.fit(
                    model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader
                )

            trainer = TorchTrainer(
                train_func,
                # You can pass in multiple datasets to the Trainer.
                datasets={"train": train_data, "validation": val_data},
                scaling_config=ScalingConfig(num_workers=4),
            )
            trainer.fit()

    .. tab-item:: HuggingFace Transformers

        .. code-block:: python
            :emphasize-lines: 7-8,13-14,17-18,30-31,41

            import ray
            import ray.train

            ...

            # Create the train and evaluation datasets.
            train_data = ray.data.from_huggingface(hf_train_ds)
            eval_data = ray.data.from_huggingface(hf_eval_ds)

            def train_func():
                # Access Ray datsets in your train_func via ``get_dataset_shard``.
                # Ray Data shards all datasets across workers by default.
                train_ds = ray.train.get_dataset_shard("train")
                eval_ds = ray.train.get_dataset_shard("evaluation")

                # Create Ray dataset iterables via ``iter_torch_batches``.
                train_iterable_ds = train_ds.iter_torch_batches(batch_size=16)
                eval_iterable_ds = eval_ds.iter_torch_batches(batch_size=16)

                ...

                args = transformers.TrainingArguments(
                    ...,
                    max_steps=max_steps # Required for iterable datasets
                )

                trainer = transformers.Trainer(
                    ...,
                    model=model,
                    train_dataset=train_iterable_ds,
                    eval_dataset=eval_iterable_ds,
                )

                # Prepare your Transformers Trainer
                trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
                trainer.train()

            trainer = TorchTrainer(
                train_func,
                # You can pass in multiple datasets to the Trainer.
                datasets={"train": train_data, "evaluation": val_data},
                scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
            )
            trainer.fit()


.. _train-datasets-load:

Loading data
~~~~~~~~~~~~

Ray Datasets can be created from many different data sources and formats. For more details, see :ref:`Loading Data <loading_data>`.

.. _train-datasets-preprocess:

Preprocessing data
~~~~~~~~~~~~~~~~~~

Ray Data supports a wide range of preprocessing operations that you can use to transform data prior to training.

- For general preprocessing, see :ref:`Transforming Data <transforming_data>`.
- For tabular data, see :ref:`Preprocessing Structured Data <preprocessing_structured_data>`.
- For PyTorch tensors, see :ref:`Transformations with torch tensors <transform_pytorch>`.
- For optimizing expensive preprocessing operations, see :ref:`Caching the preprocessed dataset <dataset_cache_performance>`.

.. _train-datasets-input:

Inputting and splitting data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your preprocessed datasets can be passed into a Ray Train Trainer (e.g. :class:`~ray.train.torch.TorchTrainer`) through the ``datasets`` argument.

The datasets passed into the Trainer's ``datasets`` can be accessed inside of the ``train_loop_per_worker`` run on each distributed training worker by calling :meth:`ray.train.get_dataset_shard`.

Ray Data splits all datasets across the training workers by default. :meth:`~ray.train.get_dataset_shard` returns ``1/n`` of the dataset, where ``n`` is the number of training workers.

Ray Data does data splitting in a streaming fashion on the fly.

.. note::

    Be aware that because Ray Data splits the evaluation dataset, you have to aggregate the evaluation results across workers.
    You might consider using `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/>`_ (:doc:`example <../examples/deepspeed/deepspeed_example>`) or
    utilities available in other frameworks that you can explore.

This behavior can be overwritten by passing in the ``dataset_config`` argument. For more information on configuring splitting logic, see :ref:`Splitting datasets <train-datasets-split>`.

.. _train-datasets-consume:

Consuming data
~~~~~~~~~~~~~~

Inside the ``train_loop_per_worker``, each worker can access its shard of the dataset via :meth:`ray.train.get_dataset_shard`.

This data can be consumed in a variety of ways:

- To create a generic Iterable of batches, you can call :meth:`~ray.data.DataIterator.iter_batches`.
- To create a replacement for a PyTorch DataLoader, you can call :meth:`~ray.data.DataIterator.iter_torch_batches`.

For more details on how to iterate over your data, see :ref:`Iterating over data <iterating-over-data>`.

.. _train-datasets-pytorch:

Starting with PyTorch data
--------------------------

Some frameworks provide their own dataset and data loading utilities. For example:

- **PyTorch:** `Dataset & DataLoader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_
- **Hugging Face:** `Dataset <https://huggingface.co/docs/datasets/index>`_
- **PyTorch Lightning:** `LightningDataModule <https://lightning.ai/docs/pytorch/stable/data/datamodule.html>`_

These utilities can still be used directly with Ray Train. In particular, you may want to do this if you already have your data ingestion pipeline set up.
However, for more performant large-scale data ingestion we do recommend migrating to Ray Data.

At a high level, you can compare these concepts as follows:

.. list-table::
   :header-rows: 1

   * - PyTorch API
     - HuggingFace API
     - Ray Data API
   * - `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_
     - `datasets.Dataset <https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset>`_
     - :class:`ray.data.Dataset`
   * - `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
     - n/a
     - :meth:`ray.data.Dataset.iter_torch_batches`


For more details, see the following sections for each framework.

.. tab-set::

    .. tab-item:: PyTorch Dataset and DataLoader

        **Option 1 (with Ray Data):** Convert your PyTorch Dataset to a Ray Dataset and pass it into the Trainer via  ``datasets`` argument.
        Inside your ``train_loop_per_worker``, you can access the dataset via :meth:`ray.train.get_dataset_shard`.
        You can convert this to replace the PyTorch DataLoader via :meth:`ray.data.DataIterator.iter_torch_batches`.

        For more details, see the :ref:`Migrating from PyTorch Datasets and DataLoaders <migrate_pytorch>`.

        **Option 2 (without Ray Data):** Instantiate the Torch Dataset and DataLoader directly in the ``train_loop_per_worker``.
        You can use the :meth:`ray.train.torch.prepare_data_loader` utility to set up the DataLoader for distributed training.

    .. tab-item:: LightningDataModule

        The ``LightningDataModule`` is created with PyTorch ``Dataset``\s and ``DataLoader``\s. You can apply the same logic here.

    .. tab-item:: Hugging Face Dataset

        **Option 1 (with Ray Data):** Convert your Hugging Face Dataset to a Ray Dataset and pass it into the Trainer via the ``datasets`` argument.
        Inside your ``train_loop_per_worker``, you can access the dataset via :meth:`ray.train.get_dataset_shard`.

        For instructions, see :ref:`Ray Data for Hugging Face <loading_datasets_from_ml_libraries>`.

        **Option 2 (without Ray Data):** Instantiate the Hugging Face Dataset directly in the ``train_loop_per_worker``.

.. tip::

    When using Torch or Hugging Face Datasets directly without Ray Data, make sure to instantiate your Dataset *inside* the ``train_loop_per_worker``.
    Instatiating the Dataset outside of the ``train_loop_per_worker`` and passing it in via global scope
    can cause errors due to the Dataset not being serializable.

.. _train-datasets-split:

Splitting datasets
------------------
By default, Ray Train splits all datasets across workers using :meth:`Dataset.streaming_split <ray.data.Dataset.streaming_split>`. Each worker sees a disjoint subset of the data, instead of iterating over the entire dataset.

If want to customize which datasets are split, pass in a :class:`DataConfig <ray.train.DataConfig>` to the Trainer constructor.

For example, to split only the training dataset, do the following:

.. testcode::

    import ray
    from ray import train
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    ds = ray.data.read_text(
        "s3://anonymous@ray-example-data/sms_spam_collection_subset.txt"
    )
    train_ds, val_ds = ds.train_test_split(0.3)

    def train_loop_per_worker():
        # Get the sharded training dataset
        train_ds = train.get_dataset_shard("train")
        for _ in range(2):
            for batch in train_ds.iter_batches(batch_size=128):
                print("Do some training on batch", batch)

        # Get the unsharded full validation dataset
        val_ds = train.get_dataset_shard("val")
        for _ in range(2):
            for batch in val_ds.iter_batches(batch_size=128):
                print("Do some evaluation on batch", batch)

    my_trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=ray.train.DataConfig(
            datasets_to_split=["train"],
        ),
    )
    my_trainer.fit()


Full customization (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For use cases not covered by the default config class, you can also fully customize exactly how your input datasets are split. Define a custom :class:`DataConfig <ray.train.DataConfig>` class (DeveloperAPI). The :class:`DataConfig <ray.train.DataConfig>` class is responsible for that shared setup and splitting of data across nodes.

.. testcode::

    # Note that this example class is doing the same thing as the basic DataConfig
    # implementation included with Ray Train.
    from typing import Optional, Dict, List

    import ray
    from ray import train
    from ray.train.torch import TorchTrainer
    from ray.train import DataConfig, ScalingConfig
    from ray.data import Dataset, DataIterator, NodeIdStr
    from ray.actor import ActorHandle

    ds = ray.data.read_text(
        "s3://anonymous@ray-example-data/sms_spam_collection_subset.txt"
    )

    def train_loop_per_worker():
        # Get an iterator to the dataset we passed in below.
        it = train.get_dataset_shard("train")
        for _ in range(2):
            for batch in it.iter_batches(batch_size=128):
                print("Do some training on batch", batch)


    class MyCustomDataConfig(DataConfig):
        def configure(
            self,
            datasets: Dict[str, Dataset],
            world_size: int,
            worker_handles: Optional[List[ActorHandle]],
            worker_node_ids: Optional[List[NodeIdStr]],
            **kwargs,
        ) -> List[Dict[str, DataIterator]]:
            assert len(datasets) == 1, "This example only handles the simple case"

            # Configure Ray Data for ingest.
            ctx = ray.data.DataContext.get_current()
            ctx.execution_options = DataConfig.default_ingest_options()

            # Split the stream into shards.
            iterator_shards = datasets["train"].streaming_split(
                world_size, equal=True, locality_hints=worker_node_ids
            )

            # Return the assigned iterators for each worker.
            return [{"train": it} for it in iterator_shards]


    my_trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": ds},
        dataset_config=MyCustomDataConfig(),
    )
    my_trainer.fit()


The subclass must be serializable, since Ray Train copies it from the driver script to the driving actor of the Trainer. Ray Train calls its :meth:`configure <ray.train.DataConfig.configure>` method on the main actor of the Trainer group to create the data iterators for each worker.

In general, you can use :class:`DataConfig <ray.train.DataConfig>` for any shared setup that has to occur ahead of time before the workers start iterating over data. The setup runs at the start of each Trainer run.


Random shuffling
----------------
Randomly shuffling data for each epoch can be important for model quality depending on what model you are training.

Ray Data provides multiple options for random shuffling, see :ref:`Shuffling Data <shuffling_data>` for more details.

Enabling reproducibility
------------------------
When developing or hyperparameter tuning models, reproducibility is important during data ingest so that data ingest does not affect model quality. Follow these three steps to enable reproducibility:

**Step 1:** Enable deterministic execution in Ray Datasets by setting the `preserve_order` flag in the :class:`DataContext <ray.data.context.DataContext>`.

.. testcode::

    import ray

    # Preserve ordering in Ray Datasets for reproducibility.
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = True

    ds = ray.data.read_text(
        "s3://anonymous@ray-example-data/sms_spam_collection_subset.txt"
    )

**Step 2:** Set a seed for any shuffling operations:

* `seed` argument to :meth:`random_shuffle <ray.data.Dataset.random_shuffle>`
* `seed` argument to :meth:`randomize_block_order <ray.data.Dataset.randomize_block_order>`
* `local_shuffle_seed` argument to :meth:`iter_batches <ray.data.DataIterator.iter_batches>`

**Step 3:** Follow the best practices for enabling reproducibility for your training framework of choice. For example, see the `Pytorch reproducibility guide <https://pytorch.org/docs/stable/notes/randomness.html>`_.



.. _preprocessing_structured_data:

Preprocessing structured data
-----------------------------

.. note::
    This section is for tabular/structured data. The recommended way for preprocessing unstructured data is to use
    Ray Data operations such as `map_batches`. See the :ref:`Ray Data Working with Pytorch guide <working_with_pytorch>` for more details.

For tabular data, use Ray Data :ref:`preprocessors <preprocessor-ref>`, which implement common data preprocessing operations.
You can use this with Ray Train Trainers by applying them on the dataset before passing the dataset into a Trainer. For example:

.. testcode::

    import numpy as np
    from tempfile import TemporaryDirectory

    import ray
    from ray import train
    from ray.train import Checkpoint, ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray.data.preprocessors import Concatenator, StandardScaler

    dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")

    # Create preprocessors to scale some columns and concatenate the results.
    scaler = StandardScaler(columns=["mean radius", "mean texture"])
    concatenator = Concatenator(exclude=["target"], dtype=np.float32)

    # Compute dataset statistics and get transformed datasets. Note that the
    # fit call is executed immediately, but the transformation is lazy.
    dataset = scaler.fit_transform(dataset)
    dataset = concatenator.fit_transform(dataset)

    def train_loop_per_worker():
        context = train.get_context()
        print(context.get_metadata())  # prints {"preprocessor_pkl": ...}

        # Get an iterator to the dataset we passed in below.
        it = train.get_dataset_shard("train")
        for _ in range(2):
            # Prefetch 10 batches at a time.
            for batch in it.iter_batches(batch_size=128, prefetch_batches=10):
                print("Do some training on batch", batch)

        # Save a checkpoint.
        with TemporaryDirectory() as temp_dir:
            train.report(
                {"score": 2.0},
                checkpoint=Checkpoint.from_directory(temp_dir),
            )

    my_trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": dataset},
        metadata={"preprocessor_pkl": scaler.serialize()},
    )

    # Get the fitted preprocessor back from the result metadata.
    metadata = my_trainer.fit().checkpoint.get_metadata()
    print(StandardScaler.deserialize(metadata["preprocessor_pkl"]))


This example persists the fitted preprocessor using the ``Trainer(metadata={...})`` constructor argument. This arg specifies a dict that is available from ``TrainContext.get_metadata()`` and ``checkpoint.get_metadata()`` for checkpoints that the Trainer saves. This design enables the recreation of the fitted preprocessor for inference.

Performance tips
----------------

Prefetching batches
~~~~~~~~~~~~~~~~~~~
While iterating over a dataset for training, you can increase ``prefetch_batches`` in :meth:`iter_batches <ray.data.DataIterator.iter_batches>` or :meth:`iter_torch_batches <ray.data.DataIterator.iter_torch_batches>` to further increase performance. While training on the current batch, this approach launches background threads to fetch and process the next ``N`` batches.

This approach can help if training is bottlenecked on cross-node data transfer or on last-mile preprocessing such as converting batches to tensors or executing ``collate_fn``. However, increasing ``prefetch_batches`` leads to more data that needs to be held in heap memory. By default, ``prefetch_batches`` is set to 1.

For example, the following code prefetches 10 batches at a time for each training worker:

.. testcode::

    import ray
    from ray import train
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    ds = ray.data.read_text(
        "s3://anonymous@ray-example-data/sms_spam_collection_subset.txt"
    )

    def train_loop_per_worker():
        # Get an iterator to the dataset we passed in below.
        it = train.get_dataset_shard("train")
        for _ in range(2):
            # Prefetch 10 batches at a time.
            for batch in it.iter_batches(batch_size=128, prefetch_batches=10):
                print("Do some training on batch", batch)

    my_trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": ds},
    )
    my_trainer.fit()

Avoid heavy transformation in collate_fn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``collate_fn`` parameter in :meth:`iter_batches <ray.data.DataIterator.iter_batches>` or :meth:`iter_torch_batches <ray.data.DataIterator.iter_torch_batches>` allows you to transform data before feeding it to the model. This operation happens locally in the training workers. Avoid adding a heavy transformation in this function as it may become the bottleneck. Instead, :ref:`apply the transformation with map or map_batches <transforming_data>` before passing the dataset to the Trainer.


.. _dataset_cache_performance:

Caching the preprocessed dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If your preprocessed Dataset is small enough to fit in Ray object store memory (by default this is 30% of total cluster RAM), *materialize* the preprocessed dataset in Ray's built-in object store, by calling :meth:`materialize() <ray.data.Dataset.materialize>` on the preprocessed dataset. This method tells Ray Data to compute the entire preprocessed and pin it in the Ray object store memory. As a result, when iterating over the dataset repeatedly, the preprocessing operations do not need to be re-run. However, if the preprocessed data is too large to fit into Ray object store memory, this approach will greatly decreases performance as data needs to be spilled to and read back from disk.

Transformations that you want to run per-epoch, such as randomization, should go after the materialize call.

.. testcode::

    from typing import Dict
    import numpy as np
    import ray

    # Load the data.
    train_ds = ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet")

    # Define a preprocessing function.
    def normalize_length(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        new_col = batch["sepal.length"] / np.max(batch["sepal.length"])
        batch["normalized.sepal.length"] = new_col
        del batch["sepal.length"]
        return batch

    # Preprocess the data. Transformations that are made before the materialize call
    # below are only run once.
    train_ds = train_ds.map_batches(normalize_length)

    # Materialize the dataset in object store memory.
    # Only do this if train_ds is small enough to fit in object store memory.
    train_ds = train_ds.materialize()

    # Dummy augmentation transform.
    def augment_data(batch):
        return batch

    # Add per-epoch preprocessing. Transformations that you want to run per-epoch, such
    # as data augmentation or randomization, should go after the materialize call.
    train_ds = train_ds.map_batches(augment_data)

    # Pass train_ds to the Trainer


Adding CPU-only nodes to your cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the GPU training is bottlenecked on expensive CPU preprocessing and the preprocessed Dataset is too large to fit in object store memory, then materializing the dataset doesn't work. In this case, Ray's native support for heterogeneous resources enables you to simply add more CPU-only nodes to your cluster, and Ray Data automatically scales out CPU-only preprocessing tasks to CPU-only nodes, making GPUs more saturated.

In general, adding CPU-only nodes can help in two ways:
* Adding more CPU cores helps further parallelize preprocessing. This approach is helpful when CPU compute time is the bottleneck.
* Increasing object store memory, which 1) allows Ray Data to buffer more data in between preprocessing and training stages, and 2) provides more memory to make it possible to :ref:`cache the preprocessed dataset <dataset_cache_performance>`. This approach is helpful when memory is the bottleneck.


.. _train-tune:

Hyperparameter Tuning with Ray Tune
===================================

Hyperparameter tuning with :ref:`Ray Tune <tune-main>` is natively supported with Ray Train.


.. https://docs.google.com/drawings/d/1yMd12iMkyo6DGrFoET1TIlKfFnXX9dfh2u3GSdTz6W4/edit

.. figure:: ../images/train-tuner.svg
    :align: center

    The `Tuner` will take in a `Trainer` and execute multiple training runs, each with different hyperparameter configurations.

Key Concepts
------------

There are a number of key concepts when doing hyperparameter optimization with a :class:`~ray.tune.Tuner`:

* A set of hyperparameters you want to tune in a *search space*.
* A *search algorithm* to effectively optimize your parameters and optionally use a
  *scheduler* to stop searches early and speed up your experiments.
* The *search space*, *search algorithm*, *scheduler*, and *Trainer* are passed to a Tuner,
  which runs the hyperparameter tuning workload by evaluating multiple hyperparameters in parallel.
* Each individual hyperparameter evaluation run is called a *trial*.
* The Tuner returns its results as a :class:`~ray.tune.ResultGrid`.

.. note::
   Tuners can also be used to launch hyperparameter tuning without using Ray Train. See
   :ref:`the Ray Tune documentation <tune-main>` for more guides and examples.

Basic usage
-----------

You can take an existing :class:`Trainer <ray.train.base_trainer.BaseTrainer>` and simply
pass it into a :class:`~ray.tune.Tuner`.

.. literalinclude:: ../doc_code/tuner.py
    :language: python
    :start-after: __basic_start__
    :end-before: __basic_end__



How to configure a Tuner?
-------------------------

There are two main configuration objects that can be passed into a Tuner: the :class:`TuneConfig <ray.tune.tune_config.TuneConfig>` and the :class:`RunConfig <ray.train.RunConfig>`.

The :class:`TuneConfig <ray.tune.TuneConfig>` contains tuning specific settings, including:

- the tuning algorithm to use
- the metric and mode to rank results
- the amount of parallelism to use

Here are some common configurations for `TuneConfig`:

.. literalinclude:: ../doc_code/tuner.py
    :language: python
    :start-after: __tune_config_start__
    :end-before: __tune_config_end__

See the :class:`TuneConfig API reference <ray.tune.tune_config.TuneConfig>` for more details.

The :class:`RunConfig <ray.train.RunConfig>` contains configurations that are more generic than tuning specific settings.
This includes:

- failure/retry configurations
- verbosity levels
- the name of the experiment
- the logging directory
- checkpoint configurations
- custom callbacks
- integration with cloud storage

Below we showcase some common configurations of :class:`RunConfig <ray.train.RunConfig>`.

.. literalinclude:: ../doc_code/tuner.py
    :language: python
    :start-after: __run_config_start__
    :end-before: __run_config_end__

See the :class:`RunConfig API reference <ray.train.RunConfig>` for more details.


Search Space configuration
--------------------------

A `Tuner` takes in a `param_space` argument where you can define the search space
from which hyperparameter configurations will be sampled.

Depending on the model and dataset, you may want to tune:

- The training batch size
- The learning rate for deep learning training (e.g., image classification)
- The maximum depth for tree-based models (e.g., XGBoost)

You can use a Tuner to tune most arguments and configurations for Ray Train, including but
not limited to:

- Ray :class:`Datasets <ray.data.Dataset>`
- :class:`~ray.train.ScalingConfig`
- and other hyperparameters.


Read more about :ref:`Tune search spaces here <tune-search-space-tutorial>`.

Train - Tune gotchas
--------------------

There are a couple gotchas about parameter specification when using Tuners with Trainers:

- By default, configuration dictionaries and config objects will be deep-merged.
- Parameters that are duplicated in the Trainer and Tuner will be overwritten by the Tuner ``param_space``.
- **Exception:** all arguments of the :class:`RunConfig <ray.train.RunConfig>` and :class:`TuneConfig <ray.tune.tune_config.TuneConfig>` are inherently un-tunable.

See :doc:`/tune/tutorials/tune_get_data_in_and_out` for an example.

Advanced Tuning
---------------

Tuners also offer the ability to tune over different data preprocessing steps and
different training/validation datasets, as shown in the following snippet.

.. literalinclude:: ../doc_code/tuner.py
    :language: python
    :start-after: __tune_dataset_start__
    :end-before: __tune_dataset_end__


:orphan:

Configuration Overview
======================

.. _train-run-config:

Run Configuration in Train (``RunConfig``)
------------------------------------------

``RunConfig`` is a configuration object used in Ray Train to define the experiment
spec that corresponds to a call to ``trainer.fit()``.

It includes settings such as the experiment name, storage path for results,
stopping conditions, custom callbacks, checkpoint configuration, verbosity level,
and logging options.

Many of these settings are configured through other config objects and passed through
the ``RunConfig``. The following sub-sections contain descriptions of these configs.

The properties of the run configuration are :ref:`not tunable <tune-search-space-tutorial>`.

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __run_config_start__
    :end-before: __run_config_end__

.. seealso::

    See the :class:`~ray.train.RunConfig` API reference.

    See :ref:`persistent-storage-guide` for storage configuration examples (related to ``storage_path``).



.. _:: ../doc_code:

.. _train-fault-tolerance:

Handling Failures and Node Preemption
=====================================

Automatically Recover from Train Worker Failures
------------------------------------------------

Ray Train has built-in fault tolerance to recover from worker failures (i.e.
``RayActorError``\s). When a failure is detected, the workers will be shut
down and new workers will be added in.

The training function will be restarted, but progress from the previous execution can
be resumed through checkpointing.

.. tip::
    In order to retain progress when recovery, your training function
    **must** implement logic for both :ref:`saving <train-dl-saving-checkpoints>`
    *and* :ref:`loading checkpoints <train-dl-loading-checkpoints>`.

Each instance of recovery from a worker failure is considered a retry. The
number of retries is configurable through the ``max_failures`` attribute of the
:class:`~ray.train.FailureConfig` argument set in the :class:`~ray.train.RunConfig`
passed to the ``Trainer``:

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __failure_config_start__
    :end-before: __failure_config_end__

Which checkpoint will be restored?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ray Train will automatically resume training from the latest available
:ref:`checkpoint reported to Ray Train <train-checkpointing>`.

This will be the last checkpoint passed to :func:`train.report() <ray.train.report>`.

.. _train-restore-guide:

Restore a Ray Train Experiment
------------------------------

At the experiment level, Trainer restoration 
allows you to resume a previously interrupted experiment from where it left off.

A Train experiment may be interrupted due to one of the following reasons:

- The experiment was manually interrupted (e.g., Ctrl+C, or pre-empted head node instance).
- The head node crashed (e.g., OOM or some other runtime error).
- The entire cluster went down (e.g., network error affecting all nodes).

Trainer restoration is possible for all of Ray Train's built-in trainers,
but we use ``TorchTrainer`` in the examples for demonstration.
We also use ``<Framework>Trainer`` to refer to methods that are shared across all
built-in trainers.

Let's say your initial Train experiment is configured as follows.
The actual training loop is just for demonstration purposes: the important detail is that
:ref:`saving <train-dl-saving-checkpoints>` *and* :ref:`loading checkpoints <train-dl-loading-checkpoints>`
has been implemented.

.. literalinclude:: ../doc_code/dl_guide.py
    :language: python
    :start-after: __ft_initial_run_start__
    :end-before: __ft_initial_run_end__

The results and checkpoints of the experiment are saved to the path configured by :class:`~ray.train.RunConfig`.
If the experiment has been interrupted due to one of the reasons listed above, use this path to resume:

.. literalinclude:: ../doc_code/dl_guide.py
    :language: python
    :start-after: __ft_restored_run_start__
    :end-before: __ft_restored_run_end__

.. tip::

    You can also restore from a remote path (e.g., from an experiment directory stored in a s3 bucket).

    .. literalinclude:: ../doc_code/dl_guide.py
        :language: python
        :dedent:
        :start-after: __ft_restore_from_cloud_initial_start__
        :end-before: __ft_restore_from_cloud_initial_end__

    .. literalinclude:: ../doc_code/dl_guide.py
        :language: python
        :dedent:
        :start-after: __ft_restore_from_cloud_restored_start__
        :end-before: __ft_restore_from_cloud_restored_end__

.. note::

    Different trainers may allow more parameters to be optionally re-specified on restore.
    Only **datasets** are required to be re-specified on restore, if they were supplied originally.

    `TorchTrainer.restore`, `TensorflowTrainer.restore`, and `HorovodTrainer.restore`
    can take in the same parameters as their parent class's
    :meth:`DataParallelTrainer.restore <ray.train.data_parallel_trainer.DataParallelTrainer.restore>`.

    Unless otherwise specified, other trainers will accept the same parameters as
    :meth:`BaseTrainer.restore <ray.train.trainer.BaseTrainer.restore>`.


Auto-resume
~~~~~~~~~~~

Adding the branching logic below will allow you to run the same script after the interrupt,
picking up training from where you left on the previous run. Notice that we use the
:meth:`<Framework>Trainer.can_restore <ray.train.trainer.BaseTrainer.can_restore>` utility method
to determine the existence and validity of the given experiment directory.

.. literalinclude:: ../doc_code/dl_guide.py
    :language: python
    :start-after: __ft_autoresume_start__
    :end-before: __ft_autoresume_end__

.. seealso::

    See the :meth:`BaseTrainer.restore <ray.train.trainer.BaseTrainer.restore>` docstring
    for a full example.

.. note::

    `<Framework>Trainer.restore` is different from
    :class:`<Framework>Trainer(..., resume_from_checkpoint=...) <ray.train.trainer.BaseTrainer>`.
    `resume_from_checkpoint` is meant to be used to start a *new* Train experiment,
    which writes results to a new directory and starts over from iteration 0.

    `<Framework>Trainer.restore` is used to continue an existing experiment, where
    new results will continue to be appended to existing logs.


.. _train-api:

Ray Train API
=============

.. _train-integration-api:
.. _train-framework-specific-ckpts:

.. currentmodule:: ray

PyTorch Ecosystem
-----------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~train.torch.TorchTrainer
    ~train.torch.TorchConfig
    ~train.torch.xla.TorchXLAConfig

.. _train-pytorch-integration:

PyTorch
~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~train.torch.get_device
    ~train.torch.get_devices
    ~train.torch.prepare_model
    ~train.torch.prepare_data_loader
    ~train.torch.enable_reproducibility

.. _train-lightning-integration:

PyTorch Lightning
~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~train.lightning.prepare_trainer
    ~train.lightning.RayLightningEnvironment
    ~train.lightning.RayDDPStrategy
    ~train.lightning.RayFSDPStrategy
    ~train.lightning.RayDeepSpeedStrategy
    ~train.lightning.RayTrainReportCallback

.. _train-transformers-integration:

Hugging Face Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~train.huggingface.transformers.prepare_trainer
    ~train.huggingface.transformers.RayTrainReportCallback


More Frameworks
---------------

Tensorflow/Keras
~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~train.tensorflow.TensorflowTrainer
    ~train.tensorflow.TensorflowConfig
    ~train.tensorflow.prepare_dataset_shard
    ~train.tensorflow.keras.ReportCheckpointCallback

Horovod
~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~train.horovod.HorovodTrainer
    ~train.horovod.HorovodConfig


XGBoost
~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~train.xgboost.XGBoostTrainer
    ~train.xgboost.RayTrainReportCallback


LightGBM
~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~train.lightgbm.LightGBMTrainer
    ~train.lightgbm.RayTrainReportCallback


.. _ray-train-configs-api:

Ray Train Configuration
-----------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~train.CheckpointConfig
    ~train.DataConfig
    ~train.FailureConfig
    ~train.RunConfig
    ~train.ScalingConfig
    ~train.SyncConfig

.. _train-loop-api:

Ray Train Utilities
-------------------

**Classes**

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~train.Checkpoint
    ~train.context.TrainContext

**Functions**

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~train.get_checkpoint
    ~train.get_context
    ~train.get_dataset_shard
    ~train.report


Ray Train Output
----------------

.. autosummary::
    :nosignatures:
    :template: autosummary/class_without_autosummary.rst
    :toctree: doc/

    ~train.Result


Ray Train Developer APIs
------------------------

.. _train-base-trainer:

Trainer Base Classes
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~train.trainer.BaseTrainer
    ~train.data_parallel_trainer.DataParallelTrainer
    ~train.base_trainer.TrainingFailedError


Train Backend Base Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _train-backend:
.. _train-backend-config:

.. autosummary::
    :nosignatures:
    :toctree: doc/
    :template: autosummary/class_without_autosummary.rst

    ~train.backend.Backend
    ~train.backend.BackendConfig


First, update your training code to support distributed training.
Begin by wrapping your code in a :ref:`training function <train-overview-training-function>`:

.. testcode::
    :skipif: True

    def train_func():
        # Your model training code here.
        ...

Each distributed training worker executes this function.

You can also specify the input argument for `train_func` as a dictionary via the Trainer's `train_loop_config`. For example:

.. testcode:: python
    :skipif: True

    def train_func(config):
        lr = config["lr"]
        num_epochs = config["num_epochs"]

    config = {"lr": 1e-4, "num_epochs": 10}
    trainer = ray.train.torch.TorchTrainer(train_func, train_loop_config=config, ...)

.. warning::

    Avoid passing large data objects through `train_loop_config` to reduce the
    serialization and deserialization overhead. Instead, it's preferred to
    initialize large objects (e.g. datasets, models) directly in `train_func`.

    .. code-block:: diff

         def load_dataset():
             # Return a large in-memory dataset
             ...
         
         def load_model():
             # Return a large in-memory model instance
             ...
 
        -config = {"data": load_dataset(), "model": load_model()}
 
         def train_func(config):
        -    data = config["data"]
        -    model = config["model"]
 
        +    data = load_dataset()
        +    model = load_model()
             ...
 
         trainer = ray.train.torch.TorchTrainer(train_func, train_loop_config=config, ...)


Configure scale and GPUs
------------------------

Outside of your training function, create a :class:`~ray.train.ScalingConfig` object to configure:

1. :class:`num_workers <ray.train.ScalingConfig>` - The number of distributed training worker processes.
2. :class:`use_gpu <ray.train.ScalingConfig>` - Whether each worker should use a GPU (or CPU).

.. testcode::

    from ray.train import ScalingConfig
    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)


For more details, see :ref:`train_scaling_config`.

Configure persistent storage
----------------------------

Create a :class:`~ray.train.RunConfig` object to specify the path where results
(including checkpoints and artifacts) will be saved.

.. testcode::

    from ray.train import RunConfig

    # Local path (/some/local/path/unique_run_name)
    run_config = RunConfig(storage_path="/some/local/path", name="unique_run_name")

    # Shared cloud storage URI (s3://bucket/unique_run_name)
    run_config = RunConfig(storage_path="s3://bucket", name="unique_run_name")

    # Shared NFS path (/mnt/nfs/unique_run_name)
    run_config = RunConfig(storage_path="/mnt/nfs", name="unique_run_name")


.. warning::

    Specifying a *shared storage location* (such as cloud storage or NFS) is
    *optional* for single-node clusters, but it is **required for multi-node clusters.**
    Using a local path will :ref:`raise an error <multinode-local-storage-warning>`
    during checkpointing for multi-node clusters.


For more details, see :ref:`persistent-storage-guide`.


Launch a training job
---------------------

Tying this all together, you can now launch a distributed training job
with a :class:`~ray.train.torch.TorchTrainer`.

.. testcode::
    :hide:

    from ray.train import ScalingConfig

    train_func = lambda: None
    scaling_config = ScalingConfig(num_workers=1)
    run_config = None

.. testcode::

    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(
        train_func, scaling_config=scaling_config, run_config=run_config
    )
    result = trainer.fit()


Access training results
-----------------------

After training completes, a :class:`~ray.train.Result` object is returned which contains
information about the training run, including the metrics and checkpoints reported during training.

.. testcode::

    result.metrics     # The metrics reported during training.
    result.checkpoint  # The latest checkpoint reported during training.
    result.path        # The path where logs are stored.
    result.error       # The exception that was raised, if training failed.

For more usage examples, see :ref:`train-inspect-results`.


:orphan:

Run Horovod Distributed Training with PyTorch and Ray Train
===========================================================

This basic example demonstrates how to run Horovod distributed training with PyTorch and Ray Train.

Code example
------------

.. literalinclude:: /../../python/ray/train/examples/horovod/horovod_example.py


See also
--------

* :ref:`Get Started with Horovod <train-horovod>` for a tutorial on using Horovod with Ray Train
* :doc:`Ray Train Examples <../../examples>` for more use cases


:orphan:

tensorflow_regression_example
=============================

.. literalinclude:: /../../python/ray/train/examples/tf/tensorflow_regression_example.py


:orphan:

Training with TensorFlow and Ray Train
======================================

This basic example runs distributed training of a TensorFlow model on MNIST with Ray Train.

Code example
------------

.. literalinclude:: /../../python/ray/train/examples/tf/tensorflow_mnist_example.py


See also
--------

* :doc:`Ray Train Examples <../../examples>` for more use cases.

* :ref:`Distributed Tensorflow & Keras <train-tensorflow-overview>` for a tutorial.


:orphan:

.. _tune_train_tf_example:

Tuning Hyperparameters of a Distributed TensorFlow Model using Ray Train & Tune
===============================================================================

.. literalinclude:: /../../python/ray/train/examples/tf/tune_tensorflow_mnist_example.py


:orphan:

Distributed Training with Hugging Face Accelerate
=================================================

This example does distributed data parallel training
with Hugging Face Accelerate, Ray Train, and Ray Data.
It fine-tunes a BERT model and is adapted from
https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py


Code example
------------

.. literalinclude:: /../../python/ray/train/examples/accelerate/accelerate_torch_trainer.py

See also
--------

* :ref:`Get Started with Hugging Face Accelerate <train-hf-accelerate>` for a tutorial on using Ray Train and HF Accelerate

* :doc:`Ray Train Examples <../../examples>` for more use cases


:orphan:

.. _tune_train_torch_example:

Tuning Hyperparameters of a Distributed PyTorch Model with PBT using Ray Train & Tune
=====================================================================================

.. literalinclude:: /../../python/ray/train/examples/pytorch/tune_cifar_torch_pbt_example.py



:orphan:

torch_regression_example
========================

.. literalinclude:: /../../python/ray/train/examples/pytorch/torch_regression_example.py


:orphan:

Train a PyTorch Model on Fashion MNIST
======================================

This example runs distributed training of a PyTorch model on Fashion MNIST with Ray Train.

Code example
------------

.. literalinclude:: /../../python/ray/train/examples/pytorch/torch_fashion_mnist_example.py

See also
--------

* :ref:`Get Started with PyTorch <train-pytorch>` for a tutorial on using Ray Train and PyTorch

* :doc:`Ray Train Examples <../../examples>` for more use cases


:orphan:

tune_torch_regression_example
=============================

.. literalinclude:: /../../python/ray/train/examples/pytorch/tune_torch_regression_example.py


:orphan:

Fine-tune of Stable Diffusion with DreamBooth and Ray Train
===========================================================

This is an intermediate example that shows how to do DreamBooth fine-tuning of a Stable Diffusion model using Ray Train.
It demonstrates how to use :ref:`Ray Data <data>` with PyTorch Lightning in Ray Train.


See the original `DreamBooth project homepage <https://dreambooth.github.io/>`_ for more details on what this fine-tuning method achieves.

.. image:: https://dreambooth.github.io/DreamBooth_files/high_level.png
  :target: https://dreambooth.github.io
  :alt: DreamBooth fine-tuning overview

This example builds on `this Hugging Face 🤗 tutorial <https://huggingface.co/docs/diffusers/training/dreambooth>`_.
See the Hugging Face tutorial for useful explanations and suggestions on hyperparameters.
**Adapting this example to Ray Train allows you to easily scale up the fine-tuning to an arbitrary number of distributed training workers.**

**Compute requirements:**

* Because of the large model sizes, you need a machine with at least 1 A10G GPU.
* Each training worker uses 1 GPU. You can use multiple GPUs or workers to leverage data-parallel training to speed up training time.

This example fine-tunes both the ``text_encoder`` and ``unet`` models used in the stable diffusion process, with respect to a prior preserving loss.


.. image:: /templates/05_dreambooth_finetuning/dreambooth/images/dreambooth_example.png
   :alt: DreamBooth overview

Find the full code repository at `https://github.com/ray-project/ray/tree/master/doc/source/templates/05_dreambooth_finetuning <https://github.com/ray-project/ray/tree/master/doc/source/templates/05_dreambooth_finetuning>`_


How it works
------------

This example uses Ray Data for data loading and Ray Train for distributed training.

Data loading
^^^^^^^^^^^^

.. note::
    Find the latest version of the code at `dataset.py <https://github.com/ray-project/ray/tree/master/doc/source/templates/05_dreambooth_finetuning/dreambooth/dataset.py>`_

    The latest version might differ slightly from the code presented here.


Use Ray Data for data loading. The code has three interesting parts.

First, load two datasets using :func:`ray.data.read_images`:

.. literalinclude:: /templates/05_dreambooth_finetuning/dreambooth/dataset.py
  :language: python
  :start-at: instance_dataset = read
  :end-at: class_dataset = read
  :dedent: 4

Then, tokenize the prompt that generated these images:

.. literalinclude:: /templates/05_dreambooth_finetuning/dreambooth/dataset.py
  :language: python
  :start-at: tokenizer = AutoTokenizer
  :end-at: instance_prompt_ids = _tokenize
  :dedent: 4


And lastly, apply a ``torchvision`` preprocessing pipeline to the images:

.. literalinclude:: /templates/05_dreambooth_finetuning/dreambooth/dataset.py
  :language: python
  :start-after: START: image preprocessing
  :end-before: END: image preprocessing
  :dedent: 4

Apply all three parts in a final step:

.. literalinclude:: /templates/05_dreambooth_finetuning/dreambooth/dataset.py
  :language: python
  :start-after: START: Apply preprocessing
  :end-before: END: Apply preprocessing
  :dedent: 4


Distributed training
^^^^^^^^^^^^^^^^^^^^


.. note::
    Find the latest version of the code at `train.py <https://github.com/ray-project/ray/tree/master/doc/source/templates/05_dreambooth_finetuning/dreambooth/train.py>`_

    The latest version might differ slightly from the code presented here.


The central part of the training code is the :ref:`training function <train-overview-training-function>`. This function accepts a configuration dict that contains the hyperparameters. It then defines a regular PyTorch training loop.

You interact with the Ray Train API in only a few locations, which follow in-line comments in the snippet below.

Remember that you want to do data-parallel training for all the models.


#. Load the data shard for each worker with `session.get_dataset_shard("train")``
#. Iterate over the dataset with `train_dataset.iter_torch_batches()``
#. Report results to Ray Train with `session.report(results)``

The code is compacted for brevity. The `full code <https://github.com/ray-project/ray/tree/master/doc/source/templates/05_dreambooth_finetuning/dreambooth/train.py>`_ is more thoroughly annotated.


.. literalinclude:: /templates/05_dreambooth_finetuning/dreambooth/train.py
  :language: python
  :start-at: def train_fn(config)
  :end-before: END: Training loop

You can then run this training function with Ray Train's TorchTrainer:


.. literalinclude:: /templates/05_dreambooth_finetuning/dreambooth/train.py
  :language: python
  :start-at: args = train_arguments
  :end-at: trainer.fit()
  :dedent: 4

Configure the scale
^^^^^^^^^^^^^^^^^^^

In the TorchTrainer, you can easily configure the scale.
The preceding example uses the ``num_workers`` argument to specify the number
of workers. This argument defaults to 2 workers with 1 GPU each, totalling to 2 GPUs.

To run the example on 4 GPUs, set the number of workers to 4 using ``--num-workers=4``.
Or you can change the scaling config directly:

.. code-block:: diff

     scaling_config=ScalingConfig(
         use_gpu=True,
    -    num_workers=args.num_workers,
    +    num_workers=4,
     )

If you're running multi-node training, make sure that all nodes have access to a shared
storage like NFS or EFS. In the following example script, you can adjust the location with the
``DATA_PREFIX`` environment variable.

Training throughput
~~~~~~~~~~~~~~~~~~~

Compare throughput of the preceding training runs that used 1,  2, and 4 workers or GPUs.

Consider the following setup:

* 1 GCE g2-standard-48-nvidia-l4-4 instance with 4 GPUs
* Model as configured below
* Data from this example
* 200 regularization images
* Training for 4 epochs (local batch size = 2)
* 3 runs per configuration

You expect that the training time should benefit from scale and decreases when running with
more workers and GPUs.

.. image:: /templates/05_dreambooth_finetuning/dreambooth/images/dreambooth_training.png
   :alt: DreamBooth training times

.. list-table::
   :header-rows: 1

   * - Number of workers/GPUs
     - Training time (seconds)
   * - 1
     - 802.14
   * - 2
     - 487.82
   * - 4
     - 313.25


While the training time decreases linearly with the amount of workers/GPUs, you can observe some penalty.
Specifically, with double the amount of workers you don't get half of the training time.

This penalty is most likely due to additional communication between processes and the transfer of large model
weights. You are also only training with a batch size of one because of the GPU memory limitation. On larger
GPUs with higher batch sizes you would expect a greater benefit from scaling out.


Run the example
---------------

First, download the pre-trained Stable Diffusion model as a starting point.

Then train this model with a few images of a subject.

To achieve this, choose a non-word as an identifier, such as ``unqtkn``. When fine-tuning the model with this subject, you teach the model that the prompt is ``A photo of a unqtkn <class>``.

After fine-tuning you can run inference with this specific prompt.
For instance: ``A photo of a unqtkn <class>`` creates an image of the subject.
Similarly, ``A photo of a unqtkn <class> at the beach`` creates an image of the subject at the beach.

Step 0: Preparation
^^^^^^^^^^^^^^^^^^^

Clone the Ray repository, go to the example directory, and install dependencies.

.. code-block:: bash

   git clone https://github.com/ray-project/ray.git
   cd doc/source/templates/05_dreambooth_finetuning
   pip install -Ur dreambooth/requirements.txt

Prepare some directories and environment variables.

.. literalinclude:: /templates/05_dreambooth_finetuning/dreambooth_run.sh
  :language: bash
  :start-after: __preparation_start__
  :end-before: __preparation_end__


Step 1: Download the pre-trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download and cache a pre-trained Stable Diffusion model locally.

.. literalinclude:: /templates/05_dreambooth_finetuning/dreambooth_run.sh
  :language: bash
  :start-after: __cache_model_start__
  :end-before: __cache_model_end__

You can access the downloaded model checkpoint at the ``$ORIG_MODEL_PATH``.

Step 2: Supply images of your subject
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use one of the sample datasets, like `dog` or `lego car`, or provide your own directory
of images, and specify the directory with the ``$INSTANCE_DIR`` environment variable.

Then, copy these images to ``$IMAGES_OWN_DIR``.

.. literalinclude:: /templates/05_dreambooth_finetuning/dreambooth_run.sh
  :language: bash
  :start-after: __supply_own_images_start__
  :end-before: __supply_own_images_end__

The ``$CLASS_NAME`` should be the general category of your subject.
The images produced by the prompt ``photo of a unqtkn <class>`` should be diverse images
that are different enough from the subject in order for generated images to clearly
show the effect of fine-tuning.

Step 3: Create the regularization images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a regularization image set for a class of subjects using the pre-trained
Stable Diffusion model. This regularization set ensures that
the model still produces decent images for random images of the same class,
rather than just optimize for producing good images of the subject.

.. literalinclude:: /templates/05_dreambooth_finetuning/dreambooth_run.sh
  :language: bash
  :start-after: Step 3: START
  :end-before: Step 3: END

Use Ray Data to do batch inference with 4 workers, to generate more images in parallel.

Step 4: Fine-tune the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save a few, like 4 to 5, images of the subject being fine-tuned
in a local directory. Then launch the training job with:

.. literalinclude:: /templates/05_dreambooth_finetuning/dreambooth_run.sh
  :language: bash
  :start-after: Step 4: START
  :end-before: Step 4: END

Step 5: Generate images of the subject
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Try your model with the same command line as Step 2, but point
to your own model this time.

.. literalinclude:: /templates/05_dreambooth_finetuning/dreambooth_run.sh
  :language: bash
  :start-after: Step 5: START
  :end-before: Step 5: END

Next, try replacing the prompt with something more interesting.

For example, for the dog subject, you can try:

- "photo of a unqtkn dog in a bucket"
- "photo of a unqtkn dog sleeping"
- "photo of a unqtkn dog in a doghouse"

See also
--------

* :doc:`Ray Train Examples <../../examples>` for more use cases

* :ref:`Ray Train User Guides <train-user-guides>` for how-to guides


:orphan:

Train with DeepSpeed ZeRO-3 and Ray Train
=========================================

This is an intermediate example that shows how to do distributed training with DeepSpeed ZeRO-3 and Ray Train.
It demonstrates how to use :ref:`Ray Data <data>` with DeepSpeed ZeRO-3 and Ray Train.
If you just want to quickly convert your existing TorchTrainer scripts into Ray Train, you can refer to the :ref:`Train with DeepSpeed <train-deepspeed>`.


Code example
------------

.. literalinclude:: /../../python/ray/train/examples/deepspeed/deepspeed_torch_trainer.py


See also
--------

* :doc:`Ray Train Examples <../../examples>` for more use cases.

* :ref:`Get Started with DeepSpeed <train-deepspeed>` for a tutorial.


:orphan:

.. _transformers_torch_trainer_basic_example :

Fine-tune a Text Classifier with Hugging Face Transformers
==========================================================

This basic example of distributed training with Ray Train and Hugging Face (HF) Transformers
fine-tunes a text classifier on the Yelp review dataset using HF Transformers and Ray Train.

Code example
------------

.. literalinclude:: /../../python/ray/train/examples/transformers/transformers_torch_trainer_basic.py

See also
--------

* :ref:`Get Started with Hugging Face Transformers <train-pytorch-transformers>` for a tutorial

* :doc:`Ray Train Examples <../../examples>` for more use cases


