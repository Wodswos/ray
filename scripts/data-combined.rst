.. _transforming_data:

=================
Transforming Data
=================

Transformations let you process and modify your dataset. You can compose transformations
to express a chain of computations.

.. note::
    Transformations are lazy by default. They aren't executed until you trigger consumption of the data by :ref:`iterating over the Dataset <iterating-over-data>`, :ref:`saving the Dataset <saving-data>`, or :ref:`inspecting properties of the Dataset <inspecting-data>`.

This guide shows you how to:

* :ref:`Transform rows <transforming_rows>`
* :ref:`Transform batches <transforming_batches>`
* :ref:`Stateful transforms <stateful_transforms>`
* :ref:`Groupby and transform groups <transforming_groupby>`

.. _transforming_rows:

Transforming rows
=================

.. tip::

    If your transformation is vectorized, call :meth:`~ray.data.Dataset.map_batches` for
    better performance. To learn more, see :ref:`Transforming batches <transforming_batches>`.

Transforming rows with map
~~~~~~~~~~~~~~~~~~~~~~~~~~

If your transformation returns exactly one row for each input row, call
:meth:`~ray.data.Dataset.map`.

.. testcode::

    import os
    from typing import Any, Dict
    import ray

    def parse_filename(row: Dict[str, Any]) -> Dict[str, Any]:
        row["filename"] = os.path.basename(row["path"])
        return row

    ds = (
        ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple", include_paths=True)
        .map(parse_filename)
    )

The user defined function passed to :meth:`~ray.data.Dataset.map` should be of type
`Callable[[Dict[str, Any]], Dict[str, Any]]`. In other words, your function should
input and output a dictionary with keys of strings and values of any type. For example:

.. testcode::

    from typing import Any, Dict

    def fn(row: Dict[str, Any]) -> Dict[str, Any]:
        # access row data
        value = row["col1"]

        # add data to row
        row["col2"] = ...

        # return row
        return row

Transforming rows with flat map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your transformation returns multiple rows for each input row, call
:meth:`~ray.data.Dataset.flat_map`.

.. testcode::

    from typing import Any, Dict, List
    import ray

    def duplicate_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [row] * 2

    print(
        ray.data.range(3)
        .flat_map(duplicate_row)
        .take_all()
    )

.. testoutput::

    [{'id': 0}, {'id': 0}, {'id': 1}, {'id': 1}, {'id': 2}, {'id': 2}]

The user defined function passed to :meth:`~ray.data.Dataset.flat_map` should be of type
`Callable[[Dict[str, Any]], List[Dict[str, Any]]]`. In other words your function should
input a dictionary with keys of strings and values of any type and output a list of
dictionaries that have the same type as the input, for example:

.. testcode::

    from typing import Any, Dict, List

    def fn(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        # access row data
        value = row["col1"]

        # add data to row
        row["col2"] = ...

        # construct output list
        output = [row, row]

        # return list of output rows
        return output

.. _transforming_batches:

Transforming batches
====================

If your transformation is vectorized like most NumPy or pandas operations, transforming
batches is more performant than transforming rows.

.. testcode::

    from typing import Dict
    import numpy as np
    import ray

    def increase_brightness(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        batch["image"] = np.clip(batch["image"] + 4, 0, 255)
        return batch

    ds = (
        ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
        .map_batches(increase_brightness)
    )

.. _configure_batch_format:

Configuring batch format
~~~~~~~~~~~~~~~~~~~~~~~~

Ray Data represents batches as dicts of NumPy ndarrays or pandas DataFrames. By
default, Ray Data represents batches as dicts of NumPy ndarrays. To configure the batch type,
specify ``batch_format`` in :meth:`~ray.data.Dataset.map_batches`. You can return either
format from your function, but ``batch_format`` should match the input of your function.

.. tab-set::

    .. tab-item:: NumPy

        .. testcode::

            from typing import Dict
            import numpy as np
            import ray

            def increase_brightness(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                batch["image"] = np.clip(batch["image"] + 4, 0, 255)
                return batch

            ds = (
                ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
                .map_batches(increase_brightness, batch_format="numpy")
            )

    .. tab-item:: pandas

        .. testcode::

            import pandas as pd
            import ray

            def drop_nas(batch: pd.DataFrame) -> pd.DataFrame:
                return batch.dropna()

            ds = (
                ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")
                .map_batches(drop_nas, batch_format="pandas")
            )

The user defined function you pass to :meth:`~ray.data.Dataset.map_batches` is more flexible. Because you can represent batches
in multiple ways (see :ref:`Configuring batch format <configure_batch_format>`), the function should be of type
``Callable[DataBatch, DataBatch]``, where ``DataBatch = Union[pd.DataFrame, Dict[str, np.ndarray]]``. In
other words, your function should take as input and output a batch of data which you can represent as a
pandas DataFrame or a dictionary with string keys and NumPy ndarrays values. For example, your function might look like:

.. testcode::

    import pandas as pd

    def fn(batch: pd.DataFrame) -> pd.DataFrame:
        # modify batch
        batch = ...

        # return batch
        return output

The user defined function can also be a Python generator that yields batches, so the function can also
be of type ``Callable[DataBatch, Iterator[[DataBatch]]``, where ``DataBatch = Union[pd.DataFrame, Dict[str, np.ndarray]]``.
In this case, your function would look like:

.. testcode::

    from typing import Dict, Iterator
    import numpy as np

    def fn(batch: Dict[str, np.ndarray]) -> Iterator[Dict[str, np.ndarray]]:
        # yield the same batch multiple times
        for _ in range(10):
            yield batch

Configuring batch size
~~~~~~~~~~~~~~~~~~~~~~

Increasing ``batch_size`` improves the performance of vectorized transformations like
NumPy functions and model inference. However, if your batch size is too large, your
program might run out of memory. If you encounter an out-of-memory error, decrease your
``batch_size``.

.. _stateful_transforms:

Stateful Transforms
===================

If your transform requires expensive setup such as downloading
model weights, use a callable Python class instead of a function to make the transform stateful. When a Python class
is used, the ``__init__`` method is called to perform setup exactly once on each worker.
In contrast, functions are stateless, so any setup must be performed for each data item.

Internally, Ray Data uses tasks to execute functions, and uses actors to execute classes.
To learn more about tasks and actors, read the
:ref:`Ray Core Key Concepts <core-key-concepts>`.

To transform data with a Python class, complete these steps:

1. Implement a class. Perform setup in ``__init__`` and transform data in ``__call__``.

2. Call :meth:`~ray.data.Dataset.map_batches`, :meth:`~ray.data.Dataset.map`, or
   :meth:`~ray.data.Dataset.flat_map`. Pass the number of concurrent workers to use with the ``concurrency`` argument. Each worker transforms a partition of data in parallel.
   Fixing the number of concurrent workers gives the most predictable performance, but you can also pass a tuple of ``(min, max)`` to allow Ray Data to automatically
   scale the number of concurrent workers.

.. tab-set::

    .. tab-item:: CPU

        .. testcode::

            from typing import Dict
            import numpy as np
            import torch
            import ray

            class TorchPredictor:

                def __init__(self):
                    self.model = torch.nn.Identity()
                    self.model.eval()

                def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                    inputs = torch.as_tensor(batch["data"], dtype=torch.float32)
                    with torch.inference_mode():
                        batch["output"] = self.model(inputs).detach().numpy()
                    return batch

            ds = (
                ray.data.from_numpy(np.ones((32, 100)))
                .map_batches(TorchPredictor, concurrency=2)
            )

        .. testcode::
            :hide:

            ds.materialize()

    .. tab-item:: GPU

        .. testcode::

            from typing import Dict
            import numpy as np
            import torch
            import ray

            class TorchPredictor:

                def __init__(self):
                    self.model = torch.nn.Identity().cuda()
                    self.model.eval()

                def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                    inputs = torch.as_tensor(batch["data"], dtype=torch.float32).cuda()
                    with torch.inference_mode():
                        batch["output"] = self.model(inputs).detach().cpu().numpy()
                    return batch

            ds = (
                ray.data.from_numpy(np.ones((32, 100)))
                .map_batches(
                    TorchPredictor,
                    # Two workers with one GPU each
                    concurrency=2,
                    # Batch size is required if you're using GPUs.
                    batch_size=4,
                    num_gpus=1
                )
            )

        .. testcode::
            :hide:

            ds.materialize()

.. _transforming_groupby:

Groupby and transforming groups
===============================

To transform groups, call :meth:`~ray.data.Dataset.groupby` to group rows. Then, call
:meth:`~ray.data.grouped_data.GroupedData.map_groups` to transform the groups.

.. tab-set::

    .. tab-item:: NumPy

        .. testcode::

            from typing import Dict
            import numpy as np
            import ray

            items = [
                {"image": np.zeros((32, 32, 3)), "label": label}
                for _ in range(10) for label in range(100)
            ]

            def normalize_images(group: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                group["image"] = (group["image"] - group["image"].mean()) / group["image"].std()
                return group

            ds = (
                ray.data.from_items(items)
                .groupby("label")
                .map_groups(normalize_images)
            )

    .. tab-item:: pandas

        .. testcode::

            import pandas as pd
            import ray

            def normalize_features(group: pd.DataFrame) -> pd.DataFrame:
                target = group.drop("target")
                group = (group - group.min()) / group.std()
                group["target"] = target
                return group

            ds = (
                ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")
                .groupby("target")
                .map_groups(normalize_features)
            )


.. _inspecting-data:

===============
Inspecting Data
===============

Inspect :class:`Datasets <ray.data.Dataset>` to better understand your data.

This guide shows you how to:

* `Describe datasets <#describing-datasets>`_
* `Inspect rows <#inspecting-rows>`_
* `Inspect batches <#inspecting-batches>`_
* `Inspect execution statistics <#inspecting-execution-statistics>`_

.. _describing-datasets:

Describing datasets
===================

:class:`Datasets <ray.data.Dataset>` are tabular. To view a dataset's column names and
types, call :meth:`Dataset.schema() <ray.data.Dataset.schema>`.

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

    print(ds.schema())

.. testoutput::

    Column             Type
    ------             ----
    sepal length (cm)  double
    sepal width (cm)   double
    petal length (cm)  double
    petal width (cm)   double
    target             int64

For more information like the number of rows, print the Dataset.

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

    print(ds)

.. testoutput::

    Dataset(
       num_rows=150,
       schema={
          sepal length (cm): double,
          sepal width (cm): double,
          petal length (cm): double,
          petal width (cm): double,
          target: int64
       }
    )

.. _inspecting-rows:

Inspecting rows
===============

To get a list of rows, call :meth:`Dataset.take() <ray.data.Dataset.take>` or
:meth:`Dataset.take_all() <ray.data.Dataset.take_all>`. Ray Data represents each row as
a dictionary.

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

    rows = ds.take(1)
    print(rows)

.. testoutput::

    [{'sepal length (cm)': 5.1, 'sepal width (cm)': 3.5, 'petal length (cm)': 1.4, 'petal width (cm)': 0.2, 'target': 0}]


For more information on working with rows, see
:ref:`Transforming rows <transforming_rows>` and
:ref:`Iterating over rows <iterating-over-rows>`.

.. _inspecting-batches:

Inspecting batches
==================

A batch contains data from multiple rows. To inspect batches, call
`Dataset.take_batch() <ray.data.Dataset.take_batch>`.

By default, Ray Data represents batches as dicts of NumPy ndarrays. To change the type
of the returned batch, set ``batch_format``.

.. tab-set::

    .. tab-item:: NumPy

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            batch = ds.take_batch(batch_size=2, batch_format="numpy")
            print("Batch:", batch)
            print("Image shape", batch["image"].shape)

        .. testoutput::
            :options: +MOCK

            Batch: {'image': array([[[[...]]]], dtype=uint8)}
            Image shape: (2, 32, 32, 3)

    .. tab-item:: pandas

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

            batch = ds.take_batch(batch_size=2, batch_format="pandas")
            print(batch)

        .. testoutput::
            :options: +NORMALIZE_WHITESPACE

               sepal length (cm)  sepal width (cm)  ...  petal width (cm)  target
            0                5.1               3.5  ...               0.2       0
            1                4.9               3.0  ...               0.2       0
            <BLANKLINE>
            [2 rows x 5 columns]

For more information on working with batches, see
:ref:`Transforming batches <transforming_batches>` and
:ref:`Iterating over batches <iterating-over-batches>`.


Inspecting execution statistics
===============================

Ray Data calculates statistics during execution for each operator, such as wall clock time and memory usage.

To view stats about your :class:`Datasets <ray.data.Dataset>`, call :meth:`Dataset.stats() <ray.data.Dataset.stats>` on an executed dataset. The stats are also persisted under `/tmp/ray/session_*/logs/ray-data/ray-data.log`.
For more on how to read this output, see :ref:`Monitoring Your Workload with the Ray Data Dashboard <monitoring-your-workload>`.

.. testcode::

    import ray
    import datasets

    def f(batch):
        return batch

    def g(row):
        return True

    hf_ds = datasets.load_dataset("mnist", "mnist")
    ds = (
        ray.data.from_huggingface(hf_ds["train"])
        .map_batches(f)
        .filter(g)
        .materialize()
    )

    print(ds.stats())

.. testoutput::
    :options: +MOCK

    Operator 1 ReadParquet->SplitBlocks(32): 1 tasks executed, 32 blocks produced in 2.92s
    * Remote wall time: 103.38us min, 1.34s max, 42.14ms mean, 1.35s total
    * Remote cpu time: 102.0us min, 164.66ms max, 5.37ms mean, 171.72ms total
    * UDF time: 0us min, 0us max, 0.0us mean, 0us total
    * Peak heap memory usage (MiB): 266375.0 min, 281875.0 max, 274491 mean
    * Output num rows per block: 1875 min, 1875 max, 1875 mean, 60000 total
    * Output size bytes per block: 537986 min, 555360 max, 545963 mean, 17470820 total
    * Output rows per task: 60000 min, 60000 max, 60000 mean, 1 tasks used
    * Tasks per node: 1 min, 1 max, 1 mean; 1 nodes used
    * Operator throughput:
        * Ray Data throughput: 20579.80984833993 rows/s
        * Estimated single node throughput: 44492.67361278733 rows/s

    Operator 2 MapBatches(f)->Filter(g): 32 tasks executed, 32 blocks produced in 3.63s
    * Remote wall time: 675.48ms min, 1.0s max, 797.07ms mean, 25.51s total
    * Remote cpu time: 673.41ms min, 897.32ms max, 768.09ms mean, 24.58s total
    * UDF time: 661.65ms min, 978.04ms max, 778.13ms mean, 24.9s total
    * Peak heap memory usage (MiB): 152281.25 min, 286796.88 max, 164231 mean
    * Output num rows per block: 1875 min, 1875 max, 1875 mean, 60000 total
    * Output size bytes per block: 530251 min, 547625 max, 538228 mean, 17223300 total
    * Output rows per task: 1875 min, 1875 max, 1875 mean, 32 tasks used
    * Tasks per node: 32 min, 32 max, 32 mean; 1 nodes used
    * Operator throughput:
        * Ray Data throughput: 16512.364546087643 rows/s
        * Estimated single node throughput: 2352.3683708977856 rows/s

    Dataset throughput:
        * Ray Data throughput: 11463.372316361854 rows/s
        * Estimated single node throughput: 25580.963670075285 rows/s


.. _working_with_pytorch:

Working with PyTorch
====================

Ray Data integrates with the PyTorch ecosystem.

This guide describes how to:

* :ref:`Iterate over your dataset as Torch tensors for model training <iterating_pytorch>`
* :ref:`Write transformations that deal with Torch tensors <transform_pytorch>`
* :ref:`Perform batch inference with Torch models <batch_inference_pytorch>`
* :ref:`Save Datasets containing Torch tensors <saving_pytorch>`
* :ref:`Migrate from PyTorch Datasets to Ray Data <migrate_pytorch>`

.. _iterating_pytorch:

Iterating over Torch tensors for training
-----------------------------------------
To iterate over batches of data in Torch format, call :meth:`Dataset.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`. Each batch is represented as `Dict[str, torch.Tensor]`, with one tensor per column in the dataset.

This is useful for training Torch models with batches from your dataset. For configuration details such as providing a ``collate_fn`` for customizing the conversion, see the API reference for :meth:`iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`.

.. testcode::

    import ray
    import torch

    ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

    for batch in ds.iter_torch_batches(batch_size=2):
        print(batch)

.. testoutput::
    :options: +MOCK

    {'image': tensor([[[[...]]]], dtype=torch.uint8)}
    ...
    {'image': tensor([[[[...]]]], dtype=torch.uint8)}

Integration with Ray Train
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ray Data integrates with :ref:`Ray Train <train-docs>` for easy data ingest for data parallel training, with support for PyTorch, PyTorch Lightning, or Hugging Face training.

.. testcode::

    import torch
    from torch import nn
    import ray
    from ray import train
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    def train_func():
        model = nn.Sequential(nn.Linear(30, 1), nn.Sigmoid())
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Datasets can be accessed in your train_func via ``get_dataset_shard``.
        train_data_shard = train.get_dataset_shard("train")

        for epoch_idx in range(2):
            for batch in train_data_shard.iter_torch_batches(batch_size=128, dtypes=torch.float32):
                features = torch.stack([batch[col_name] for col_name in batch.keys() if col_name != "target"], axis=1)
                predictions = model(features)
                train_loss = loss_fn(predictions, batch["target"].unsqueeze(1))
                train_loss.backward()
                optimizer.step()


    train_dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")

    trainer = TorchTrainer(
        train_func,
        datasets={"train": train_dataset},
        scaling_config=ScalingConfig(num_workers=2)
    )
    trainer.fit()


For more details, see the :ref:`Ray Train user guide <data-ingest-torch>`.

.. _transform_pytorch:

Transformations with Torch tensors
----------------------------------
Transformations applied with `map` or `map_batches` can return Torch tensors.

.. caution::

    Under the hood, Ray Data automatically converts Torch tensors to NumPy arrays. Subsequent transformations accept NumPy arrays as input, not Torch tensors.

.. tab-set::

    .. tab-item:: map

        .. testcode::

            from typing import Dict
            import numpy as np
            import torch
            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            def convert_to_torch(row: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
                return {"tensor": torch.as_tensor(row["image"])}

            # The tensor gets converted into a Numpy array under the hood
            transformed_ds = ds.map(convert_to_torch)
            print(transformed_ds.schema())

            # Subsequent transformations take in Numpy array as input.
            def check_numpy(row: Dict[str, np.ndarray]):
                assert isinstance(row["tensor"], np.ndarray)
                return row

            transformed_ds.map(check_numpy).take_all()

        .. testoutput::

            Column  Type
            ------  ----
            tensor  numpy.ndarray(shape=(32, 32, 3), dtype=uint8)

    .. tab-item:: map_batches

        .. testcode::

            from typing import Dict
            import numpy as np
            import torch
            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            def convert_to_torch(batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
                return {"tensor": torch.as_tensor(batch["image"])}

            # The tensor gets converted into a Numpy array under the hood
            transformed_ds = ds.map_batches(convert_to_torch, batch_size=2)
            print(transformed_ds.schema())

            # Subsequent transformations take in Numpy array as input.
            def check_numpy(batch: Dict[str, np.ndarray]):
                assert isinstance(batch["tensor"], np.ndarray)
                return batch

            transformed_ds.map_batches(check_numpy, batch_size=2).take_all()

        .. testoutput::

            Column  Type
            ------  ----
            tensor  numpy.ndarray(shape=(32, 32, 3), dtype=uint8)

For more information on transforming data, see :ref:`Transforming data <transforming_data>`.

Built-in PyTorch transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use built-in Torch transforms from ``torchvision``, ``torchtext``, and ``torchaudio``.

.. tab-set::

    .. tab-item:: torchvision

        .. testcode::

            from typing import Dict
            import numpy as np
            import torch
            from torchvision import transforms
            import ray

            # Create the Dataset.
            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            # Define the torchvision transform.
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.CenterCrop(10)
                ]
            )

            # Define the map function
            def transform_image(row: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
                row["transformed_image"] = transform(row["image"])
                return row

            # Apply the transform over the dataset.
            transformed_ds = ds.map(transform_image)
            print(transformed_ds.schema())

        .. testoutput::

            Column             Type
            ------             ----
            image              numpy.ndarray(shape=(32, 32, 3), dtype=uint8)
            transformed_image  numpy.ndarray(shape=(3, 10, 10), dtype=float)

    .. tab-item:: torchtext

        .. testcode::

            from typing import Dict, List
            import numpy as np
            from torchtext import transforms
            import ray

            # Create the Dataset.
            ds = ray.data.read_text("s3://anonymous@ray-example-data/simple.txt")

            # Define the torchtext transform.
            VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
            transform = transforms.BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)

            # Define the map_batches function.
            def tokenize_text(batch: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
                batch["tokenized_text"] = transform(list(batch["text"]))
                return batch

            # Apply the transform over the dataset.
            transformed_ds = ds.map_batches(tokenize_text, batch_size=2)
            print(transformed_ds.schema())

        .. testoutput::

            Column          Type
            ------          ----
            text            <class 'object'>
            tokenized_text  <class 'object'>

.. _batch_inference_pytorch:

Batch inference with PyTorch
----------------------------

With Ray Datasets, you can do scalable offline batch inference with Torch models by mapping a pre-trained model over your data.

.. testcode::

    from typing import Dict
    import numpy as np
    import torch
    import torch.nn as nn

    import ray

    # Step 1: Create a Ray Dataset from in-memory Numpy arrays.
    # You can also create a Ray Dataset from many other sources and file
    # formats.
    ds = ray.data.from_numpy(np.ones((1, 100)))

    # Step 2: Define a Predictor class for inference.
    # Use a class to initialize the model just once in `__init__`
    # and re-use it for inference across multiple batches.
    class TorchPredictor:
        def __init__(self):
            # Load a dummy neural network.
            # Set `self.model` to your pre-trained PyTorch model.
            self.model = nn.Sequential(
                nn.Linear(in_features=100, out_features=1),
                nn.Sigmoid(),
            )
            self.model.eval()

        # Logic for inference on 1 batch of data.
        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            tensor = torch.as_tensor(batch["data"], dtype=torch.float32)
            with torch.inference_mode():
                # Get the predictions from the input batch.
                return {"output": self.model(tensor).numpy()}

    # Step 3: Map the Predictor over the Dataset to get predictions.
    # Use 2 parallel actors for inference. Each actor predicts on a
    # different partition of data.
    predictions = ds.map_batches(TorchPredictor, concurrency=2)
    # Step 4: Show one prediction output.
    predictions.show(limit=1)

.. testoutput::
    :options: +MOCK

    {'output': array([0.5590901], dtype=float32)}

For more details, see the :ref:`Batch inference user guide <batch_inference_home>`.

.. _saving_pytorch:

Saving Datasets containing Torch tensors
----------------------------------------

Datasets containing Torch tensors can be saved to files, like parquet or NumPy.

For more information on saving data, read
:ref:`Saving data <saving-data>`.

.. caution::

    Torch tensors that are on GPU devices can't be serialized and written to disk. Convert the tensors to CPU (``tensor.to("cpu")``) before saving the data.

.. tab-set::

    .. tab-item:: Parquet

        .. testcode::

            import torch
            import ray

            tensor = torch.Tensor(1)
            ds = ray.data.from_items([{"tensor": tensor}])

            ds.write_parquet("local:///tmp/tensor")

    .. tab-item:: Numpy

        .. testcode::

            import torch
            import ray

            tensor = torch.Tensor(1)
            ds = ray.data.from_items([{"tensor": tensor}])

            ds.write_numpy("local:///tmp/tensor", column="tensor")

.. _migrate_pytorch:

Migrating from PyTorch Datasets and DataLoaders
-----------------------------------------------

If you're currently using PyTorch Datasets and DataLoaders, you can migrate to Ray Data for working with distributed datasets.

PyTorch Datasets are replaced by the :class:`Dataset <ray.data.Dataset>` abstraction, and the PyTorch DataLoader is replaced by :meth:`Dataset.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`.

Built-in PyTorch Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using built-in PyTorch datasets, for example from ``torchvision``, these can be converted to a Ray Dataset using the :meth:`from_torch() <ray.data.from_torch>` API.

.. testcode::

    import torchvision
    import ray

    mnist = torchvision.datasets.MNIST(root="/tmp/", download=True)
    ds = ray.data.from_torch(mnist)

    # The data for each item of the Torch dataset is under the "item" key.
    print(ds.schema())

..
    The following `testoutput` is mocked to avoid illustrating download logs like
    "Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz".

.. testoutput::
    :options: +MOCK

    Column  Type
    ------  ----
    item    <class 'object'>

Custom PyTorch Datasets
~~~~~~~~~~~~~~~~~~~~~~~

If you have a custom PyTorch Dataset, you can migrate to Ray Data by converting the logic in ``__getitem__`` to Ray Data read and transform operations.

Any logic for reading data from cloud storage and disk can be replaced by one of the Ray Data ``read_*`` APIs, and any transformation logic can be applied as a :meth:`map <ray.data.Dataset.map>` call on the Dataset.

The following example shows a custom PyTorch Dataset, and what the analogous would look like with Ray Data.

.. note::

    Unlike PyTorch Map-style datasets, Ray Datasets aren't indexable.

.. tab-set::

    .. tab-item:: PyTorch Dataset

        .. testcode::

            import tempfile
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config

            from torchvision import transforms
            from torch.utils.data import Dataset
            from PIL import Image

            class ImageDataset(Dataset):
                def __init__(self, bucket_name: str, dir_path: str):
                    self.s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
                    self.bucket = self.s3.Bucket(bucket_name)
                    self.files = [obj.key for obj in self.bucket.objects.filter(Prefix=dir_path)]

                    self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((128, 128)),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

                def __len__(self):
                    return len(self.files)

                def __getitem__(self, idx):
                    img_name = self.files[idx]

                    # Infer the label from the file name.
                    last_slash_idx = img_name.rfind("/")
                    dot_idx = img_name.rfind(".")
                    label = int(img_name[last_slash_idx+1:dot_idx])

                    # Download the S3 file locally.
                    obj = self.bucket.Object(img_name)
                    tmp = tempfile.NamedTemporaryFile()
                    tmp_name = "{}.jpg".format(tmp.name)

                    with open(tmp_name, "wb") as f:
                        obj.download_fileobj(f)
                        f.flush()
                        f.close()
                        image = Image.open(tmp_name)

                    # Preprocess the image.
                    image = self.transform(image)

                    return image, label

            dataset = ImageDataset(bucket_name="ray-example-data", dir_path="batoidea/JPEGImages/")

    .. tab-item:: Ray Data

        .. testcode::

            import torchvision
            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/batoidea/JPEGImages", include_paths=True)

            # Extract the label from the file path.
            def extract_label(row: dict):
                filepath = row["path"]
                last_slash_idx = filepath.rfind("/")
                dot_idx = filepath.rfind('.')
                label = int(filepath[last_slash_idx+1:dot_idx])
                row["label"] = label
                return row

            transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((128, 128)),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

            # Preprocess the images.
            def transform_image(row: dict):
                row["transformed_image"] = transform(row["image"])
                return row

            # Map the transformations over the dataset.
            ds = ds.map(extract_label).map(transform_image)

PyTorch DataLoader
~~~~~~~~~~~~~~~~~~

The PyTorch DataLoader can be replaced by calling :meth:`Dataset.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>` to iterate over batches of the dataset.

The following table describes how the arguments for PyTorch DataLoader map to Ray Data. Note the behavior may not necessarily be identical. For exact semantics and usage, see the API reference for :meth:`iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`.

.. list-table::
   :header-rows: 1

   * - PyTorch DataLoader arguments
     - Ray Data API
   * - ``batch_size``
     - ``batch_size`` argument to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`
   * - ``shuffle``
     - ``local_shuffle_buffer_size`` argument to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`
   * - ``collate_fn``
     - ``collate_fn`` argument to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`
   * - ``sampler``
     - Not supported. Can be manually implemented after iterating through the dataset with :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`.
   * - ``batch_sampler``
     - Not supported. Can be manually implemented after iterating through the dataset with :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`.
   * - ``drop_last``
     - ``drop_last`` argument to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`
   * - ``num_workers``
     - Use ``prefetch_batches`` argument to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>` to indicate how many batches to prefetch. The number of prefetching threads are automatically configured according to ``prefetch_batches``.
   * - ``prefetch_factor``
     - Use ``prefetch_batches`` argument to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>` to indicate how many batches to prefetch. The number of prefetching threads are automatically configured according to ``prefetch_batches``.
   * - ``pin_memory``
     - Pass in ``device`` to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>` to get tensors that have already been moved to the correct device.


.. _working_with_images:

Working with Images
===================

With Ray Data, you can easily read and transform large image datasets.

This guide shows you how to:

* :ref:`Read images <reading_images>`
* :ref:`Transform images <transforming_images>`
* :ref:`Perform inference on images <performing_inference_on_images>`
* :ref:`Save images <saving_images>`

.. _reading_images:

Reading images
--------------

Ray Data can read images from a variety of formats.

To view the full list of supported file formats, see the
:ref:`Input/Output reference <input-output>`.

.. tab-set::

    .. tab-item:: Raw images

        To load raw images like JPEG files, call :func:`~ray.data.read_images`.

        .. note::

            :func:`~ray.data.read_images` uses
            `PIL <https://pillow.readthedocs.io/en/stable/index.html>`_. For a list of
            supported file formats, see
            `Image file formats <https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html>`_.

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/batoidea/JPEGImages")

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            image   numpy.ndarray(shape=(32, 32, 3), dtype=uint8)

    .. tab-item:: NumPy

        To load images stored in NumPy format, call :func:`~ray.data.read_numpy`.

        .. testcode::

            import ray

            ds = ray.data.read_numpy("s3://anonymous@air-example-data/cifar-10/images.npy")

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            data    numpy.ndarray(shape=(32, 32, 3), dtype=uint8)

    .. tab-item:: TFRecords

        Image datasets often contain ``tf.train.Example`` messages that look like this:

        .. code-block::

            features {
                feature {
                    key: "image"
                    value {
                        bytes_list {
                            value: ...  # Raw image bytes
                        }
                    }
                }
                feature {
                    key: "label"
                    value {
                        int64_list {
                            value: 3
                        }
                    }
                }
            }

        To load examples stored in this format, call :func:`~ray.data.read_tfrecords`.
        Then, call :meth:`~ray.data.Dataset.map` to decode the raw image bytes.

        .. testcode::

            import io
            from typing import Any, Dict
            import numpy as np
            from PIL import Image
            import ray

            def decode_bytes(row: Dict[str, Any]) -> Dict[str, Any]:
                data = row["image"]
                image = Image.open(io.BytesIO(data))
                row["image"] = np.array(image)
                return row

            ds = (
                ray.data.read_tfrecords(
                    "s3://anonymous@air-example-data/cifar-10/tfrecords"
                )
                .map(decode_bytes)
            )

            print(ds.schema())

        ..
            The following `testoutput` is mocked because the order of column names can
            be non-deterministic. For an example, see
            https://buildkite.com/ray-project/oss-ci-build-branch/builds/4849#01892c8b-0cd0-4432-bc9f-9f86fcd38edd.

        .. testoutput::
            :options: +MOCK

            Column  Type
            ------  ----
            image   numpy.ndarray(shape=(32, 32, 3), dtype=uint8)
            label   int64

    .. tab-item:: Parquet

        To load image data stored in Parquet files, call :func:`ray.data.read_parquet`.

        .. testcode::

            import ray

            ds = ray.data.read_parquet("s3://anonymous@air-example-data/cifar-10/parquet")

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            image   numpy.ndarray(shape=(32, 32, 3), dtype=uint8)
            label   int64


For more information on creating datasets, see :ref:`Loading Data <loading_data>`.

.. _transforming_images:

Transforming images
-------------------

To transform images, call :meth:`~ray.data.Dataset.map` or
:meth:`~ray.data.Dataset.map_batches`.

.. testcode::

    from typing import Any, Dict
    import numpy as np
    import ray

    def increase_brightness(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        batch["image"] = np.clip(batch["image"] + 4, 0, 255)
        return batch

    ds = (
        ray.data.read_images("s3://anonymous@ray-example-data/batoidea/JPEGImages")
        .map_batches(increase_brightness)
    )

For more information on transforming data, see
:ref:`Transforming data <transforming_data>`.

.. _performing_inference_on_images:

Performing inference on images
------------------------------

To perform inference with a pre-trained model, first load and transform your data.

.. testcode::

    from typing import Any, Dict
    from torchvision import transforms
    import ray

    def transform_image(row: Dict[str, Any]) -> Dict[str, Any]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])
        row["image"] = transform(row["image"])
        return row

    ds = (
        ray.data.read_images("s3://anonymous@ray-example-data/batoidea/JPEGImages")
        .map(transform_image)
    )

Next, implement a callable class that sets up and invokes your model.

.. testcode::

    import torch
    from torchvision import models

    class ImageClassifier:
        def __init__(self):
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
            self.model.eval()

        def __call__(self, batch):
            inputs = torch.from_numpy(batch["image"])
            with torch.inference_mode():
                outputs = self.model(inputs)
            return {"class": outputs.argmax(dim=1)}

Finally, call :meth:`Dataset.map_batches() <ray.data.Dataset.map_batches>`.

.. testcode::

    predictions = ds.map_batches(
        ImageClassifier,
        concurrency=2,
        batch_size=4
    )
    predictions.show(3)

.. testoutput::

    {'class': 118}
    {'class': 153}
    {'class': 296}

For more information on performing inference, see
:ref:`End-to-end: Offline Batch Inference <batch_inference_home>`
and :ref:`Stateful Transforms <stateful_transforms>`.

.. _saving_images:

Saving images
-------------

Save images with formats like PNG, Parquet, and NumPy. To view all supported formats,
see the :ref:`Input/Output reference <input-output>`.

.. tab-set::

    .. tab-item:: Images

        To save images as image files, call :meth:`~ray.data.Dataset.write_images`.

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            ds.write_images("/tmp/simple", column="image", file_format="png")

    .. tab-item:: Parquet

        To save images in Parquet files, call :meth:`~ray.data.Dataset.write_parquet`.

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            ds.write_parquet("/tmp/simple")


    .. tab-item:: NumPy

        To save images in a NumPy file, call :meth:`~ray.data.Dataset.write_numpy`.

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            ds.write_numpy("/tmp/simple", column="image")

For more information on saving data, see :ref:`Saving data <loading_data>`.


.. _data_user_guide:

===========
User Guides
===========

If you’re new to Ray Data, start with the
:ref:`Ray Data Quickstart <data_quickstart>`.
This user guide helps you navigate the Ray Data project and
show you how achieve several tasks.

.. toctree::
    :maxdepth: 2

    loading-data
    inspecting-data
    transforming-data
    iterating-over-data
    shuffling-data
    saving-data
    working-with-images
    working-with-text
    working-with-tensors
    working-with-pytorch
    monitoring-your-workload
    execution-configurations
    batch_inference
    performance-tips
    custom-datasource-example

.. _data_overview:

Ray Data Overview
=================

.. _data-intro:

..
  https://docs.google.com/drawings/d/16AwJeBNR46_TsrkOmMbGaBK7u-OPsf_V8fHjU-d2PPQ/edit

Ray Data is a scalable data processing library for ML workloads, particularly suited for the following workloads:

-  :ref:`Offline batch inference <batch_inference_overview>`
-  :ref:`Data preprocessing and ingest for ML training <ml_ingest_overview>`

It provides flexible and performant APIs for distributed data processing. For more details, see :ref:`Transforming Data <transforming_data>`.

Ray Data is built on top of Ray, so it scales effectively to large clusters and offers scheduling support for both CPU and GPU resources. Ray Data uses `streaming execution <https://www.anyscale.com/blog/streaming-distributed-execution-across-cpus-and-gpus>`__ to efficiently process large datasets.

Why choose Ray Data?
--------------------

.. dropdown:: Faster and cheaper for modern deep learning applications

    Ray Data is designed for deep learning applications that involve both CPU preprocessing and GPU inference. Ray Data streams working data from CPU preprocessing tasks to GPU inferencing or training tasks, allowing you to utilize both sets of resources concurrently.

    By using Ray Data, your GPUs are no longer idle during CPU computation, reducing overall cost of the batch inference job.

.. dropdown:: Cloud, framework, and data format agnostic

    Ray Data has no restrictions on cloud provider, ML framework, or data format.

    You can start a Ray cluster on AWS, GCP, or Azure clouds. You can use any ML framework of your choice, including PyTorch, HuggingFace, or Tensorflow. Ray Data also does not require a particular file format, and supports a :ref:`wide variety of formats <loading_data>` including Parquet, images, JSON, text, CSV, etc.

.. dropdown:: Out-of-the-box scaling on heterogeneous clusters

    Ray Data is built on Ray, so it easily scales on a heterogeneous cluster, which has different types of CPU and GPU machines. Code that works on one machine also runs on a large cluster without any changes.

    Ray Data can easily scale to hundreds of nodes to process hundreds of TB of data.

.. dropdown:: Unified API and backend for batch inference and ML training

    With Ray Data, you can express batch inference and ML training job directly under the same Ray Dataset API.


.. _batch_inference_overview:

Offline Batch Inference
-----------------------

Offline batch inference is a process for generating model predictions on a fixed set of input data. Ray Data offers an efficient and scalable solution for batch inference, providing faster execution and cost-effectiveness for deep learning applications. For more details on how to use Ray Data for offline batch inference, see the :ref:`batch inference user guide <batch_inference_home>`.

.. image:: images/stream-example.png
   :width: 650px
   :align: center

..
 https://docs.google.com/presentation/d/1l03C1-4jsujvEFZUM4JVNy8Ju8jnY5Lc_3q7MBWi2PQ/edit#slide=id.g230eb261ad2_0_0


How does Ray Data compare to other solutions for offline inference?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: Batch Services: AWS Batch, GCP Batch

    Cloud providers such as AWS, GCP, and Azure provide batch services to manage compute infrastructure for you. Each service uses the same process: you provide the code, and the service runs your code on each node in a cluster. However, while infrastructure management is necessary, it is often not enough. These services have limitations, such as a lack of software libraries to address optimized parallelization, efficient data transfer, and easy debugging. These solutions are suitable only for experienced users who can write their own optimized batch inference code.

    Ray Data abstracts away not only the infrastructure management, but also the sharding your dataset, the parallelization of the inference over these shards, and the transfer of data from storage to CPU to GPU.


.. dropdown:: Online inference solutions: Bento ML, Sagemaker Batch Transform

    Solutions like `Bento ML <https://www.bentoml.com/>`_, `Sagemaker Batch Transform <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html>`_, or :ref:`Ray Serve <rayserve>` provide APIs to make it easy to write performant inference code and can abstract away infrastructure complexities. But they are designed for online inference rather than offline batch inference, which are two different problems with different sets of requirements. These solutions introduce additional complexity like HTTP, and cannot effectively handle large datasets leading inference service providers like `Bento ML to integrating with Apache Spark <https://modelserving.com/blog/unifying-real-time-and-batch-inference-with-bentoml-and-spark>`_ for offline inference.

    Ray Data is built for offline batch jobs, without all the extra complexities of starting servers or sending HTTP requests.

    For a more detailed performance comparison between Ray Data and Sagemaker Batch Transform, see `Offline Batch Inference: Comparing Ray, Apache Spark, and SageMaker <https://www.anyscale.com/blog/offline-batch-inference-comparing-ray-apache-spark-and-sagemaker>`_.

.. dropdown:: Distributed Data Processing Frameworks: Apache Spark

    Ray Data handles many of the same batch processing workloads as `Apache Spark <https://spark.apache.org/>`_, but with a streaming paradigm that is better suited for GPU workloads for deep learning inference.

    Ray Data doesn't have a SQL interface and isn't meant as a replacement for generic ETL pipelines like Spark.

    For a more detailed performance comarison between Ray Data and Apache Spark, see `Offline Batch Inference: Comparing Ray, Apache Spark, and SageMaker <https://www.anyscale.com/blog/offline-batch-inference-comparing-ray-apache-spark-and-sagemaker>`_.

Batch inference case studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `ByteDance scales offline inference with multi-modal LLMs to 200 TB on Ray Data <https://www.anyscale.com/blog/how-bytedance-scales-offline-inference-with-multi-modal-llms-to-200TB-data>`_
- `Spotify's new ML platform built on Ray Data for batch inference <https://engineering.atspotify.com/2023/02/unleashing-ml-innovation-at-spotify-with-ray/>`_
- `Sewer AI speeds up object detection on videos 3x using Ray Data <https://www.anyscale.com/blog/inspecting-sewer-line-safety-using-thousands-of-hours-of-video>`_

.. _ml_ingest_overview:

Preprocessing and ingest for ML training
----------------------------------------

Use Ray Data to load and preprocess data for distributed :ref:`ML training pipelines <train-docs>` in a streaming fashion.
Key supported features for distributed training include:

- Fast out-of-memory recovery
- Support for heterogeneous clusters
- No dropped rows during distributed dataset iteration

Ray Data serves as a last-mile bridge from storage or ETL pipeline outputs to distributed
applications and libraries in Ray. Use it for unstructured data processing. For more details
on how to use Ray Data for preprocessing and ingest for ML training, see
:ref:`Data loading for ML training <data-ingest-torch>`.

.. image:: images/dataset-loading-1.svg
   :width: 650px
   :align: center

..
  https://docs.google.com/presentation/d/1l03C1-4jsujvEFZUM4JVNy8Ju8jnY5Lc_3q7MBWi2PQ/edit


How does Ray Data compare to other solutions for ML training ingest?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: PyTorch Dataset and DataLoader

    * **Framework-agnostic:** Datasets is framework-agnostic and portable between different distributed training frameworks, while `Torch datasets <https://pytorch.org/docs/stable/data.html>`__ are specific to Torch.
    * **No built-in IO layer:** Torch datasets do not have an I/O layer for common file formats or in-memory exchange with other frameworks; users need to bring in other libraries and roll this integration themselves.
    * **Generic distributed data processing:** Datasets is more general: it can handle generic distributed operations, including global per-epoch shuffling, which would otherwise have to be implemented by stitching together two separate systems. Torch datasets would require such stitching for anything more involved than batch-based preprocessing, and does not natively support shuffling across worker shards. See our `blog post <https://www.anyscale.com/blog/deep-dive-data-ingest-in-a-third-generation-ml-architecture>`__ on why this shared infrastructure is important for 3rd generation ML architectures.
    * **Lower overhead:** Datasets is lower overhead: it supports zero-copy exchange between processes, in contrast to the multi-processing-based pipelines of Torch datasets.


.. dropdown:: TensorFlow Dataset

    * **Framework-agnostic:** Datasets is framework-agnostic and portable between different distributed training frameworks, while `TensorFlow datasets <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`__ is specific to TensorFlow.
    * **Unified single-node and distributed:** Datasets unifies single and multi-node training under the same abstraction. TensorFlow datasets presents `separate concepts <https://www.tensorflow.org/api_docs/python/tf/distribute/DistributedDataset>`__ for distributed data loading and prevents code from being seamlessly scaled to larger clusters.
    * **Generic distributed data processing:** Datasets is more general: it can handle generic distributed operations, including global per-epoch shuffling, which would otherwise have to be implemented by stitching together two separate systems. TensorFlow datasets would require such stitching for anything more involved than basic preprocessing, and does not natively support full-shuffling across worker shards; only file interleaving is supported. See our `blog post <https://www.anyscale.com/blog/deep-dive-data-ingest-in-a-third-generation-ml-architecture>`__ on why this shared infrastructure is important for 3rd generation ML architectures.
    * **Lower overhead:** Datasets is lower overhead: it supports zero-copy exchange between processes, in contrast to the multi-processing-based pipelines of TensorFlow datasets.

.. dropdown:: Petastorm

    * **Supported data types:** `Petastorm <https://github.com/uber/petastorm>`__ only supports Parquet data, while Ray Data supports many file formats.
    * **Lower overhead:** Datasets is lower overhead: it supports zero-copy exchange between processes, in contrast to the multi-processing-based pipelines used by Petastorm.
    * **No data processing:** Petastorm does not expose any data processing APIs.


.. dropdown:: NVTabular

    * **Supported data types:** `NVTabular <https://github.com/NVIDIA-Merlin/NVTabular>`__ only supports tabular (Parquet, CSV, Avro) data, while Ray Data supports many other file formats.
    * **Lower overhead:** Datasets is lower overhead: it supports zero-copy exchange between processes, in contrast to the multi-processing-based pipelines used by NVTabular.
    * **Heterogeneous compute:** NVTabular doesn't support mixing heterogeneous resources in dataset transforms (e.g. both CPU and GPU transformations), while Ray Data supports this.

ML training ingest case studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `Pinterest uses Ray Data to do last mile data processing for model training <https://medium.com/pinterest-engineering/last-mile-data-processing-with-ray-629affbf34ff>`_
- `DoorDash elevates model training with Ray Data <https://raysummit.anyscale.com/agenda/sessions/144>`_
- `Instacart builds distributed machine learning model training on Ray Data <https://tech.instacart.com/distributed-machine-learning-at-instacart-4b11d7569423>`_
- `Predibase speeds up image augmentation for model training using Ray Data <https://predibase.com/blog/ludwig-v0-7-fine-tuning-pretrained-image-and-text-models-50x-faster-and>`_


.. _execution_configurations:

========================
Execution Configurations
========================

Ray Data provides a number of configurations that control various aspects
of Ray Dataset execution. You can modify these configurations by using
:class:`~ray.data.ExecutionOptions` and :class:`~ray.data.DataContext`. 
This guide describes the most important of these configurations and when to use them.

Configuring :class:`~ray.data.ExecutionOptions`
===============================================

The :class:`~ray.data.ExecutionOptions` class is used to configure options during Ray Dataset execution.
To use it, modify the attributes in the current :class:`~ray.data.DataContext` object's `execution_options`. For example:

.. testcode::
   :hide:
    
   import ray

.. testcode::

   ctx = ray.data.DataContext.get_current()
   ctx.execution_options.verbose_progress = True

* `resource_limits`: Set a soft limit on the resource usage during execution. For example, if there are other parts of the code which require some minimum amount of resources, you may want to limit the amount of resources that Ray Data uses. Auto-detected by default.
* `exclude_resources`: Amount of resources to exclude from Ray Data. Set this if you have other workloads running on the same cluster. Note: 

  * If you're using Ray Data with Ray Train, training resources are automatically excluded. Otherwise, off by default.
  * For each resource type, you can't set both ``resource_limits`` and ``exclude_resources``.

* `locality_with_output`: Set this to prefer running tasks on the same node as the output node (node driving the execution). It can also be set to a list of node ids to spread the outputs across those nodes. This parameter applies to both :meth:`~ray.data.Dataset.map` and :meth:`~ray.data.Dataset.streaming_split` operations. This setting is useful if you know you are consuming the output data directly on the consumer node (such as for ML training ingest). However, other use cases can incur a performance penalty with this setting. Off by default.
* `preserve_order`: Set this to preserve the ordering between blocks processed by operators under the streaming executor. Off by default.
* `actor_locality_enabled`: Whether to enable locality-aware task dispatch to actors. This parameter applies to stateful :meth:`~ray.data.Dataset.map` operations. This setting is useful if you know you are consuming the output data directly on the consumer node (such as for ML batch inference). However, other use cases can incur a performance penalty with this setting. Off by default.
* `verbose_progress`: Whether to report progress individually per operator. By default, only AllToAll operators and global progress is reported. This option is useful for performance debugging. On by default.

For more details on each of the preceding options, see :class:`~ray.data.ExecutionOptions`.

Configuring :class:`~ray.data.DataContext`
==========================================
The :class:`~ray.data.DataContext` class is used to configure more general options for Ray Data usage, such as observability/logging options,
error handling/retry behavior, and internal data formats. To use it, modify the attributes in the current :class:`~ray.data.DataContext` object. For example:

.. testcode::
	   :hide:
	    
	   import ray 
	    
.. testcode::

   ctx = ray.data.DataContext.get_current()
   ctx.verbose_stats_logs = True

Many of the options in :class:`~ray.data.DataContext` are intended for advanced use cases or debugging, 
and most users shouldn't need to modify them. However, some of the most important options are:

* `max_errored_blocks`: Max number of blocks that are allowed to have errors, unlimited if negative. This option allows application-level exceptions in block processing tasks. These exceptions may be caused by UDFs (for example, due to corrupted data samples) or IO errors. Data in the failed blocks are dropped. This option can be useful to prevent a long-running job from failing due to a small number of bad blocks. By default, no retries are allowed.
* `write_file_retry_on_errors`: A list of sub-strings of error messages that should trigger a retry when writing files. This is useful for handling transient errors when writing to remote storage systems. By default, retries on common transient AWS S3 errors.
* `verbose_stats_logs`: Whether stats logs should be verbose. This includes fields such as ``extra_metrics`` in the stats output, which are excluded by default. Off by default.
* `log_internal_stack_trace_to_stdout`: Whether to include internal Ray Data/Ray Core code stack frames when logging to ``stdout``. The full stack trace is always written to the Ray Data log file. Off by default.

For more details on each of the preceding options, see :class:`~ray.data.DataContext`.

.. _datasets_scheduling:

==================
Ray Data Internals
==================

This guide describes the implementation of Ray Data. The intended audience is advanced
users and Ray Data developers.

For a gentler introduction to Ray Data, see :ref:`Quickstart <data_quickstart>`.

.. _dataset_concept:

Key concepts
============

Datasets and blocks
-------------------

Datasets
~~~~~~~~

:class:`Dataset <ray.data.Dataset>` is the main user-facing Python API. It represents a 
distributed data collection, and defines data loading and processing operations. You 
typically use the API in this way:

1. Create a Ray Dataset from external storage or in-memory data.
2. Apply transformations to the data. 
3. Write the outputs to external storage or feed the outputs to training workers. 

Blocks
~~~~~~

A *block* is the basic unit of data bulk that Ray Data stores in the object store and 
transfers over the network. Each block contains a disjoint subset of rows, and Ray Data 
loads and transforms these blocks in parallel. 

The following figure visualizes a dataset with three blocks, each holding 1000 rows.
Ray Data holds the :class:`~ray.data.Dataset` on the process that triggers execution 
(which is usually the driver) and stores the blocks as objects in Ray's shared-memory 
:ref:`object store <objects-in-ray>`.

.. image:: images/dataset-arch.svg

..
  https://docs.google.com/drawings/d/1PmbDvHRfVthme9XD7EYM-LIHPXtHdOfjCbc1SCsM64k/edit

Block formats
~~~~~~~~~~~~~

Blocks are Arrow tables or `pandas` DataFrames. Generally, blocks are Arrow tables 
unless Arrow can’t represent your data. 

The block format doesn’t affect the type of data returned by APIs like 
:meth:`~ray.data.Dataset.iter_batches`.

Block size limiting
~~~~~~~~~~~~~~~~~~~

Ray Data bounds block sizes to avoid excessive communication overhead and prevent 
out-of-memory errors. Small blocks are good for latency and more streamed execution, 
while large blocks reduce scheduler and communication overhead. The default range 
attempts to make a good tradeoff for most jobs.

Ray Data attempts to bound block sizes between 1 MiB and 128 MiB. To change the block 
size range, configure the ``target_min_block_size`` and  ``target_max_block_size`` 
attributes of :class:`~ray.data.context.DataContext`.

.. testcode::

    import ray

    ctx = ray.data.DataContext.get_current()
    ctx.target_min_block_size = 1 * 1024 * 1024
    ctx.target_max_block_size = 128 * 1024 * 1024

Dynamic block splitting
~~~~~~~~~~~~~~~~~~~~~~~

If a block is larger than 192 MiB (50% more than the target max size), Ray Data 
dynamically splits the block into smaller blocks. 

To change the size at which Ray Data splits blocks, configure 
``MAX_SAFE_BLOCK_SIZE_FACTOR``. The default value is 1.5.

.. testcode::

    import ray

    ray.data.context.MAX_SAFE_BLOCK_SIZE_FACTOR = 1.5

Ray Data can’t split rows. So, if your dataset contains large rows (for example, large 
images), then Ray Data can’t bound the block size.

Operators, plans, and planning
------------------------------

Operators
~~~~~~~~~

There are two types of operators: *logical operators* and *physical operators*. Logical 
operators are stateless objects that describe “what” to do. Physical operators are 
stateful objects that describe “how” to do it. An example of a logical operator is 
``ReadOp``, and an example of a physical operator is ``TaskPoolMapOperator``.

Plans
~~~~~

A *logical plan* is a series of logical operators, and a *physical plan* is a series of 
physical operators. When you call APIs like :func:`ray.data.read_images` and 
:meth:`ray.data.Dataset.map_batches`, Ray Data produces a logical plan. When execution 
starts, the planner generates a corresponding physical plan. 

The planner
~~~~~~~~~~~

The Ray Data planner translates logical operators to one or more physical operators. For 
example, the planner translates the ``ReadOp`` logical operator into two physical 
operators: an ``InputDataBuffer`` and ``TaskPoolMapOperator``. Whereas the ``ReadOp``
logical operator only describes the input data, the ``TaskPoolMapOperator`` physical 
operator actually launches tasks to read the data.

Plan optimization
~~~~~~~~~~~~~~~~~

Ray Data applies optimizations to both logical and physical plans. For example, the 
``OperatorFusionRule`` combines a chain of physical map operators into a single map 
operator. This prevents unnecessary serialization between map operators.

To add custom optimization rules, implement a class that extends ``Rule`` and configure
``DEFAULT_LOGICAL_RULES`` or ``DEFAULT_PHYSICAL_RULES``.

.. testcode::

    import ray
    from ray.data._internal.logical.interfaces import Rule

    class CustomRule(Rule):
        def apply(self, plan):
            ...

    ray.data._internal.logical.optimizers.DEFAULT_LOGICAL_RULES.append(CustomRule)

Types of physical operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Physical operators take in a stream of block references and output another stream of 
block references. Some physical operators launch Ray Tasks and Actors to transform  
the blocks, and others only manipulate the references.

``MapOperator`` is the most common operator. All read, transform, and write operations 
are implemented with it. To process data, ``MapOperator`` implementations use either Ray 
Tasks or Ray Actors.

Non-map operators include ``OutputSplitter`` and ``LimitOperator``. These two operators 
manipulate references to data, but don’t launch tasks or modify the underlying data. 

Execution
---------

The executor
~~~~~~~~~~~~

The *executor* schedules tasks and moves data between physical operators.

The executor and operators are located on the process where dataset execution starts. 
For batch inference jobs, this process is usually the driver. For training jobs, the 
executor runs on a special actor called ``SplitCoordinator`` which handles 
:meth:`~ray.data.Dataset.streaming_split`.

Tasks and actors launched by operators are scheduled across the cluster, and outputs are 
stored in Ray’s distributed object store. The executor manipulates references to 
objects, and doesn’t fetch the underlying data itself to the executor.

Out queues
~~~~~~~~~~

Each physical operator has an associated *out queue*. When a physical operator produces 
outputs, the executor moves the outputs to the operator’s out queue. 

.. _streaming_execution:

Streaming execution
~~~~~~~~~~~~~~~~~~~

In contrast to bulk synchronous execution, Ray Data’s streaming execution doesn’t wait 
for one operator to complete to start the next. Each operator takes in and outputs a 
stream of blocks. This approach allows you to process datasets that are too large to fit 
in your cluster’s memory.

The scheduling loop
~~~~~~~~~~~~~~~~~~~

The executor runs a loop. Each step works like this:

1. Wait until running tasks and actors have new outputs.
2. Move new outputs into the appropriate operator out queues.
3. Choose some operators and assign new inputs to them. These operator process the new 
   inputs either by launching new tasks or manipulating metadata.

Choosing the best operator to assign inputs is one of the most important decisions in 
Ray Data. This decision is critical to the performance, stability, and scalability of a 
Ray Data job. The executor can schedule an operator if the operator satisfies the 
following conditions:

* The operator has inputs.
* There are adequate resources available.
* The operator isn’t backpressured. 

If there are multiple viable operators, the executor chooses the operator with the 
smallest out queue. 

Scheduling
==========

Ray Data uses Ray Core for execution. Below is a summary of the :ref:`scheduling strategy <ray-scheduling-strategies>` for Ray Data:

* The ``SPREAD`` scheduling strategy ensures that data blocks and map tasks are evenly balanced across the cluster.
* Dataset tasks ignore placement groups by default, see :ref:`Ray Data and Placement Groups <datasets_pg>`.
* Map operations use the ``SPREAD`` scheduling strategy if the total argument size is less than 50 MB; otherwise, they use the ``DEFAULT`` scheduling strategy.
* Read operations use the ``SPREAD`` scheduling strategy.
* All other operations, such as split, sort, and shuffle, use the ``DEFAULT`` scheduling strategy.

.. _datasets_pg:

Ray Data and placement groups
-----------------------------

By default, Ray Data configures its tasks and actors to use the cluster-default scheduling strategy (``"DEFAULT"``). You can inspect this configuration variable here:
:class:`ray.data.DataContext.get_current().scheduling_strategy <ray.data.DataContext>`. This scheduling strategy schedules these Tasks and Actors outside any present
placement group. To use current placement group resources specifically for Ray Data, set ``ray.data.DataContext.get_current().scheduling_strategy = None``.

Consider this override only for advanced use cases to improve performance predictability. The general recommendation is to let Ray Data run outside placement groups.

.. _datasets_tune:

Ray Data and Tune
-----------------

When using Ray Data in conjunction with :ref:`Ray Tune <tune-main>`, it's important to ensure there are enough free CPUs for Ray Data to run on. By default, Tune tries to fully utilize cluster CPUs. This can prevent Ray Data from scheduling tasks, reducing performance or causing workloads to hang.

To ensure CPU resources are always available for Ray Data execution, limit the number of concurrent Tune trials with the ``max_concurrent_trials`` Tune option.

.. literalinclude:: ./doc_code/key_concepts.py
  :language: python
  :start-after: __resource_allocation_1_begin__
  :end-before: __resource_allocation_1_end__

Memory Management
=================

This section describes how Ray Data manages execution and object store memory.

Execution Memory
----------------

During execution, a task can read multiple input blocks, and write multiple output blocks. Input and output blocks consume both worker heap memory and shared memory through Ray's object store.
Ray caps object store memory usage by spilling to disk, but excessive worker heap memory usage can cause out-of-memory errors.

For more information on tuning memory usage and preventing out-of-memory errors, see the :ref:`performance guide <data_memory>`.

Object Store Memory
-------------------

Ray Data uses the Ray object store to store data blocks, which means it inherits the memory management features of the Ray object store. This section discusses the relevant features:

* Object Spilling: Since Ray Data uses the Ray object store to store data blocks, any blocks that can't fit into object store memory are automatically spilled to disk. The objects are automatically reloaded when needed by downstream compute tasks:
* Locality Scheduling: Ray preferentially schedules compute tasks on nodes that already have a local copy of the object, reducing the need to transfer objects between nodes in the cluster.
* Reference Counting: Dataset blocks are kept alive by object store reference counting as long as there is any Dataset that references them. To free memory, delete any Python references to the Dataset object.


.. _batch_inference_home:

End-to-end: Offline Batch Inference
===================================

Offline batch inference is a process for generating model predictions on a fixed set of input data. Ray Data offers an efficient and scalable solution for batch inference, providing faster execution and cost-effectiveness for deep learning applications.

For an overview on why you should use Ray Data for offline batch inference, and how it compares to alternatives, see the :ref:`Ray Data Overview <data_overview>`.

.. figure:: images/batch_inference.png


.. _batch_inference_quickstart:

Quickstart
----------
To start, install Ray Data:

.. code-block:: bash

    pip install -U "ray[data]"

Using Ray Data for offline inference involves four basic steps:

- **Step 1:** Load your data into a Ray Dataset. Ray Data supports many different datasources and formats. For more details, see :ref:`Loading Data <loading_data>`.
- **Step 2:** Define a Python class to load the pre-trained model.
- **Step 3:** Transform your dataset using the pre-trained model by calling :meth:`ds.map_batches() <ray.data.Dataset.map_batches>`. For more details, see :ref:`Transforming Data <transforming_data>`.
- **Step 4:** Get the final predictions by either iterating through the output or saving the results. For more details, see the :ref:`Iterating over data <iterating-over-data>` and :ref:`Saving data <saving-data>` user guides.

For more in-depth examples for your use case, see :doc:`the batch inference examples</data/examples>`.
For how to configure batch inference, see :ref:`the configuration guide<batch_inference_configuration>`.

.. tab-set::

    .. tab-item:: HuggingFace
        :sync: HuggingFace

        .. testcode::

            from typing import Dict
            import numpy as np

            import ray

            # Step 1: Create a Ray Dataset from in-memory Numpy arrays.
            # You can also create a Ray Dataset from many other sources and file
            # formats.
            ds = ray.data.from_numpy(np.asarray(["Complete this", "for me"]))

            # Step 2: Define a Predictor class for inference.
            # Use a class to initialize the model just once in `__init__`
            # and re-use it for inference across multiple batches.
            class HuggingFacePredictor:
                def __init__(self):
                    from transformers import pipeline
                    # Initialize a pre-trained GPT2 Huggingface pipeline.
                    self.model = pipeline("text-generation", model="gpt2")

                # Logic for inference on 1 batch of data.
                def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
                    # Get the predictions from the input batch.
                    predictions = self.model(list(batch["data"]), max_length=20, num_return_sequences=1)
                    # `predictions` is a list of length-one lists. For example:
                    # [[{'generated_text': 'output_1'}], ..., [{'generated_text': 'output_2'}]]
                    # Modify the output to get it into the following format instead:
                    # ['output_1', 'output_2']
                    batch["output"] = [sequences[0]["generated_text"] for sequences in predictions]
                    return batch

            # Step 2: Map the Predictor over the Dataset to get predictions.
            # Use 2 parallel actors for inference. Each actor predicts on a
            # different partition of data.
            predictions = ds.map_batches(HuggingFacePredictor, concurrency=2)
            # Step 3: Show one prediction output.
            predictions.show(limit=1)

        .. testoutput::
            :options: +MOCK

            {'data': 'Complete this', 'output': 'Complete this information or purchase any item from this site.\n\nAll purchases are final and non-'}


    .. tab-item:: PyTorch
        :sync: PyTorch

        .. testcode::

            from typing import Dict
            import numpy as np
            import torch
            import torch.nn as nn

            import ray

            # Step 1: Create a Ray Dataset from in-memory Numpy arrays.
            # You can also create a Ray Dataset from many other sources and file
            # formats.
            ds = ray.data.from_numpy(np.ones((1, 100)))

            # Step 2: Define a Predictor class for inference.
            # Use a class to initialize the model just once in `__init__`
            # and re-use it for inference across multiple batches.
            class TorchPredictor:
                def __init__(self):
                    # Load a dummy neural network.
                    # Set `self.model` to your pre-trained PyTorch model.
                    self.model = nn.Sequential(
                        nn.Linear(in_features=100, out_features=1),
                        nn.Sigmoid(),
                    )
                    self.model.eval()

                # Logic for inference on 1 batch of data.
                def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                    tensor = torch.as_tensor(batch["data"], dtype=torch.float32)
                    with torch.inference_mode():
                        # Get the predictions from the input batch.
                        return {"output": self.model(tensor).numpy()}

            # Step 2: Map the Predictor over the Dataset to get predictions.
            # Use 2 parallel actors for inference. Each actor predicts on a
            # different partition of data.
            predictions = ds.map_batches(TorchPredictor, concurrency=2)
            # Step 3: Show one prediction output.
            predictions.show(limit=1)

        .. testoutput::
            :options: +MOCK

            {'output': array([0.5590901], dtype=float32)}

    .. tab-item:: TensorFlow
        :sync: TensorFlow

        .. testcode::

            from typing import Dict
            import numpy as np

            import ray

            # Step 1: Create a Ray Dataset from in-memory Numpy arrays.
            # You can also create a Ray Dataset from many other sources and file
            # formats.
            ds = ray.data.from_numpy(np.ones((1, 100)))

            # Step 2: Define a Predictor class for inference.
            # Use a class to initialize the model just once in `__init__`
            # and re-use it for inference across multiple batches.
            class TFPredictor:
                def __init__(self):
                    from tensorflow import keras

                    # Load a dummy neural network.
                    # Set `self.model` to your pre-trained Keras model.
                    input_layer = keras.Input(shape=(100,))
                    output_layer = keras.layers.Dense(1, activation="sigmoid")
                    self.model = keras.Sequential([input_layer, output_layer])

                # Logic for inference on 1 batch of data.
                def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                    # Get the predictions from the input batch.
                    return {"output": self.model(batch["data"]).numpy()}

            # Step 2: Map the Predictor over the Dataset to get predictions.
            # Use 2 parallel actors for inference. Each actor predicts on a
            # different partition of data.
            predictions = ds.map_batches(TFPredictor, concurrency=2)
             # Step 3: Show one prediction output.
            predictions.show(limit=1)

        .. testoutput::
            :options: +MOCK

            {'output': array([0.625576], dtype=float32)}

.. _batch_inference_configuration:

Configuration and troubleshooting
---------------------------------

.. _batch_inference_gpu:

Using GPUs for inference
~~~~~~~~~~~~~~~~~~~~~~~~

To use GPUs for inference, make the following changes to your code:

1. Update the class implementation to move the model and data to and from GPU.
2. Specify ``num_gpus=1`` in the :meth:`ds.map_batches() <ray.data.Dataset.map_batches>` call to indicate that each actor should use 1 GPU.
3. Specify a ``batch_size`` for inference. For more details on how to configure the batch size, see :ref:`Configuring Batch Size <batch_inference_batch_size>`.

The remaining is the same as the :ref:`Quickstart <batch_inference_quickstart>`.

.. tab-set::

    .. tab-item:: HuggingFace
        :sync: HuggingFace

        .. testcode::

            from typing import Dict
            import numpy as np

            import ray

            ds = ray.data.from_numpy(np.asarray(["Complete this", "for me"]))

            class HuggingFacePredictor:
                def __init__(self):
                    from transformers import pipeline
                    # Set "cuda:0" as the device so the Huggingface pipeline uses GPU.
                    self.model = pipeline("text-generation", model="gpt2", device="cuda:0")

                def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
                    predictions = self.model(list(batch["data"]), max_length=20, num_return_sequences=1)
                    batch["output"] = [sequences[0]["generated_text"] for sequences in predictions]
                    return batch

            # Use 2 actors, each actor using 1 GPU. 2 GPUs total.
            predictions = ds.map_batches(
                HuggingFacePredictor,
                num_gpus=1,
                # Specify the batch size for inference.
                # Increase this for larger datasets.
                batch_size=1,
                # Set the concurrency to the number of GPUs in your cluster.
                concurrency=2,
                )
            predictions.show(limit=1)

        .. testoutput::
            :options: +MOCK

            {'data': 'Complete this', 'output': 'Complete this poll. Which one do you think holds the most promise for you?\n\nThank you'}


    .. tab-item:: PyTorch
        :sync: PyTorch

        .. testcode::

            from typing import Dict
            import numpy as np
            import torch
            import torch.nn as nn

            import ray

            ds = ray.data.from_numpy(np.ones((1, 100)))

            class TorchPredictor:
                def __init__(self):
                    # Move the neural network to GPU device by specifying "cuda".
                    self.model = nn.Sequential(
                        nn.Linear(in_features=100, out_features=1),
                        nn.Sigmoid(),
                    ).cuda()
                    self.model.eval()

                def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                    # Move the input batch to GPU device by specifying "cuda".
                    tensor = torch.as_tensor(batch["data"], dtype=torch.float32, device="cuda")
                    with torch.inference_mode():
                        # Move the prediction output back to CPU before returning.
                        return {"output": self.model(tensor).cpu().numpy()}

            # Use 2 actors, each actor using 1 GPU. 2 GPUs total.
            predictions = ds.map_batches(
                TorchPredictor,
                num_gpus=1,
                # Specify the batch size for inference.
                # Increase this for larger datasets.
                batch_size=1,
                # Set the concurrency to the number of GPUs in your cluster.
                concurrency=2,
                )
            predictions.show(limit=1)

        .. testoutput::
            :options: +MOCK

            {'output': array([0.5590901], dtype=float32)}

    .. tab-item:: TensorFlow
        :sync: TensorFlow

        .. testcode::

            from typing import Dict
            import numpy as np

            import ray

            ds = ray.data.from_numpy(np.ones((1, 100)))

            class TFPredictor:
                def __init__(self):
                    import tensorflow as tf
                    from tensorflow import keras

                    # Move the neural network to GPU by specifying the GPU device.
                    with tf.device("GPU:0"):
                        input_layer = keras.Input(shape=(100,))
                        output_layer = keras.layers.Dense(1, activation="sigmoid")
                        self.model = keras.Sequential([input_layer, output_layer])

                def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                    import tensorflow as tf

                    # Move the input batch to GPU by specifying GPU device.
                    with tf.device("GPU:0"):
                        return {"output": self.model(batch["data"]).numpy()}

            # Use 2 actors, each actor using 1 GPU. 2 GPUs total.
            predictions = ds.map_batches(
                TFPredictor,
                num_gpus=1,
                # Specify the batch size for inference.
                # Increase this for larger datasets.
                batch_size=1,
                # Set the concurrency to the number of GPUs in your cluster.
                concurrency=2,
            )
            predictions.show(limit=1)

        .. testoutput::
            :options: +MOCK

            {'output': array([0.625576], dtype=float32)}

.. _batch_inference_batch_size:

Configuring Batch Size
~~~~~~~~~~~~~~~~~~~~~~

Configure the size of the input batch that's passed to ``__call__`` by setting the ``batch_size`` argument for :meth:`ds.map_batches() <ray.data.Dataset.map_batches>`

Increasing batch size results in faster execution because inference is a vectorized operation. For GPU inference, increasing batch size increases GPU utilization. Set the batch size to as large possible without running out of memory. If you encounter out-of-memory errors, decreasing ``batch_size`` may help.

.. testcode::

    import numpy as np

    import ray

    ds = ray.data.from_numpy(np.ones((10, 100)))

    def assert_batch(batch: Dict[str, np.ndarray]):
        assert len(batch) == 2
        return batch

    # Specify that each input batch should be of size 2.
    ds.map_batches(assert_batch, batch_size=2)

.. caution::
  The default ``batch_size`` of ``4096`` may be too large for datasets with large rows
  (for example, tables with many columns or a collection of large images).

Handling GPU out-of-memory failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you run into CUDA out-of-memory issues, your batch size is likely too large. Decrease 
the batch size by following :ref:`these steps <batch_inference_batch_size>`. If your 
batch size is already set to 1, then use either a smaller model or GPU devices with more 
memory.

For advanced users working with large models, you can use model parallelism to shard the model across multiple GPUs.

Optimizing expensive CPU preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your workload involves expensive CPU preprocessing in addition to model inference, you can optimize throughput by separating the preprocessing and inference logic into separate operations. This separation allows inference on batch :math:`N` to execute concurrently with preprocessing on batch :math:`N+1`.

For an example where preprocessing is done in a separate `map` call, see :doc:`Image Classification Batch Inference with PyTorch ResNet18 </data/examples/pytorch_resnet_batch_prediction>`.

Handling CPU out-of-memory failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you run out of CPU RAM, you likely have too many model replicas that are running concurrently on the same node. For example, if a model
uses 5 GB of RAM when created / run, and a machine has 16 GB of RAM total, then no more
than three of these models can be run at the same time. The default resource assignments
of one CPU per task/actor might lead to `OutOfMemoryError` from Ray in this situation.

Suppose your cluster has 4 nodes, each with 16 CPUs. To limit to at most
3 of these actors per node, you can override the CPU or memory:

.. testcode::
    :skipif: True

    from typing import Dict
    import numpy as np

    import ray

    ds = ray.data.from_numpy(np.asarray(["Complete this", "for me"]))

    class HuggingFacePredictor:
        def __init__(self):
            from transformers import pipeline
            self.model = pipeline("text-generation", model="gpt2")

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            predictions = self.model(list(batch["data"]), max_length=20, num_return_sequences=1)
            batch["output"] = [sequences[0]["generated_text"] for sequences in predictions]
            return batch

    predictions = ds.map_batches(
        HuggingFacePredictor,
        # Require 5 CPUs per actor (so at most 3 can fit per 16 CPU node).
        num_cpus=5,
        # 3 actors per node, with 4 nodes in the cluster means concurrency of 12.
        concurrency=12,
        )
    predictions.show(limit=1)


.. _data_quickstart:

Quickstart
==========

Learn about :class:`Dataset <ray.data.Dataset>` and the capabilities it provides.

This guide provides a lightweight introduction to:

* :ref:`Loading data <loading_key_concept>`
* :ref:`Transforming data <transforming_key_concept>`
* :ref:`Consuming data <consuming_key_concept>`
* :ref:`Saving data <saving_key_concept>`

Datasets
--------

Ray Data's main abstraction is a :class:`Dataset <ray.data.Dataset>`, which
is a distributed data collection. Datasets are designed for machine learning, and they
can represent data collections that exceed a single machine's memory.

.. _loading_key_concept:

Loading data
------------

Create datasets from on-disk files, Python objects, and cloud storage services like S3.
Ray Data can read from any `filesystem supported by Arrow
<http://arrow.apache.org/docs/python/generated/pyarrow.fs.FileSystem.html>`__.

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")
    ds.show(limit=1)

.. testoutput::

    {'sepal length (cm)': 5.1, 'sepal width (cm)': 3.5, 'petal length (cm)': 1.4, 'petal width (cm)': 0.2, 'target': 0}

To learn more about creating datasets, read :ref:`Loading data <loading_data>`.

.. _transforming_key_concept:

Transforming data
-----------------

Apply user-defined functions (UDFs) to transform datasets. Ray executes transformations
in parallel for performance.

.. testcode::

    from typing import Dict
    import numpy as np

    # Compute a "petal area" attribute.
    def transform_batch(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        vec_a = batch["petal length (cm)"]
        vec_b = batch["petal width (cm)"]
        batch["petal area (cm^2)"] = vec_a * vec_b
        return batch

    transformed_ds = ds.map_batches(transform_batch)
    print(transformed_ds.materialize())

.. testoutput::

    MaterializedDataset(
       num_blocks=...,
       num_rows=150,
       schema={
          sepal length (cm): double,
          sepal width (cm): double,
          petal length (cm): double,
          petal width (cm): double,
          target: int64,
          petal area (cm^2): double
       }
    )

To learn more about transforming datasets, read
:ref:`Transforming data <transforming_data>`.

.. _consuming_key_concept:

Consuming data
--------------

Pass datasets to Ray Tasks or Actors, and access records with methods like
:meth:`~ray.data.Dataset.take_batch` and :meth:`~ray.data.Dataset.iter_batches`.

.. testcode::

    print(transformed_ds.take_batch(batch_size=3))

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    {'sepal length (cm)': array([5.1, 4.9, 4.7]),
        'sepal width (cm)': array([3.5, 3. , 3.2]),
        'petal length (cm)': array([1.4, 1.4, 1.3]),
        'petal width (cm)': array([0.2, 0.2, 0.2]),
        'target': array([0, 0, 0]),
        'petal area (cm^2)': array([0.28, 0.28, 0.26])}

To learn more about consuming datasets, see
:ref:`Iterating over Data <iterating-over-data>` and :ref:`Saving Data <saving-data>`.

.. _saving_key_concept:

Saving data
-----------

Call methods like :meth:`~ray.data.Dataset.write_parquet` to save dataset contents to local
or remote filesystems.

.. testcode::
    :hide:

    # The number of blocks can be non-determinstic. Repartition the dataset beforehand
    # so that the number of written files is consistent.
    transformed_ds = transformed_ds.repartition(2)

.. testcode::

    import os

    transformed_ds.write_parquet("/tmp/iris")

    print(os.listdir("/tmp/iris"))

.. testoutput::
    :options: +MOCK

    ['..._000000.parquet', '..._000001.parquet']


To learn more about saving dataset contents, see :ref:`Saving data <saving-data>`.


.. _monitoring-your-workload:

Monitoring Your Workload
========================

This section helps you debug and monitor the execution of your :class:`~ray.data.Dataset` by viewing the:

* :ref:`Ray Data dashboard <ray-data-dashboard>`
* :ref:`Ray Data logs <ray-data-logs>`
* :ref:`Ray Data stats <ray-data-stats>`


.. _ray-data-dashboard:

Ray Data dashboard
------------------

Ray Data emits Prometheus metrics in real-time while a Dataset is executing. These metrics are tagged by both dataset and operator, and are displayed in multiple views across the Ray dashboard.

.. note::
   Most metrics are only available for physical operators that use the map operation. For example, physical operators created by :meth:`~ray.data.Dataset.map_batches`, :meth:`~ray.data.Dataset.map`, and :meth:`~ray.data.Dataset.flat_map`.

Jobs: Ray Data overview
~~~~~~~~~~~~~~~~~~~~~~~

For an overview of all datasets that have been running on your cluster, see the Ray Data Overview in the :ref:`jobs view <dash-jobs-view>`. This table appears once the first dataset starts executing on the cluster, and shows dataset details such as:

* execution progress (measured in blocks)
* execution state (running, failed, or finished)
* dataset start/end time
* dataset-level metrics (for example, sum of rows processed over all operators)

.. image:: images/data-overview-table.png
   :align: center

For a more fine-grained overview, each dataset row in the table can also be expanded to display the same details for individual operators.

.. image:: images/data-overview-table-expanded.png
   :align: center

.. tip::

    To evaluate a dataset-level metric where it's not appropriate to sum the values of all the individual operators, it may be more useful to look at the operator-level metrics of the last operator. For example, to calculate a dataset's throughput, use the "Rows Outputted" of the dataset's last operator, because the dataset-level metric contains the sum of rows outputted over all operators.

Ray dashboard metrics
~~~~~~~~~~~~~~~~~~~~~

For a time-series view of these metrics, see the Ray Data section in the :ref:`Metrics view <dash-metrics-view>`. This section contains time-series graphs of all metrics emitted by Ray Data. Execution metrics are grouped by dataset and operator, and iteration metrics are grouped by dataset.

The metrics recorded include:

* Bytes spilled by objects from object store to disk
* Bytes of objects allocated in object store
* Bytes of objects freed in object store
* Current total bytes of objects in object store
* Logical CPUs allocated to dataset operators
* Logical GPUs allocated to dataset operators
* Bytes outputted by dataset operators
* Rows outputted by dataset operators
* Input blocks received by data operators
* Input blocks/bytes processed in tasks by data operators
* Input bytes submitted to tasks by data operators
* Output blocks/bytes/rows generated in tasks by data operators
* Output blocks/bytes taken by downstream operators
* Output blocks/bytes from finished tasks
* Submitted tasks
* Running tasks
* Tasks with at least one output block
* Finished tasks
* Failed tasks
* Operator internal inqueue size (in blocks/bytes)
* Operator internal outqueue size (in blocks/bytes)
* Size of blocks used in pending tasks
* Freed memory in object store
* Spilled memory in object store
* Time spent generating blocks
* Time spent in task submission backpressure
* Time spent to initialize iteration.
* Time user code is blocked during iteration.
* Time spent in user code during iteration.

.. image:: images/data-dashboard.png
   :align: center


To learn more about the Ray dashboard, including detailed setup instructions, see :ref:`Ray Dashboard <observability-getting-started>`.

.. _ray-data-logs:

Ray Data logs
-------------
During execution, Ray Data periodically logs updates to `ray-data.log`.

Every five seconds, Ray Data logs the execution progress of every operator in the dataset. For more frequent updates, set `RAY_DATA_TRACE_SCHEDULING=1` so that the progress is logged after each task is dispatched.

.. code-block:: text

   Execution Progress:
   0: - Input: 0 active, 0 queued, 0.0 MiB objects, Blocks Outputted: 200/200
   1: - ReadRange->MapBatches(<lambda>): 10 active, 190 queued, 381.47 MiB objects, Blocks Outputted: 100/200

When an operator completes, the metrics for that operator are also logged.

.. code-block:: text

   Operator InputDataBuffer[Input] -> TaskPoolMapOperator[ReadRange->MapBatches(<lambda>)] completed. Operator Metrics:
   {'num_inputs_received': 20, 'bytes_inputs_received': 46440, 'num_task_inputs_processed': 20, 'bytes_task_inputs_processed': 46440, 'num_task_outputs_generated': 20, 'bytes_task_outputs_generated': 800, 'rows_task_outputs_generated': 100, 'num_outputs_taken': 20, 'bytes_outputs_taken': 800, 'num_outputs_of_finished_tasks': 20, 'bytes_outputs_of_finished_tasks': 800, 'num_tasks_submitted': 20, 'num_tasks_running': 0, 'num_tasks_have_outputs': 20, 'num_tasks_finished': 20, 'obj_store_mem_freed': 46440, 'obj_store_mem_spilled': 0, 'block_generation_time': 1.191296085, 'cpu_usage': 0, 'gpu_usage': 0, 'ray_remote_args': {'num_cpus': 1, 'scheduling_strategy': 'SPREAD'}}

This log file can be found locally at `/tmp/ray/{SESSION_NAME}/logs/ray-data/ray-data.log`. It can also be found on the Ray Dashboard under the head node's logs in the :ref:`Logs view <dash-logs-view>`.

.. _ray-data-stats:

Ray Data stats
--------------
To see detailed stats on the execution of a dataset you can use the :meth:`~ray.data.Dataset.stats` method.

Operator stats
~~~~~~~~~~~~~~
The stats output includes a summary on the individual operator's execution stats for each operator. Ray Data calculates this
summary across many different blocks, so some stats show the min, max, mean, and sum of the stats aggregated over all the blocks.
The following are descriptions of the various stats included at the operator level:

* **Remote wall time**: The wall time is the start to finish time for an operator. It includes the time where the operator
  isn't processing data, sleeping, waiting for I/O, etc.
* **Remote CPU time**: The CPU time is the process time for an operator which excludes time slept. This time includes both
  user and system CPU time.
* **UDF time**: The UDF time is time spent in functions defined by the user. This time includes functions you pass into Ray
  Data methods, including :meth:`~ray.data.Dataset.map`, :meth:`~ray.data.Dataset.map_batches`, :meth:`~ray.data.Dataset.filter`,
  etc. You can use this stat to track the time spent in functions you define and how much time optimizing those functions could save.
* **Memory usage**: The output displays memory usage per block in MiB.
* **Output stats**: The output includes stats on the number of rows output and size of output in bytes per block. The number of
  output rows per task is also included. All of this together gives you insight into how much data Ray Data is outputting at a per
  block and per task level.
* **Task Stats**: The output shows the scheduling of tasks to nodes, which allows you to see if you are utilizing all of your nodes
  as expected.
* **Throughput**: The summary calculates the throughput for the operator, and for a point of comparison, it also computes an estimate of
  the throughput of the same task on a single node. This estimate assumes the total time of the work remains the same, but with no
  concurrency. The overall summary also calculates the throughput at the dataset level, including a single node estimate.

Iterator stats
~~~~~~~~~~~~~~
If you iterate over the data, Ray Data also generates iteration stats. Even if you aren't directly iterating over the data, you
might see iteration stats, for example, if you call :meth:`~ray.data.Dataset.take_all`. Some of the stats that Ray Data includes
at the iterator level are:

* **Iterator initialization**: The time Ray Data spent initializing the iterator. This time is internal to Ray Data.
* **Time user thread is blocked**: The time Ray Data spent producing data in the iterator. This time is often the primary execution of a
  dataset if you haven't previously materialized it.
* **Time in user thread**: The time spent in the user thread that's iterating over the dataset outside of the Ray Data code.
  If this time is high, consider optimizing the body of the loop that's iterating over the dataset.
* **Batch iteration stats**: Ray Data also includes stats about the prefetching of batches. These times are internal to Ray
  Data code, but you can further optimize this time by tuning the prefetching process.

Verbose stats
~~~~~~~~~~~~~~
By default, Ray Data only logs the most important high-level stats. To enable verbose stats outputs, include
the following snippet in your Ray Data code:

.. testcode::

   from ray.data import DataContext

   context = DataContext.get_current()
   context.verbose_stats_logs = True


By enabling verbosity Ray Data adds a few more outputs:

* **Extra metrics**: Operators, executors, etc. can add to this dictionary of various metrics. There is
  some duplication of stats between the default output and this dictionary, but for advanced users this stat provides more
  insight into the dataset's execution.
* **Runtime metrics**: These metrics are a high-level breakdown of the runtime of the dataset execution. These stats are a per
  operator summary of the time each operator took to complete and the fraction of the total execution time that the operator took
  to complete. As there are potentially multiple concurrent operators, these percentages don't necessarily sum to 100%. Instead,
  they show how long running each of the operators is in the context of the full dataset execution.

Example stats
~~~~~~~~~~~~~
As a concrete example, below is a stats output from :doc:`Image Classification Batch Inference with PyTorch ResNet18 </data/examples/pytorch_resnet_batch_prediction>`:

.. code-block:: text

   Operator 1 ReadImage->Map(preprocess_image): 384 tasks executed, 386 blocks produced in 9.21s
   * Remote wall time: 33.55ms min, 2.22s max, 1.03s mean, 395.65s total
   * Remote cpu time: 34.93ms min, 3.36s max, 1.64s mean, 632.26s total
   * UDF time: 535.1ms min, 2.16s max, 975.7ms mean, 376.62s total
   * Peak heap memory usage (MiB): 556.32 min, 1126.95 max, 655 mean
   * Output num rows per block: 4 min, 25 max, 24 mean, 9469 total
   * Output size bytes per block: 6060399 min, 105223020 max, 31525416 mean, 12168810909 total
   * Output rows per task: 24 min, 25 max, 24 mean, 384 tasks used
   * Tasks per node: 32 min, 64 max, 48 mean; 8 nodes used
   * Operator throughput:
         * Ray Data throughput: 1028.5218637702708 rows/s
         * Estimated single node throughput: 23.932674100499128 rows/s

   Operator 2 MapBatches(ResnetModel): 14 tasks executed, 48 blocks produced in 27.43s
   * Remote wall time: 523.93us min, 7.01s max, 1.82s mean, 87.18s total
   * Remote cpu time: 523.23us min, 6.23s max, 1.76s mean, 84.61s total
   * UDF time: 4.49s min, 17.81s max, 10.52s mean, 505.08s total
   * Peak heap memory usage (MiB): 4025.42 min, 7920.44 max, 5803 mean
   * Output num rows per block: 84 min, 334 max, 197 mean, 9469 total
   * Output size bytes per block: 72317976 min, 215806447 max, 134739694 mean, 6467505318 total
   * Output rows per task: 319 min, 720 max, 676 mean, 14 tasks used
   * Tasks per node: 3 min, 4 max, 3 mean; 4 nodes used
   * Operator throughput:
         * Ray Data throughput: 345.1533728632648 rows/s
         * Estimated single node throughput: 108.62003864820711 rows/s

   Dataset iterator time breakdown:
   * Total time overall: 38.53s
      * Total time in Ray Data iterator initialization code: 16.86s
      * Total time user thread is blocked by Ray Data iter_batches: 19.76s
      * Total execution time for user thread: 1.9s
   * Batch iteration time breakdown (summed across prefetch threads):
      * In ray.get(): 70.49ms min, 2.16s max, 272.8ms avg, 13.09s total
      * In batch creation: 3.6us min, 5.95us max, 4.26us avg, 204.41us total
      * In batch formatting: 4.81us min, 7.88us max, 5.5us avg, 263.94us total

   Dataset throughput:
         * Ray Data throughput: 1026.5318925757008 rows/s
         * Estimated single node throughput: 19.611578909587674 rows/s

For the same example with verbosity enabled, the stats output is:

.. code-block:: text

   Operator 1 ReadImage->Map(preprocess_image): 384 tasks executed, 387 blocks produced in 9.49s
   * Remote wall time: 22.81ms min, 2.5s max, 999.95ms mean, 386.98s total
   * Remote cpu time: 24.06ms min, 3.36s max, 1.63s mean, 629.93s total
   * UDF time: 552.79ms min, 2.41s max, 956.84ms mean, 370.3s total
   * Peak heap memory usage (MiB): 550.95 min, 1186.28 max, 651 mean
   * Output num rows per block: 4 min, 25 max, 24 mean, 9469 total
   * Output size bytes per block: 4444092 min, 105223020 max, 31443955 mean, 12168810909 total
   * Output rows per task: 24 min, 25 max, 24 mean, 384 tasks used
   * Tasks per node: 39 min, 60 max, 48 mean; 8 nodes used
   * Operator throughput:
         * Ray Data throughput: 997.9207015895857 rows/s
         * Estimated single node throughput: 24.46899945870273 rows/s
   * Extra metrics: {'num_inputs_received': 384, 'bytes_inputs_received': 1104723940, 'num_task_inputs_processed': 384, 'bytes_task_inputs_processed': 1104723940, 'bytes_inputs_of_submitted_tasks': 1104723940, 'num_task_outputs_generated': 387, 'bytes_task_outputs_generated': 12168810909, 'rows_task_outputs_generated': 9469, 'num_outputs_taken': 387, 'bytes_outputs_taken': 12168810909, 'num_outputs_of_finished_tasks': 387, 'bytes_outputs_of_finished_tasks': 12168810909, 'num_tasks_submitted': 384, 'num_tasks_running': 0, 'num_tasks_have_outputs': 384, 'num_tasks_finished': 384, 'num_tasks_failed': 0, 'block_generation_time': 386.97945193799995, 'task_submission_backpressure_time': 7.263684450000142, 'obj_store_mem_internal_inqueue_blocks': 0, 'obj_store_mem_internal_inqueue': 0, 'obj_store_mem_internal_outqueue_blocks': 0, 'obj_store_mem_internal_outqueue': 0, 'obj_store_mem_pending_task_inputs': 0, 'obj_store_mem_freed': 1104723940, 'obj_store_mem_spilled': 0, 'obj_store_mem_used': 12582535566, 'cpu_usage': 0, 'gpu_usage': 0, 'ray_remote_args': {'num_cpus': 1, 'scheduling_strategy': 'SPREAD'}}

   Operator 2 MapBatches(ResnetModel): 14 tasks executed, 48 blocks produced in 28.81s
   * Remote wall time: 134.84us min, 7.23s max, 1.82s mean, 87.16s total
   * Remote cpu time: 133.78us min, 6.28s max, 1.75s mean, 83.98s total
   * UDF time: 4.56s min, 17.78s max, 10.28s mean, 493.48s total
   * Peak heap memory usage (MiB): 3925.88 min, 7713.01 max, 5688 mean
   * Output num rows per block: 125 min, 259 max, 197 mean, 9469 total
   * Output size bytes per block: 75531617 min, 187889580 max, 134739694 mean, 6467505318 total
   * Output rows per task: 325 min, 719 max, 676 mean, 14 tasks used
   * Tasks per node: 3 min, 4 max, 3 mean; 4 nodes used
   * Operator throughput:
         * Ray Data throughput: 328.71474145609153 rows/s
         * Estimated single node throughput: 108.6352856660782 rows/s
   * Extra metrics: {'num_inputs_received': 387, 'bytes_inputs_received': 12168810909, 'num_task_inputs_processed': 0, 'bytes_task_inputs_processed': 0, 'bytes_inputs_of_submitted_tasks': 12168810909, 'num_task_outputs_generated': 1, 'bytes_task_outputs_generated': 135681874, 'rows_task_outputs_generated': 252, 'num_outputs_taken': 1, 'bytes_outputs_taken': 135681874, 'num_outputs_of_finished_tasks': 0, 'bytes_outputs_of_finished_tasks': 0, 'num_tasks_submitted': 14, 'num_tasks_running': 14, 'num_tasks_have_outputs': 1, 'num_tasks_finished': 0, 'num_tasks_failed': 0, 'block_generation_time': 7.229860895999991, 'task_submission_backpressure_time': 0, 'obj_store_mem_internal_inqueue_blocks': 13, 'obj_store_mem_internal_inqueue': 413724657, 'obj_store_mem_internal_outqueue_blocks': 0, 'obj_store_mem_internal_outqueue': 0, 'obj_store_mem_pending_task_inputs': 12168810909, 'obj_store_mem_freed': 0, 'obj_store_mem_spilled': 0, 'obj_store_mem_used': 1221136866.0, 'cpu_usage': 0, 'gpu_usage': 4}

   Dataset iterator time breakdown:
   * Total time overall: 42.29s
      * Total time in Ray Data iterator initialization code: 20.24s
      * Total time user thread is blocked by Ray Data iter_batches: 19.96s
      * Total execution time for user thread: 2.08s
   * Batch iteration time breakdown (summed across prefetch threads):
      * In ray.get(): 73.0ms min, 2.15s max, 246.3ms avg, 11.82s total
      * In batch creation: 3.62us min, 6.6us max, 4.39us avg, 210.7us total
      * In batch formatting: 4.75us min, 8.67us max, 5.52us avg, 264.98us total

   Dataset throughput:
         * Ray Data throughput: 468.11051989434594 rows/s
         * Estimated single node throughput: 972.8197093015862 rows/s

   Runtime Metrics:
   * ReadImage->Map(preprocess_image): 9.49s (46.909%)
   * MapBatches(ResnetModel): 28.81s (142.406%)
   * Scheduling: 6.16s (30.448%)
   * Total: 20.23s (100.000%)

Working with Text
=================

With Ray Data, you can easily read and transform large amounts of text data.

This guide shows you how to:

* :ref:`Read text files <reading-text-files>`
* :ref:`Transform text data <transforming-text>`
* :ref:`Perform inference on text data <performing-inference-on-text>`
* :ref:`Save text data <saving-text>`

.. _reading-text-files:

Reading text files
------------------

Ray Data can read lines of text and JSONL. Alternatively, you can read raw binary
files and manually decode data.

.. tab-set::

    .. tab-item:: Text lines

        To read lines of text, call :func:`~ray.data.read_text`. Ray Data creates a
        row for each line of text.

        .. testcode::

            import ray

            ds = ray.data.read_text("s3://anonymous@ray-example-data/this.txt")

            ds.show(3)

        .. testoutput::

            {'text': 'The Zen of Python, by Tim Peters'}
            {'text': 'Beautiful is better than ugly.'}
            {'text': 'Explicit is better than implicit.'}

    .. tab-item:: JSON Lines

        `JSON Lines <https://jsonlines.org/>`_ is a text format for structured data.
        It's typically used to process data one record at a time.

        To read JSON Lines files, call :func:`~ray.data.read_json`. Ray Data creates a
        row for each JSON object.

        .. testcode::

            import ray

            ds = ray.data.read_json("s3://anonymous@ray-example-data/logs.json")

            ds.show(3)

        .. testoutput::

            {'timestamp': datetime.datetime(2022, 2, 8, 15, 43, 41), 'size': 48261360}
            {'timestamp': datetime.datetime(2011, 12, 29, 0, 19, 10), 'size': 519523}
            {'timestamp': datetime.datetime(2028, 9, 9, 5, 6, 7), 'size': 2163626}


    .. tab-item:: Other formats

        To read other text formats, call :func:`~ray.data.read_binary_files`. Then,
        call :meth:`~ray.data.Dataset.map` to decode your data.

        .. testcode::

            from typing import Any, Dict
            from bs4 import BeautifulSoup
            import ray

            def parse_html(row: Dict[str, Any]) -> Dict[str, Any]:
                html = row["bytes"].decode("utf-8")
                soup = BeautifulSoup(html, features="html.parser")
                return {"text": soup.get_text().strip()}

            ds = (
                ray.data.read_binary_files("s3://anonymous@ray-example-data/index.html")
                .map(parse_html)
            )

            ds.show()

        .. testoutput::

            {'text': 'Batoidea\nBatoidea is a superorder of cartilaginous fishes...'}

For more information on reading files, see :ref:`Loading data <loading_data>`.

.. _transforming-text:

Transforming text
-----------------

To transform text, implement your transformation in a function or callable class. Then,
call :meth:`Dataset.map() <ray.data.Dataset.map>` or
:meth:`Dataset.map_batches() <ray.data.Dataset.map_batches>`. Ray Data transforms your
text in parallel.

.. testcode::

    from typing import Any, Dict
    import ray

    def to_lower(row: Dict[str, Any]) -> Dict[str, Any]:
        row["text"] = row["text"].lower()
        return row

    ds = (
        ray.data.read_text("s3://anonymous@ray-example-data/this.txt")
        .map(to_lower)
    )

    ds.show(3)

.. testoutput::

    {'text': 'the zen of python, by tim peters'}
    {'text': 'beautiful is better than ugly.'}
    {'text': 'explicit is better than implicit.'}

For more information on transforming data, see
:ref:`Transforming data <transforming_data>`.

.. _performing-inference-on-text:

Performing inference on text
----------------------------

To perform inference with a pre-trained model on text data, implement a callable class
that sets up and invokes a model. Then, call
:meth:`Dataset.map_batches() <ray.data.Dataset.map_batches>`.

.. testcode::

    from typing import Dict

    import numpy as np
    from transformers import pipeline

    import ray

    class TextClassifier:
        def __init__(self):

            self.model = pipeline("text-classification")

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            predictions = self.model(list(batch["text"]))
            batch["label"] = [prediction["label"] for prediction in predictions]
            return batch

    ds = (
        ray.data.read_text("s3://anonymous@ray-example-data/this.txt")
        .map_batches(TextClassifier, concurrency=2)
    )

    ds.show(3)

.. testoutput::

    {'text': 'The Zen of Python, by Tim Peters', 'label': 'POSITIVE'}
    {'text': 'Beautiful is better than ugly.', 'label': 'POSITIVE'}
    {'text': 'Explicit is better than implicit.', 'label': 'POSITIVE'}

For more information on performing inference, see
:ref:`End-to-end: Offline Batch Inference <batch_inference_home>`
and :ref:`Stateful Transforms <stateful_transforms>`.

.. _saving-text:

Saving text
-----------

To save text, call a method like :meth:`~ray.data.Dataset.write_parquet`. Ray Data can
save text in many formats.

To view the full list of supported file formats, see the
:ref:`Input/Output reference <input-output>`.

.. testcode::

    import ray

    ds = ray.data.read_text("s3://anonymous@ray-example-data/this.txt")

    ds.write_parquet("local:///tmp/results")

For more information on saving data, see :ref:`Saving data <saving-data>`.


.. _data:

==================================
Ray Data: Scalable Datasets for ML
==================================

.. toctree::
    :hidden:

    Overview <overview>
    quickstart
    user-guide
    examples
    api/api
    data-internals

Ray Data is a scalable data processing library for ML workloads. It provides flexible and performant APIs for scaling :ref:`Offline batch inference <batch_inference_overview>` and :ref:`Data preprocessing and ingest for ML training <ml_ingest_overview>`. Ray Data uses `streaming execution <https://www.anyscale.com/blog/streaming-distributed-execution-across-cpus-and-gpus>`__ to efficiently process large datasets.

..
  https://docs.google.com/drawings/d/16AwJeBNR46_TsrkOmMbGaBK7u-OPsf_V8fHjU-d2PPQ/edit

Install Ray Data
----------------

To install Ray Data, run:

.. code-block:: console

    $ pip install -U 'ray[data]'

To learn more about installing Ray and its libraries, see
:ref:`Installing Ray <installation>`.

Learn more
----------

.. grid:: 1 2 2 2
    :gutter: 1
    :class-container: container pb-6

    .. grid-item-card::

        **Ray Data Overview**
        ^^^

        Get an overview of Ray Data, the workloads that it supports, and how it compares to alternatives.

        +++
        .. button-ref:: data_overview
            :color: primary
            :outline:
            :expand:

            Ray Data Overview

    .. grid-item-card::

        **Quickstart**
        ^^^

        Understand the key concepts behind Ray Data. Learn what
        Datasets are and how they're used.

        +++
        .. button-ref:: data_quickstart
            :color: primary
            :outline:
            :expand:

            Quickstart

    .. grid-item-card::

        **User Guides**
        ^^^

        Learn how to use Ray Data, from basic usage to end-to-end guides.

        +++
        .. button-ref:: data_user_guide
            :color: primary
            :outline:
            :expand:

            Learn how to use Ray Data

    .. grid-item-card::

        **Examples**
        ^^^

        Find both simple and scaling-out examples of using Ray Data.

        +++
        .. button-ref:: examples
            :color: primary
            :outline:
            :expand:

            Ray Data Examples

    .. grid-item-card::

        **API**
        ^^^

        Get more in-depth information about the Ray Data API.

        +++
        .. button-ref:: data-api
            :color: primary
            :outline:
            :expand:

            Read the API Reference

    .. grid-item-card::

        **Ray Blogs**
        ^^^

        Get the latest on engineering updates from the Ray team and how companies are using Ray Data.

        +++
        .. button-link:: https://www.anyscale.com/blog?tag=ray-datasets
            :color: primary
            :outline:
            :expand:

            Read the Ray blogs


.. _shuffling_data:

==============
Shuffling Data
==============

When consuming or iterating over Ray :class:`Datasets <ray.data.dataset.Dataset>`, it can be useful to
shuffle or randomize the order of data (for example, randomizing data ingest order during ML training).
This guide shows several different methods of shuffling data with Ray Data and their respective trade-offs.

Types of shuffling
==================

Ray Data provides several different options for shuffling data, trading off the granularity of shuffle
control with memory consumption and runtime. The options below are listed in increasing order of
resource consumption and runtime; choose the most appropriate method for your use case.

.. _shuffling_file_order:

Shuffle the ordering of files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To randomly shuffle the ordering of input files before reading, call a :ref:`read function <input-output>` function that supports shuffling, such as
:func:`~ray.data.read_images`, and use the ``shuffle="files"`` parameter. This randomly assigns
input files to workers for reading.

This is the fastest option for shuffle, and is a purely metadata operation. This
option doesn't shuffle the actual rows inside files, so the randomness might be
poor if each file has many rows.

.. testcode::

    import ray

    ds = ray.data.read_images(
        "s3://anonymous@ray-example-data/image-datasets/simple",
        shuffle="files",
    )

.. _local_shuffle_buffer:

Local shuffle when iterating over batches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To locally shuffle a subset of rows using iteration methods, such as :meth:`~ray.data.Dataset.iter_batches`,
:meth:`~ray.data.Dataset.iter_torch_batches`, and :meth:`~ray.data.Dataset.iter_tf_batches`,
specify `local_shuffle_buffer_size`. This shuffles the rows up to a provided buffer
size during iteration. See more details in
:ref:`Iterating over batches with shuffling <iterating-over-batches-with-shuffling>`.

This is slower than shuffling ordering of files, and shuffles rows locally without
network transfer. This local shuffle buffer can be used together with shuffling
ordering of files; see :ref:`Shuffle the ordering of files <shuffling_file_order>`.

.. testcode::

    import ray

    ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

    for batch in ds.iter_batches(
        batch_size=2,
        batch_format="numpy",
        local_shuffle_buffer_size=250,
    ):
        print(batch)

.. tip::

    If you observe reduced throughput when using ``local_shuffle_buffer_size``,
    check the total time spent in batch creation by
    examining the ``ds.stats()`` output (``In batch formatting``, under
    ``Batch iteration time breakdown``). If this time is significantly larger than the
    time spent in other steps, decrease ``local_shuffle_buffer_size`` or turn off the local
    shuffle buffer altogether and only :ref:`shuffle the ordering of files <shuffling_file_order>`.

Shuffling block order
~~~~~~~~~~~~~~~~~~~~~

This option randomizes the order of blocks in a dataset. Blocks are the basic unit of data chunk that Ray Data stores in the object store. Applying this operation alone doesn't involve heavy computation and communication. However, it requires Ray Data to materialize all blocks in memory before applying the operation. Only use this option when your dataset is small enough to fit into the object store memory.

To perform block order shuffling, use :meth:`randomize_block_order <ray.data.Dataset.randomize_block_order>`.

.. testcode::
    import ray

    ds = ray.data.read_text(
        "s3://anonymous@ray-example-data/sms_spam_collection_subset.txt"
    )

    # Randomize the block order of this dataset.
    ds = ds.randomize_block_order()

Shuffle all rows
~~~~~~~~~~~~~~~~

To randomly shuffle all rows globally, call :meth:`~ray.data.Dataset.random_shuffle`.
This is the slowest option for shuffle, and requires transferring data across
network between workers. This option achieves the best randomness among all options.

.. testcode::

    import ray

    ds = (
        ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
        .random_shuffle()
    )

.. _optimizing_shuffles:

Advanced: Optimizing shuffles
=============================
.. note:: This is an active area of development. If your Dataset uses a shuffle operation and you are having trouble configuring shuffle,
    `file a Ray Data issue on GitHub <https://github.com/ray-project/ray/issues/new?assignees=&labels=bug%2Ctriage%2Cdata&projects=&template=bug-report.yml&title=[data]+>`_.

When should you use global per-epoch shuffling?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use global per-epoch shuffling only if your model is sensitive to the
randomness of the training data. Based on a
`theoretical foundation <https://arxiv.org/abs/1709.10432>`__, all
gradient-descent-based model trainers benefit from improved (global) shuffle quality.
In practice, the benefit is particularly pronounced for tabular data/models.
However, the more global the shuffle is, the more expensive the shuffling operation.
The increase compounds with distributed data-parallel training on a multi-node cluster due
to data transfer costs. This cost can be prohibitive when using very large datasets.

The best route for determining the best tradeoff between preprocessing time and cost and
per-epoch shuffle quality is to measure the precision gain per training step for your
particular model under different shuffling policies:

* no shuffling,
* local (per-shard) limited-memory shuffle buffer,
* local (per-shard) shuffling,
* windowed (pseudo-global) shuffling, and
* fully global shuffling.

As long as your data loading and shuffling throughput is higher than your training throughput, your GPU should
be saturated. If you have shuffle-sensitive models, push the
shuffle quality higher until this threshold is hit.

.. _shuffle_performance_tips:

Enabling push-based shuffle
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some Dataset operations require a *shuffle* operation, meaning that data is shuffled from all of the input partitions to all of the output partitions.
These operations include :meth:`Dataset.random_shuffle <ray.data.Dataset.random_shuffle>`,
:meth:`Dataset.sort <ray.data.Dataset.sort>` and :meth:`Dataset.groupby <ray.data.Dataset.groupby>`.
For example, during a sort operation, data is reordered between blocks and therefore requires shuffling across partitions.
Shuffling can be challenging to scale to large data sizes and clusters, especially when the total dataset size can't fit into memory.

Ray Data provides an alternative shuffle implementation known as push-based shuffle for improving large-scale performance.
Try this out if your dataset has more than 1000 blocks or is larger than 1 TB in size.

To try this out locally or on a cluster, you can start with the `nightly release test <https://github.com/ray-project/ray/blob/master/release/nightly_tests/dataset/sort.py>`_ that Ray runs for :meth:`Dataset.random_shuffle <ray.data.Dataset.random_shuffle>` and :meth:`Dataset.sort <ray.data.Dataset.sort>`.
To get an idea of the performance you can expect, here are some run time results for :meth:`Dataset.random_shuffle <ray.data.Dataset.random_shuffle>` on 1-10 TB of data on 20 machines (m5.4xlarge instances on AWS EC2, each with 16 vCPUs, 64 GB RAM).

.. image:: https://docs.google.com/spreadsheets/d/e/2PACX-1vQvBWpdxHsW0-loasJsBpdarAixb7rjoo-lTgikghfCeKPQtjQDDo2fY51Yc1B6k_S4bnYEoChmFrH2/pubchart?oid=598567373&format=image
   :align: center

To try out push-based shuffle, set the environment variable ``RAY_DATA_PUSH_BASED_SHUFFLE=1`` when running your application:

.. code-block:: bash

    $ wget https://raw.githubusercontent.com/ray-project/ray/master/release/nightly_tests/dataset/sort.py
    $ RAY_DATA_PUSH_BASED_SHUFFLE=1 python sort.py --num-partitions=10 --partition-size=1e7

    # Dataset size: 10 partitions, 0.01GB partition size, 0.1GB total
    # [dataset]: Run `pip install tqdm` to enable progress reporting.
    # 2022-05-04 17:30:28,806	INFO push_based_shuffle.py:118 -- Using experimental push-based shuffle.
    # Finished in 9.571171760559082
    # ...

You can also specify the shuffle implementation during program execution by
setting the ``DataContext.use_push_based_shuffle`` flag:

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import ray

    ctx = ray.data.DataContext.get_current()
    ctx.use_push_based_shuffle = True

    ds = (
        ray.data.range(1000)
        .random_shuffle()
    )

Large-scale shuffles can take a while to finish.
For debugging purposes, shuffle operations support executing only part of the shuffle, so that you can collect an execution profile more quickly.
Here is an example that shows how to limit a random shuffle operation to two output blocks:

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import ray

    ctx = ray.data.DataContext.get_current()
    ctx.set_config(
        "debug_limit_shuffle_execution_to_num_blocks", 2
    )

    ds = (
        ray.data.range(1000, override_num_blocks=10)
        .random_shuffle()
        .materialize()
    )
    print(ds.stats())

.. testoutput::
    :options: +MOCK

    Operator 1 ReadRange->RandomShuffle: executed in 0.08s

        Suboperator 0 ReadRange->RandomShuffleMap: 2/2 blocks executed
        ...


.. _data_performance_tips:

Advanced: Performance Tips and Tuning
=====================================

Optimizing transforms
---------------------

Batching transforms
~~~~~~~~~~~~~~~~~~~

If your transformation is vectorized like most NumPy or pandas operations, use
:meth:`~ray.data.Dataset.map_batches` rather than :meth:`~ray.data.Dataset.map`. It's
faster.

If your transformation isn't vectorized, there's no performance benefit.

Optimizing reads
----------------

.. _read_output_blocks:

Tuning output blocks for read
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, Ray Data automatically selects the number of output blocks for read according to the following procedure:

- The ``override_num_blocks`` parameter passed to Ray Data's :ref:`read APIs <input-output>` specifies the number of output blocks, which is equivalent to the number of read tasks to create.
- Usually, if the read is followed by a :func:`~ray.data.Dataset.map` or :func:`~ray.data.Dataset.map_batches`, the map is fused with the read; therefore ``override_num_blocks`` also determines the number of map tasks.

Ray Data decides the default value for number of output blocks based on the following heuristics, applied in order:

1. Start with the default value of 200. You can overwrite this by setting :class:`DataContext.read_op_min_num_blocks <ray.data.context.DataContext>`.
2. Min block size (default=1 MiB). If number of blocks would make blocks smaller than this threshold, reduce number of blocks to avoid the overhead of tiny blocks. You can override by setting :class:`DataContext.target_min_block_size <ray.data.context.DataContext>` (bytes).
3. Max block size (default=128 MiB). If number of blocks would make blocks larger than this threshold, increase number of blocks to avoid out-of-memory errors during processing. You can override by setting :class:`DataContext.target_max_block_size <ray.data.context.DataContext>` (bytes).
4. Available CPUs. Increase number of blocks to utilize all of the available CPUs in the cluster. Ray Data chooses the number of read tasks to be at least 2x the number of available CPUs.

Occasionally, it's advantageous to manually tune the number of blocks to optimize the application.
For example, the following code batches multiple files into the same read task to avoid creating blocks that are too large.

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import ray
    # Pretend there are two CPUs.
    ray.init(num_cpus=2)

    # Repeat the iris.csv file 16 times.
    ds = ray.data.read_csv(["example://iris.csv"] * 16)
    print(ds.materialize())

.. testoutput::
    :options: +MOCK

    MaterializedDataset(
       num_blocks=4,
       num_rows=2400,
       ...
    )

But suppose that you knew that you wanted to read all 16 files in parallel.
This could be, for example, because you know that additional CPUs should get added to the cluster by the autoscaler or because you want the downstream operator to transform each file's contents in parallel.
You can get this behavior by setting the ``override_num_blocks`` parameter.
Notice how the number of output blocks is equal to ``override_num_blocks`` in the following code:

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import ray
    # Pretend there are two CPUs.
    ray.init(num_cpus=2)

    # Repeat the iris.csv file 16 times.
    ds = ray.data.read_csv(["example://iris.csv"] * 16, override_num_blocks=16)
    print(ds.materialize())

.. testoutput::
    :options: +MOCK

    MaterializedDataset(
       num_blocks=16,
       num_rows=2400,
       ...
    )


When using the default auto-detected number of blocks, Ray Data attempts to cap each task's output to :class:`DataContext.target_max_block_size <ray.data.context.DataContext>` many bytes.
Note however that Ray Data can't perfectly predict the size of each task's output, so it's possible that each task produces one or more output blocks.
Thus, the total blocks in the final :class:`~ray.data.Dataset` may differ from the specified ``override_num_blocks``.
Here's an example where we manually specify ``override_num_blocks=1``, but the one task still produces multiple blocks in the materialized Dataset:

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import ray
    # Pretend there are two CPUs.
    ray.init(num_cpus=2)

    # Generate ~400MB of data.
    ds = ray.data.range_tensor(5_000, shape=(10_000, ), override_num_blocks=1)
    print(ds.materialize())

.. testoutput::
    :options: +MOCK

    MaterializedDataset(
       num_blocks=3,
       num_rows=5000,
       schema={data: numpy.ndarray(shape=(10000,), dtype=int64)}
    )


Currently, Ray Data can assign at most one read task per input file.
Thus, if the number of input files is smaller than ``override_num_blocks``, the number of read tasks is capped to the number of input files.
To ensure that downstream transforms can still execute with the desired number of blocks, Ray Data splits the read tasks' outputs into a total of ``override_num_blocks`` blocks and prevents fusion with the downstream transform.
In other words, each read task's output blocks are materialized to Ray's object store before the consuming map task executes.
For example, the following code executes :func:`~ray.data.read_csv` with only one task, but its output is split into 4 blocks before executing the :func:`~ray.data.Dataset.map`:

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import ray
    # Pretend there are two CPUs.
    ray.init(num_cpus=2)

    ds = ray.data.read_csv("example://iris.csv").map(lambda row: row)
    print(ds.materialize().stats())

.. testoutput::
    :options: +MOCK

    ...
    Operator 1 ReadCSV->SplitBlocks(4): 1 tasks executed, 4 blocks produced in 0.01s
    ...
    
    Operator 2 Map(<lambda>): 4 tasks executed, 4 blocks produced in 0.3s
    ...

To turn off this behavior and allow the read and map operators to be fused, set ``override_num_blocks`` manually.
For example, this code sets the number of files equal to ``override_num_blocks``:

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import ray
    # Pretend there are two CPUs.
    ray.init(num_cpus=2)

    ds = ray.data.read_csv("example://iris.csv", override_num_blocks=1).map(lambda row: row)
    print(ds.materialize().stats())

.. testoutput::
    :options: +MOCK

    ...
    Operator 1 ReadCSV->Map(<lambda>): 1 tasks executed, 1 blocks produced in 0.01s
    ...


.. _tuning_read_resources:

Tuning read resources
~~~~~~~~~~~~~~~~~~~~~

By default, Ray requests 1 CPU per read task, which means one read task per CPU can execute concurrently.
For datasources that benefit from more IO parallelism, you can specify a lower ``num_cpus`` value for the read function with the ``ray_remote_args`` parameter.
For example, use ``ray.data.read_parquet(path, ray_remote_args={"num_cpus": 0.25})`` to allow up to four read tasks per CPU.

Parquet column pruning
~~~~~~~~~~~~~~~~~~~~~~

Current Dataset reads all Parquet columns into memory.
If you only need a subset of the columns, make sure to specify the list of columns
explicitly when calling :func:`ray.data.read_parquet` to
avoid loading unnecessary data (projection pushdown).
For example, use ``ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet", columns=["sepal.length", "variety"])`` to read
just two of the five columns of Iris dataset.

.. _parquet_row_pruning:

Parquet row pruning
~~~~~~~~~~~~~~~~~~~

Similarly, you can pass in a filter to :func:`ray.data.read_parquet` (filter pushdown)
which is applied at the file scan so only rows that match the filter predicate
are returned.
For example, use ``ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet", filter=pyarrow.dataset.field("sepal.length") > 5.0)``
(where ``pyarrow`` has to be imported)
to read rows with sepal.length greater than 5.0.
This can be used in conjunction with column pruning when appropriate to get the benefits of both.


.. _data_memory:

Reducing memory usage
---------------------

.. _data_out_of_memory:

Troubleshooting out-of-memory errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During execution, a task can read multiple input blocks, and write multiple output blocks. Input and output blocks consume both worker heap memory and shared memory through Ray's object store.
Ray caps object store memory usage by spilling to disk, but excessive worker heap memory usage can cause out-of-memory situations.

Ray Data attempts to bound its heap memory usage to ``num_execution_slots * max_block_size``. The number of execution slots is by default equal to the number of CPUs, unless custom resources are specified.
The maximum block size is set by the configuration parameter :class:`DataContext.target_max_block_size <ray.data.context.DataContext>` and is set to 128MiB by default.
If the Dataset includes an :ref:`all-to-all shuffle operation <optimizing_shuffles>` (such as :func:`~ray.data.Dataset.random_shuffle`), then the default maximum block size is controlled by :class:`DataContext.target_shuffle_max_block_size <ray.data.context.DataContext>`, set to 1GiB by default to avoid creating too many tiny blocks.

.. note::
    It's **not** recommended to modify :class:`DataContext.target_max_block_size <ray.data.context.DataContext>`. The default is already chosen to balance between high overheads from too many tiny blocks vs. excessive heap memory usage from too-large blocks.

When a task's output is larger than the maximum block size, the worker automatically splits the output into multiple smaller blocks to avoid running out of heap memory.
However, too-large blocks are still possible, and they can lead to out-of-memory situations.
To avoid these issues:

1. Make sure no single item in your dataset is too large. Aim for rows that are <10 MB each.
2. Always call :meth:`ds.map_batches() <ray.data.Dataset.map_batches>` with a batch size small enough such that the output batch can comfortably fit into heap memory. Or, if vectorized execution is not necessary, use :meth:`ds.map() <ray.data.Dataset.map>`.
3. If neither of these is sufficient, manually increase the :ref:`read output blocks <read_output_blocks>` or modify your application code to ensure that each task reads a smaller amount of data.

As an example of tuning batch size, the following code uses one task to load a 1 GB :class:`~ray.data.Dataset` with 1000 1 MB rows and applies an identity function using :func:`~ray.data.Dataset.map_batches`.
Because the default ``batch_size`` for :func:`~ray.data.Dataset.map_batches` is 1024 rows, this code produces only one very large batch, causing the heap memory usage to increase to 4 GB.

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import ray
    # Pretend there are two CPUs.
    ray.init(num_cpus=2)

    # Force Ray Data to use one task to show the memory issue.
    ds = ray.data.range_tensor(1000, shape=(125_000, ), override_num_blocks=1)
    # The default batch size is 1024 rows.
    ds = ds.map_batches(lambda batch: batch)
    print(ds.materialize().stats())

.. testoutput::
    :options: +MOCK

    Operator 1 ReadRange->MapBatches(<lambda>): 1 tasks executed, 7 blocks produced in 1.33s
      ...
    * Peak heap memory usage (MiB): 3302.17 min, 4233.51 max, 4100 mean
    * Output num rows: 125 min, 125 max, 125 mean, 1000 total
    * Output size bytes: 134000536 min, 196000784 max, 142857714 mean, 1000004000 total
      ...

Setting a lower batch size produces lower peak heap memory usage:

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import ray
    # Pretend there are two CPUs.
    ray.init(num_cpus=2)

    ds = ray.data.range_tensor(1000, shape=(125_000, ), override_num_blocks=1)
    ds = ds.map_batches(lambda batch: batch, batch_size=32)
    print(ds.materialize().stats())

.. testoutput::
    :options: +MOCK

    Operator 1 ReadRange->MapBatches(<lambda>): 1 tasks executed, 7 blocks produced in 0.51s
    ...
    * Peak heap memory usage (MiB): 587.09 min, 1569.57 max, 1207 mean
    * Output num rows: 40 min, 160 max, 142 mean, 1000 total
    * Output size bytes: 40000160 min, 160000640 max, 142857714 mean, 1000004000 total
    ...

Improving heap memory usage in Ray Data is an active area of development.
Here are the current known cases in which heap memory usage may be very high:

1. Reading large (1 GiB or more) binary files.
2. Transforming a Dataset where individual rows are large (100 MiB or more).

In these cases, the last resort is to reduce the number of concurrent execution slots.
This can be done with custom resources.
For example, use :meth:`ds.map_batches(fn, num_cpus=2) <ray.data.Dataset.map_batches>` to halve the number of execution slots for the ``map_batches`` tasks.

If these strategies are still insufficient, `file a Ray Data issue on GitHub`_.


Avoiding object spilling
~~~~~~~~~~~~~~~~~~~~~~~~

A Dataset's intermediate and output blocks are stored in Ray's object store.
Although Ray Data attempts to minimize object store usage with :ref:`streaming execution <streaming_execution>`, it's still possible that the working set exceeds the object store capacity.
In this case, Ray begins spilling blocks to disk, which can slow down execution significantly or even cause out-of-disk errors.

There are some cases where spilling is expected. In particular, if the total Dataset's size is larger than object store capacity, and one of the following is true:

1. An :ref:`all-to-all shuffle operation <optimizing_shuffles>` is used. Or,
2. There is a call to :meth:`ds.materialize() <ray.data.Dataset.materialize>`.

Otherwise, it's best to tune your application to avoid spilling.
The recommended strategy is to manually increase the :ref:`read output blocks <read_output_blocks>` or modify your application code to ensure that each task reads a smaller amount of data.

.. note:: This is an active area of development. If your Dataset is causing spilling and you don't know why, `file a Ray Data issue on GitHub`_.

Handling too-small blocks
~~~~~~~~~~~~~~~~~~~~~~~~~

When different operators of your Dataset produce different-sized outputs, you may end up with very small blocks, which can hurt performance and even cause crashes from excessive metadata.
Use :meth:`ds.stats() <ray.data.Dataset.stats>` to check that each operator's output blocks are each at least 1 MB and ideally 100 MB.

If your blocks are smaller than this, consider repartitioning into larger blocks.
There are two ways to do this:

1. If you need control over the exact number of output blocks, use :meth:`ds.repartition(num_partitions) <ray.data.Dataset.repartition>`. Note that this is an :ref:`all-to-all operation <optimizing_shuffles>` and it materializes all blocks into memory before performing the repartition.
2. If you don't need control over the exact number of output blocks and just want to produce larger blocks, use :meth:`ds.map_batches(lambda batch: batch, batch_size=batch_size) <ray.data.Dataset.map_batches>` and set ``batch_size`` to the desired number of rows per block. This is executed in a streaming fashion and avoids materialization.

When :meth:`ds.map_batches() <ray.data.Dataset.map_batches>` is used, Ray Data coalesces blocks so that each map task can process at least this many rows.
Note that the chosen ``batch_size`` is a lower bound on the task's input block size but it does not necessarily determine the task's final *output* block size; see :ref:`the section <data_out_of_memory>` on block memory usage for more information on how block size is determined.

To illustrate these, the following code uses both strategies to coalesce the 10 tiny blocks with 1 row each into 1 larger block with 10 rows:

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import ray
    # Pretend there are two CPUs.
    ray.init(num_cpus=2)

    # 1. Use ds.repartition().
    ds = ray.data.range(10, override_num_blocks=10).repartition(1)
    print(ds.materialize().stats())

    # 2. Use ds.map_batches().
    ds = ray.data.range(10, override_num_blocks=10).map_batches(lambda batch: batch, batch_size=10)
    print(ds.materialize().stats())

.. testoutput::
    :options: +MOCK

    # 1. ds.repartition() output.
    Operator 1 ReadRange: 10 tasks executed, 10 blocks produced in 0.33s
    ...
    * Output num rows: 1 min, 1 max, 1 mean, 10 total
    ...
    Operator 2 Repartition: executed in 0.36s

            Suboperator 0 RepartitionSplit: 10 tasks executed, 10 blocks produced
            ...

            Suboperator 1 RepartitionReduce: 1 tasks executed, 1 blocks produced
            ...
            * Output num rows: 10 min, 10 max, 10 mean, 10 total
            ...


    # 2. ds.map_batches() output.
    Operator 1 ReadRange->MapBatches(<lambda>): 1 tasks executed, 1 blocks produced in 0s
    ...
    * Output num rows: 10 min, 10 max, 10 mean, 10 total

Configuring execution
---------------------

Configuring resources and locality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the CPU and GPU limits are set to the cluster size, and the object store memory limit conservatively to 1/4 of the total object store size to avoid the possibility of disk spilling.

You may want to customize these limits in the following scenarios:
- If running multiple concurrent jobs on the cluster, setting lower limits can avoid resource contention between the jobs.
- If you want to fine-tune the memory limit to maximize performance.
- For data loading into training jobs, you may want to set the object store memory to a low value (for example, 2 GB) to limit resource usage.

You can configure execution options with the global DataContext. The options are applied for future jobs launched in the process:

.. code-block::

   ctx = ray.data.DataContext.get_current()
   ctx.execution_options.resource_limits.cpu = 10
   ctx.execution_options.resource_limits.gpu = 5
   ctx.execution_options.resource_limits.object_store_memory = 10e9

.. note::
    It's **not** recommended to modify the Ray Core object store memory limit, as this can reduce available memory for task execution. The one exception to this is if you are using machines with a very large amount of RAM (1 TB or more each); then it's recommended to set the object store to ~30-40%.

Locality with output (ML ingest use case)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   ctx.execution_options.locality_with_output = True

Setting this parameter to True tells Ray Data to prefer placing operator tasks onto the consumer node in the cluster, rather than spreading them evenly across the cluster. This setting can be useful if you know you are consuming the output data directly on the consumer node (such as, for ML training ingest). However, other use cases may incur a performance penalty with this setting.

Reproducibility
---------------

Deterministic execution
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   # By default, this is set to False.
   ctx.execution_options.preserve_order = True

To enable deterministic execution, set the preceding to True. This setting may decrease performance, but ensures block ordering is preserved through execution. This flag defaults to False.


.. _`file a Ray Data issue on GitHub`: https://github.com/ray-project/ray/issues/new?assignees=&labels=bug%2Ctriage%2Cdata&projects=&template=bug-report.yml&title=[data]+


.. _working_with_tensors:

Working with Tensors / NumPy
============================

N-dimensional arrays (in other words, tensors) are ubiquitous in ML workloads. This guide
describes the limitations and best practices of working with such data.

Tensor data representation
--------------------------

Ray Data represents tensors as
`NumPy ndarrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`__.

.. testcode::

    import ray

    ds = ray.data.read_images("s3://anonymous@air-example-data/digits")
    print(ds)

.. testoutput::

    Dataset(
       num_rows=100,
       schema={image: numpy.ndarray(shape=(28, 28), dtype=uint8)}
    )

Batches of fixed-shape tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your tensors have a fixed shape, Ray Data represents batches as regular ndarrays.

.. doctest::

    >>> import ray
    >>> ds = ray.data.read_images("s3://anonymous@air-example-data/digits")
    >>> batch = ds.take_batch(batch_size=32)
    >>> batch["image"].shape
    (32, 28, 28)
    >>> batch["image"].dtype
    dtype('uint8')

Batches of variable-shape tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your tensors vary in shape, Ray Data represents batches as arrays of object dtype.

.. doctest::

    >>> import ray
    >>> ds = ray.data.read_images("s3://anonymous@air-example-data/AnimalDetection")
    >>> batch = ds.take_batch(batch_size=32)
    >>> batch["image"].shape
    (32,)
    >>> batch["image"].dtype
    dtype('O')

The individual elements of these object arrays are regular ndarrays.

.. doctest::

    >>> batch["image"][0].dtype
    dtype('uint8')
    >>> batch["image"][0].shape  # doctest: +SKIP
    (375, 500, 3)
    >>> batch["image"][3].shape  # doctest: +SKIP
    (333, 465, 3)

.. _transforming_tensors:

Transforming tensor data
------------------------

Call :meth:`~ray.data.Dataset.map` or :meth:`~ray.data.Dataset.map_batches` to transform tensor data.

.. testcode::

    from typing import Any, Dict

    import ray
    import numpy as np

    ds = ray.data.read_images("s3://anonymous@air-example-data/AnimalDetection")

    def increase_brightness(row: Dict[str, Any]) -> Dict[str, Any]:
        row["image"] = np.clip(row["image"] + 4, 0, 255)
        return row

    # Increase the brightness, record at a time.
    ds.map(increase_brightness)

    def batch_increase_brightness(batch: Dict[str, np.ndarray]) -> Dict:
        batch["image"] = np.clip(batch["image"] + 4, 0, 255)
        return batch

    # Increase the brightness, batch at a time.
    ds.map_batches(batch_increase_brightness)

In addition to NumPy ndarrays, Ray Data also treats returned lists of NumPy ndarrays and
objects implementing ``__array__`` (for example, ``torch.Tensor``) as tensor data.

For more information on transforming data, read
:ref:`Transforming data <transforming_data>`.


Saving tensor data
------------------

Save tensor data with formats like Parquet, NumPy, and JSON. To view all supported
formats, see the :ref:`Input/Output reference <input-output>`.

.. tab-set::

    .. tab-item:: Parquet

        Call :meth:`~ray.data.Dataset.write_parquet` to save data in Parquet files.

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            ds.write_parquet("/tmp/simple")


    .. tab-item:: NumPy

        Call :meth:`~ray.data.Dataset.write_numpy` to save an ndarray column in NumPy
        files.

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            ds.write_numpy("/tmp/simple", column="image")

    .. tab-item:: JSON

        To save images in a JSON file, call :meth:`~ray.data.Dataset.write_json`.

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            ds.write_json("/tmp/simple")

For more information on saving data, read :ref:`Saving data <saving-data>`.


.. _iterating-over-data:

===================
Iterating over Data
===================

Ray Data lets you iterate over rows or batches of data.

This guide shows you how to:

* `Iterate over rows <#iterating-over-rows>`_
* `Iterate over batches <#iterating-over-batches>`_
* `Iterate over batches with shuffling <#iterating-over-batches-with-shuffling>`_
* `Split datasets for distributed parallel training <#splitting-datasets-for-distributed-parallel-training>`_

.. _iterating-over-rows:

Iterating over rows
===================

To iterate over the rows of your dataset, call
:meth:`Dataset.iter_rows() <ray.data.Dataset.iter_rows>`. Ray Data represents each row
as a dictionary.

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

    for row in ds.iter_rows():
        print(row)

.. testoutput::

    {'sepal length (cm)': 5.1, 'sepal width (cm)': 3.5, 'petal length (cm)': 1.4, 'petal width (cm)': 0.2, 'target': 0}
    {'sepal length (cm)': 4.9, 'sepal width (cm)': 3.0, 'petal length (cm)': 1.4, 'petal width (cm)': 0.2, 'target': 0}
    ...
    {'sepal length (cm)': 5.9, 'sepal width (cm)': 3.0, 'petal length (cm)': 5.1, 'petal width (cm)': 1.8, 'target': 2}


For more information on working with rows, see
:ref:`Transforming rows <transforming_rows>` and
:ref:`Inspecting rows <inspecting-rows>`.

.. _iterating-over-batches:

Iterating over batches
======================

A batch contains data from multiple rows. Iterate over batches of dataset in different
formats by calling one of the following methods:

* `Dataset.iter_batches() <ray.data.Dataset.iter_batches>`
* `Dataset.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`
* `Dataset.to_tf() <ray.data.Dataset.to_tf>`

.. tab-set::

    .. tab-item:: NumPy
        :sync: NumPy

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            for batch in ds.iter_batches(batch_size=2, batch_format="numpy"):
                print(batch)

        .. testoutput::
            :options: +MOCK

            {'image': array([[[[...]]]], dtype=uint8)}
            ...
            {'image': array([[[[...]]]], dtype=uint8)}

    .. tab-item:: pandas
        :sync: pandas

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

            for batch in ds.iter_batches(batch_size=2, batch_format="pandas"):
                print(batch)

        .. testoutput::
            :options: +MOCK

               sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
            0                5.1               3.5                1.4               0.2       0
            1                4.9               3.0                1.4               0.2       0
            ...
               sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
            0                6.2               3.4                5.4               2.3       2
            1                5.9               3.0                5.1               1.8       2

    .. tab-item:: Torch
        :sync: Torch

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            for batch in ds.iter_torch_batches(batch_size=2):
                print(batch)

        .. testoutput::
            :options: +MOCK

            {'image': tensor([[[[...]]]], dtype=torch.uint8)}
            ...
            {'image': tensor([[[[...]]]], dtype=torch.uint8)}

    .. tab-item:: TensorFlow
        :sync: TensorFlow

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

            tf_dataset = ds.to_tf(
                feature_columns="sepal length (cm)",
                label_columns="target",
                batch_size=2
            )
            for features, labels in tf_dataset:
                print(features, labels)

        .. testoutput::

            tf.Tensor([5.1 4.9], shape=(2,), dtype=float64) tf.Tensor([0 0], shape=(2,), dtype=int64)
            ...
            tf.Tensor([6.2 5.9], shape=(2,), dtype=float64) tf.Tensor([2 2], shape=(2,), dtype=int64)

For more information on working with batches, see
:ref:`Transforming batches <transforming_batches>` and
:ref:`Inspecting batches <inspecting-batches>`.

.. _iterating-over-batches-with-shuffling:

Iterating over batches with shuffling
=====================================

:class:`Dataset.random_shuffle <ray.data.Dataset.random_shuffle>` is slow because it
shuffles all rows. If a full global shuffle isn't required, you can shuffle a subset of
rows up to a provided buffer size during iteration by specifying
``local_shuffle_buffer_size``. While this isn't a true global shuffle like
``random_shuffle``, it's more performant because it doesn't require excessive data
movement. For more details about these options, see :doc:`Shuffling Data <shuffling-data>`.

.. tip::

    To configure ``local_shuffle_buffer_size``, choose the smallest value that achieves
    sufficient randomness. Higher values result in more randomness at the cost of slower
    iteration. See :ref:`Local shuffle when iterating over batches <local_shuffle_buffer>`
    on how to diagnose slowdowns.

.. tab-set::

    .. tab-item:: NumPy
        :sync: NumPy

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            for batch in ds.iter_batches(
                batch_size=2,
                batch_format="numpy",
                local_shuffle_buffer_size=250,
            ):
                print(batch)


        .. testoutput::
            :options: +MOCK

            {'image': array([[[[...]]]], dtype=uint8)}
            ...
            {'image': array([[[[...]]]], dtype=uint8)}

    .. tab-item:: pandas
        :sync: pandas

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

            for batch in ds.iter_batches(
                batch_size=2,
                batch_format="pandas",
                local_shuffle_buffer_size=250,
            ):
                print(batch)

        .. testoutput::
            :options: +MOCK

               sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
            0                6.3               2.9                5.6               1.8       2
            1                5.7               4.4                1.5               0.4       0
            ...
               sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
            0                5.6               2.7                4.2               1.3       1
            1                4.8               3.0                1.4               0.1       0

    .. tab-item:: Torch
        :sync: Torch

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            for batch in ds.iter_torch_batches(
                batch_size=2,
                local_shuffle_buffer_size=250,
            ):
                print(batch)

        .. testoutput::
            :options: +MOCK

            {'image': tensor([[[[...]]]], dtype=torch.uint8)}
            ...
            {'image': tensor([[[[...]]]], dtype=torch.uint8)}

    .. tab-item:: TensorFlow
        :sync: TensorFlow

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

            tf_dataset = ds.to_tf(
                feature_columns="sepal length (cm)",
                label_columns="target",
                batch_size=2,
                local_shuffle_buffer_size=250,
            )
            for features, labels in tf_dataset:
                print(features, labels)

        .. testoutput::
            :options: +MOCK

            tf.Tensor([5.2 6.3], shape=(2,), dtype=float64) tf.Tensor([1 2], shape=(2,), dtype=int64)
            ...
            tf.Tensor([5.  5.8], shape=(2,), dtype=float64) tf.Tensor([0 0], shape=(2,), dtype=int64)

Splitting datasets for distributed parallel training
====================================================

If you're performing distributed data parallel training, call
:meth:`Dataset.streaming_split <ray.data.Dataset.streaming_split>` to split your dataset
into disjoint shards.

.. note::

  If you're using :ref:`Ray Train <train-docs>`, you don't need to split the dataset.
  Ray Train automatically splits your dataset for you. To learn more, see
  :ref:`Data Loading for ML Training guide <data-ingest-torch>`.

.. testcode::

    import ray

    @ray.remote
    class Worker:

        def train(self, data_iterator):
            for batch in data_iterator.iter_batches(batch_size=8):
                pass

    ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")
    workers = [Worker.remote() for _ in range(4)]
    shards = ds.streaming_split(n=4, equal=True)
    ray.get([w.train.remote(s) for w, s in zip(workers, shards)])


.. _loading_data:

============
Loading Data
============

Ray Data loads data from various sources. This guide shows you how to:

* `Read files <#reading-files>`_ like images
* `Load in-memory data <#loading-data-from-other-libraries>`_ like pandas DataFrames
* `Read databases <#reading-databases>`_ like MySQL

.. _reading-files:

Reading files
=============

Ray Data reads files from local disk or cloud storage in a variety of file formats.
To view the full list of supported file formats, see the
:ref:`Input/Output reference <input-output>`.

.. tab-set::

    .. tab-item:: Parquet

        To read Parquet files, call :func:`~ray.data.read_parquet`.

        .. testcode::

            import ray

            ds = ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet")

            print(ds.schema())

        .. testoutput::

            Column        Type
            ------        ----
            sepal.length  double
            sepal.width   double
            petal.length  double
            petal.width   double
            variety       string

    .. tab-item:: Images

        To read raw images, call :func:`~ray.data.read_images`. Ray Data represents
        images as NumPy ndarrays.

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/batoidea/JPEGImages/")

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            image   numpy.ndarray(shape=(32, 32, 3), dtype=uint8)

    .. tab-item:: Text

        To read lines of text, call :func:`~ray.data.read_text`.

        .. testcode::

            import ray

            ds = ray.data.read_text("s3://anonymous@ray-example-data/this.txt")

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            text    string

    .. tab-item:: CSV

        To read CSV files, call :func:`~ray.data.read_csv`.

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

            print(ds.schema())

        .. testoutput::

            Column             Type
            ------             ----
            sepal length (cm)  double
            sepal width (cm)   double
            petal length (cm)  double
            petal width (cm)   double
            target             int64

    .. tab-item:: Binary

        To read raw binary files, call :func:`~ray.data.read_binary_files`.

        .. testcode::

            import ray

            ds = ray.data.read_binary_files("s3://anonymous@ray-example-data/documents")

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            bytes   binary

    .. tab-item:: TFRecords

        To read TFRecords files, call :func:`~ray.data.read_tfrecords`.

        .. testcode::

            import ray

            ds = ray.data.read_tfrecords("s3://anonymous@ray-example-data/iris.tfrecords")

            print(ds.schema())

        .. testoutput::
            :options: +MOCK

            Column        Type
            ------        ----
            label         binary
            petal.length  float
            sepal.width   float
            petal.width   float
            sepal.length  float


Reading files from local disk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To read files from local disk, call a function like :func:`~ray.data.read_parquet` and
specify paths with the ``local://`` schema. Paths can point to files or directories.

To read formats other than Parquet, see the :ref:`Input/Output reference <input-output>`.

.. tip::

    If your files are accessible on every node, exclude ``local://`` to parallelize the
    read tasks across the cluster.

.. testcode::
    :skipif: True

    import ray

    ds = ray.data.read_parquet("local:///tmp/iris.parquet")

    print(ds.schema())

.. testoutput::

    Column        Type
    ------        ----
    sepal.length  double
    sepal.width   double
    petal.length  double
    petal.width   double
    variety       string

Reading files from cloud storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To read files in cloud storage, authenticate all nodes with your cloud service provider.
Then, call a method like :func:`~ray.data.read_parquet` and specify URIs with the
appropriate schema. URIs can point to buckets, folders, or objects.

To read formats other than Parquet, see the :ref:`Input/Output reference <input-output>`.

.. tab-set::

    .. tab-item:: S3

        To read files from Amazon S3, specify URIs with the ``s3://`` scheme.

        .. testcode::

            import ray

            ds = ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet")

            print(ds.schema())

        .. testoutput::

            Column        Type
            ------        ----
            sepal.length  double
            sepal.width   double
            petal.length  double
            petal.width   double
            variety       string

        Ray Data relies on PyArrow for authenticaion with Amazon S3. For more on how to configure
        your credentials to be compatible with PyArrow, see their
        `S3 Filesytem docs <https://arrow.apache.org/docs/python/filesystems.html#s3>`_.

    .. tab-item:: GCS

        To read files from Google Cloud Storage, install the
        `Filesystem interface to Google Cloud Storage <https://gcsfs.readthedocs.io/en/latest/>`_

        .. code-block:: console

            pip install gcsfs

        Then, create a ``GCSFileSystem`` and specify URIs with the ``gcs://`` scheme.

        .. testcode::
            :skipif: True

            import ray

            filesystem = gcsfs.GCSFileSystem(project="my-google-project")
            ds = ray.data.read_parquet(
                "gcs://anonymous@ray-example-data/iris.parquet",
                filesystem=filesystem
            )

            print(ds.schema())

        .. testoutput::

            Column        Type
            ------        ----
            sepal.length  double
            sepal.width   double
            petal.length  double
            petal.width   double
            variety       string

        Ray Data relies on PyArrow for authenticaion with Google Cloud Storage. For more on how
        to configure your credentials to be compatible with PyArrow, see their
        `GCS Filesytem docs <https://arrow.apache.org/docs/python/filesystems.html#google-cloud-storage-file-system>`_.

    .. tab-item:: ABS

        To read files from Azure Blob Storage, install the
        `Filesystem interface to Azure-Datalake Gen1 and Gen2 Storage <https://pypi.org/project/adlfs/>`_

        .. code-block:: console

            pip install adlfs

        Then, create a ``AzureBlobFileSystem`` and specify URIs with the `az://` scheme.

        .. testcode::
            :skipif: True

            import adlfs
            import ray

            ds = ray.data.read_parquet(
                "az://ray-example-data/iris.parquet",
                adlfs.AzureBlobFileSystem(account_name="azureopendatastorage")
            )

            print(ds.schema())

        .. testoutput::

            Column        Type
            ------        ----
            sepal.length  double
            sepal.width   double
            petal.length  double
            petal.width   double
            variety       string

        Ray Data relies on PyArrow for authenticaion with Azure Blob Storage. For more on how
        to configure your credentials to be compatible with PyArrow, see their
        `fsspec-compatible filesystems docs <https://arrow.apache.org/docs/python/filesystems.html#using-fsspec-compatible-filesystems-with-arrow>`_.

Reading files from NFS
~~~~~~~~~~~~~~~~~~~~~~

To read files from NFS filesystems, call a function like :func:`~ray.data.read_parquet`
and specify files on the mounted filesystem. Paths can point to files or directories.

To read formats other than Parquet, see the :ref:`Input/Output reference <input-output>`.

.. testcode::
    :skipif: True

    import ray

    ds = ray.data.read_parquet("/mnt/cluster_storage/iris.parquet")

    print(ds.schema())

.. testoutput::

    Column        Type
    ------        ----
    sepal.length  double
    sepal.width   double
    petal.length  double
    petal.width   double
    variety       string

Handling compressed files
~~~~~~~~~~~~~~~~~~~~~~~~~

To read a compressed file, specify ``compression`` in ``arrow_open_stream_args``.
You can use any `codec supported by Arrow <https://arrow.apache.org/docs/python/generated/pyarrow.CompressedInputStream.html>`__.

.. testcode::

    import ray

    ds = ray.data.read_csv(
        "s3://anonymous@ray-example-data/iris.csv.gz",
        arrow_open_stream_args={"compression": "gzip"},
    )

Loading data from other libraries
=================================

Loading data from single-node data libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray Data interoperates with libraries like pandas, NumPy, and Arrow.

.. tab-set::

    .. tab-item:: Python objects

        To create a :class:`~ray.data.dataset.Dataset` from Python objects, call
        :func:`~ray.data.from_items` and pass in a list of ``Dict``. Ray Data treats
        each ``Dict`` as a row.

        .. testcode::

            import ray

            ds = ray.data.from_items([
                {"food": "spam", "price": 9.34},
                {"food": "ham", "price": 5.37},
                {"food": "eggs", "price": 0.94}
            ])

            print(ds)

        .. testoutput::

            MaterializedDataset(
               num_blocks=3,
               num_rows=3,
               schema={food: string, price: double}
            )

        You can also create a :class:`~ray.data.dataset.Dataset` from a list of regular
        Python objects.

        .. testcode::

            import ray

            ds = ray.data.from_items([1, 2, 3, 4, 5])

            print(ds)

        .. testoutput::

            MaterializedDataset(num_blocks=5, num_rows=5, schema={item: int64})

    .. tab-item:: NumPy

        To create a :class:`~ray.data.dataset.Dataset` from a NumPy array, call
        :func:`~ray.data.from_numpy`. Ray Data treats the outer axis as the row
        dimension.

        .. testcode::

            import numpy as np
            import ray

            array = np.ones((3, 2, 2))
            ds = ray.data.from_numpy(array)

            print(ds)

        .. testoutput::

            MaterializedDataset(
               num_blocks=1,
               num_rows=3,
               schema={data: numpy.ndarray(shape=(2, 2), dtype=double)}
            )

    .. tab-item:: pandas

        To create a :class:`~ray.data.dataset.Dataset` from a pandas DataFrame, call
        :func:`~ray.data.from_pandas`.

        .. testcode::

            import pandas as pd
            import ray

            df = pd.DataFrame({
                "food": ["spam", "ham", "eggs"],
                "price": [9.34, 5.37, 0.94]
            })
            ds = ray.data.from_pandas(df)

            print(ds)

        .. testoutput::

            MaterializedDataset(
               num_blocks=1,
               num_rows=3,
               schema={food: object, price: float64}
            )

    .. tab-item:: PyArrow

        To create a :class:`~ray.data.dataset.Dataset` from an Arrow table, call
        :func:`~ray.data.from_arrow`.

        .. testcode::

            import pyarrow as pa

            table = pa.table({
                "food": ["spam", "ham", "eggs"],
                "price": [9.34, 5.37, 0.94]
            })
            ds = ray.data.from_arrow(table)

            print(ds)

        .. testoutput::

            MaterializedDataset(
               num_blocks=1,
               num_rows=3,
               schema={food: string, price: double}
            )

.. _loading_datasets_from_distributed_df:

Loading data from distributed DataFrame libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray Data interoperates with distributed data processing frameworks like
:ref:`Dask <dask-on-ray>`, :ref:`Spark <spark-on-ray>`, :ref:`Modin <modin-on-ray>`, and
:ref:`Mars <mars-on-ray>`.

.. note::

    The Ray Community provides these operations but may not actively maintain them. If you run into issues,
    create a GitHub issue `here <https://github.com/ray-project/ray/issues>`__.

.. tab-set::

    .. tab-item:: Dask

        To create a :class:`~ray.data.dataset.Dataset` from a
        `Dask DataFrame <https://docs.dask.org/en/stable/dataframe.html>`__, call
        :func:`~ray.data.from_dask`. This function constructs a
        ``Dataset`` backed by the distributed Pandas DataFrame partitions that underly
        the Dask DataFrame.

        .. testcode::

            import dask.dataframe as dd
            import pandas as pd
            import ray

            df = pd.DataFrame({"col1": list(range(10000)), "col2": list(map(str, range(10000)))})
            ddf = dd.from_pandas(df, npartitions=4)
            # Create a Dataset from a Dask DataFrame.
            ds = ray.data.from_dask(ddf)

            ds.show(3)

        .. testoutput::

            {'col1': 0, 'col2': '0'}
            {'col1': 1, 'col2': '1'}
            {'col1': 2, 'col2': '2'}

    .. tab-item:: Spark

        To create a :class:`~ray.data.dataset.Dataset` from a `Spark DataFrame
        <https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html>`__,
        call :func:`~ray.data.from_spark`. This function creates a ``Dataset`` backed by
        the distributed Spark DataFrame partitions that underly the Spark DataFrame.

        .. 
            TODO: This code snippet might not work correctly. We should test it.

        .. testcode::
            :skipif: True

            import ray
            import raydp

            spark = raydp.init_spark(app_name="Spark -> Datasets Example",
                                    num_executors=2,
                                    executor_cores=2,
                                    executor_memory="500MB")
            df = spark.createDataFrame([(i, str(i)) for i in range(10000)], ["col1", "col2"])
            ds = ray.data.from_spark(df)

            ds.show(3)

        .. testoutput::

            {'col1': 0, 'col2': '0'}
            {'col1': 1, 'col2': '1'}
            {'col1': 2, 'col2': '2'}

    .. tab-item:: Modin

        To create a :class:`~ray.data.dataset.Dataset` from a Modin DataFrame, call
        :func:`~ray.data.from_modin`. This function constructs a ``Dataset`` backed by
        the distributed Pandas DataFrame partitions that underly the Modin DataFrame.

        .. testcode::

            import modin.pandas as md
            import pandas as pd
            import ray

            df = pd.DataFrame({"col1": list(range(10000)), "col2": list(map(str, range(10000)))})
            mdf = md.DataFrame(df)
            # Create a Dataset from a Modin DataFrame.
            ds = ray.data.from_modin(mdf)

            ds.show(3)

        .. testoutput::

            {'col1': 0, 'col2': '0'}
            {'col1': 1, 'col2': '1'}
            {'col1': 2, 'col2': '2'}

    .. tab-item:: Mars

        To create a :class:`~ray.data.dataset.Dataset` from a Mars DataFrame, call
        :func:`~ray.data.from_mars`. This function constructs a ``Dataset``
        backed by the distributed Pandas DataFrame partitions that underly the Mars
        DataFrame.

        .. testcode::

            import mars
            import mars.dataframe as md
            import pandas as pd
            import ray

            cluster = mars.new_cluster_in_ray(worker_num=2, worker_cpu=1)

            df = pd.DataFrame({"col1": list(range(10000)), "col2": list(map(str, range(10000)))})
            mdf = md.DataFrame(df, num_partitions=8)
            # Create a tabular Dataset from a Mars DataFrame.
            ds = ray.data.from_mars(mdf)

            ds.show(3)

        .. testoutput::

            {'col1': 0, 'col2': '0'}
            {'col1': 1, 'col2': '1'}
            {'col1': 2, 'col2': '2'}

.. _loading_datasets_from_ml_libraries:

Loading data from ML libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray Data interoperates with HuggingFace, PyTorch, and TensorFlow datasets.

.. tab-set::

    .. tab-item:: HuggingFace

        To convert a HuggingFace Dataset to a Ray Datasets, call
        :func:`~ray.data.from_huggingface`. This function accesses the underlying Arrow
        table and converts it to a Dataset directly.

        .. warning::
            :class:`~ray.data.from_huggingface` only supports parallel reads in certain
            instances, namely for untransformed public HuggingFace Datasets. For those datasets,
            Ray Data uses `hosted parquet files <https://huggingface.co/docs/datasets-server/parquet#list-parquet-files>`_
            to perform a distributed read; otherwise, Ray Data uses a single node read.
            This behavior shouldn't be an issue with in-memory HuggingFace Datasets, but may cause a failure with
            large memory-mapped HuggingFace Datasets. Additionally, HuggingFace `DatasetDict <https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.DatasetDict>`_ and
            `IterableDatasetDict <https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.IterableDatasetDict>`_
            objects aren't supported.

        .. testcode::

            import ray.data
            from datasets import load_dataset

            hf_ds = load_dataset("wikitext", "wikitext-2-raw-v1")
            ray_ds = ray.data.from_huggingface(hf_ds["train"])
            ray_ds.take(2)

        .. testoutput::
            :options: +MOCK

            [{'text': ''}, {'text': ' = Valkyria Chronicles III = \n'}]

    .. tab-item:: PyTorch

        To convert a PyTorch dataset to a Ray Dataset, call :func:`~ray.data.from_torch`.

        .. testcode::

            import ray
            from torch.utils.data import Dataset
            from torchvision import datasets
            from torchvision.transforms import ToTensor

            tds = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())
            ds = ray.data.from_torch(tds)

            print(ds)

        .. testoutput::
            :options: +MOCK

            Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to data/cifar-10-python.tar.gz
            100%|███████████████████████| 170498071/170498071 [00:07<00:00, 23494838.54it/s]
            Extracting data/cifar-10-python.tar.gz to data
            Dataset(num_rows=50000, schema={item: object})


    .. tab-item:: TensorFlow

        To convert a TensorFlow dataset to a Ray Dataset, call :func:`~ray.data.from_tf`.

        .. warning::
            :class:`~ray.data.from_tf` doesn't support parallel reads. Only use this
            function with small datasets like MNIST or CIFAR.

        .. testcode::

            import ray
            import tensorflow_datasets as tfds

            tf_ds, _ = tfds.load("cifar10", split=["train", "test"])
            ds = ray.data.from_tf(tf_ds)

            print(ds)

        ..
            The following `testoutput` is mocked to avoid illustrating download logs like
            "Downloading and preparing dataset 162.17 MiB".

        .. testoutput::
            :options: +MOCK

            MaterializedDataset(
               num_blocks=...,
               num_rows=50000,
               schema={
                  id: binary,
                  image: numpy.ndarray(shape=(32, 32, 3), dtype=uint8),
                  label: int64
               }
            )

Reading databases
=================

Ray Data reads from databases like MySQL, PostgreSQL, MongoDB, and BigQuery.

.. _reading_sql:

Reading SQL databases
~~~~~~~~~~~~~~~~~~~~~

Call :func:`~ray.data.read_sql` to read data from a database that provides a
`Python DB API2-compliant <https://peps.python.org/pep-0249/>`_ connector.

.. tab-set::

    .. tab-item:: MySQL

        To read from MySQL, install
        `MySQL Connector/Python <https://dev.mysql.com/doc/connector-python/en/>`_. It's the
        first-party MySQL database connector.

        .. code-block:: console

            pip install mysql-connector-python

        Then, define your connection logic and query the database.

        .. testcode::
            :skipif: True

            import mysql.connector

            import ray

            def create_connection():
                return mysql.connector.connect(
                    user="admin",
                    password=...,
                    host="example-mysql-database.c2c2k1yfll7o.us-west-2.rds.amazonaws.com",
                    connection_timeout=30,
                    database="example",
                )

            # Get all movies
            dataset = ray.data.read_sql("SELECT * FROM movie", create_connection)
            # Get movies after the year 1980
            dataset = ray.data.read_sql(
                "SELECT title, score FROM movie WHERE year >= 1980", create_connection
            )
            # Get the number of movies per year
            dataset = ray.data.read_sql(
                "SELECT year, COUNT(*) FROM movie GROUP BY year", create_connection
            )


    .. tab-item:: PostgreSQL

        To read from PostgreSQL, install `Psycopg 2 <https://www.psycopg.org/docs>`_. It's
        the most popular PostgreSQL database connector.

        .. code-block:: console

            pip install psycopg2-binary

        Then, define your connection logic and query the database.

        .. testcode::
            :skipif: True

            import psycopg2

            import ray

            def create_connection():
                return psycopg2.connect(
                    user="postgres",
                    password=...,
                    host="example-postgres-database.c2c2k1yfll7o.us-west-2.rds.amazonaws.com",
                    dbname="example",
                )

            # Get all movies
            dataset = ray.data.read_sql("SELECT * FROM movie", create_connection)
            # Get movies after the year 1980
            dataset = ray.data.read_sql(
                "SELECT title, score FROM movie WHERE year >= 1980", create_connection
            )
            # Get the number of movies per year
            dataset = ray.data.read_sql(
                "SELECT year, COUNT(*) FROM movie GROUP BY year", create_connection
            )

    .. tab-item:: Snowflake

        To read from Snowflake, install the
        `Snowflake Connector for Python <https://docs.snowflake.com/en/user-guide/python-connector>`_.

        .. code-block:: console

            pip install snowflake-connector-python

        Then, define your connection logic and query the database.

        .. testcode::
            :skipif: True

            import snowflake.connector

            import ray

            def create_connection():
                return snowflake.connector.connect(
                    user=...,
                    password=...
                    account="ZZKXUVH-IPB52023",
                    database="example",
                )

            # Get all movies
            dataset = ray.data.read_sql("SELECT * FROM movie", create_connection)
            # Get movies after the year 1980
            dataset = ray.data.read_sql(
                "SELECT title, score FROM movie WHERE year >= 1980", create_connection
            )
            # Get the number of movies per year
            dataset = ray.data.read_sql(
                "SELECT year, COUNT(*) FROM movie GROUP BY year", create_connection
            )


    .. tab-item:: Databricks

        To read from Databricks, set the ``DATABRICKS_TOKEN`` environment variable to
        your Databricks warehouse access token.

        .. code-block:: console

            export DATABRICKS_TOKEN=...

        If you're not running your program on the Databricks runtime, also set the
        ``DATABRICKS_HOST`` environment variable.

        .. code-block:: console

            export DATABRICKS_HOST=adb-<workspace-id>.<random-number>.azuredatabricks.net

        Then, call :func:`ray.data.read_databricks_tables` to read from the Databricks 
        SQL warehouse.

        .. testcode::
            :skipif: True

            import ray

            dataset = ray.data.read_databricks_tables(
                warehouse_id='...',  # Databricks SQL warehouse ID
                catalog='catalog_1',  # Unity catalog name
                schema='db_1',  # Schema name
                query="SELECT title, score FROM movie WHERE year >= 1980",
            )

    .. tab-item:: BigQuery

        To read from BigQuery, install the
        `Python Client for Google BigQuery <https://cloud.google.com/python/docs/reference/bigquery/latest>`_ and the `Python Client for Google BigQueryStorage <https://cloud.google.com/python/docs/reference/bigquerystorage/latest>`_.

        .. code-block:: console

            pip install google-cloud-bigquery
            pip install google-cloud-bigquery-storage

        To read data from BigQuery, call :func:`~ray.data.read_bigquery` and specify the project id, dataset, and query (if applicable).

        .. testcode::
            :skipif: True

            import ray

            # Read the entire dataset. Do not specify query.
            ds = ray.data.read_bigquery(
                project_id="my_gcloud_project_id",
                dataset="bigquery-public-data.ml_datasets.iris",
            )

            # Read from a SQL query of the dataset. Do not specify dataset.
            ds = ray.data.read_bigquery(
                project_id="my_gcloud_project_id",
                query = "SELECT * FROM `bigquery-public-data.ml_datasets.iris` LIMIT 50",
            )

            # Write back to BigQuery
            ds.write_bigquery(
                project_id="my_gcloud_project_id",
                dataset="destination_dataset.destination_table",
                overwrite_table=True,
            )

.. _reading_mongodb:

Reading MongoDB
~~~~~~~~~~~~~~~

To read data from MongoDB, call :func:`~ray.data.read_mongo` and specify
the source URI, database, and collection. You also need to specify a pipeline to
run against the collection.

.. testcode::
    :skipif: True

    import ray

    # Read a local MongoDB.
    ds = ray.data.read_mongo(
        uri="mongodb://localhost:27017",
        database="my_db",
        collection="my_collection",
        pipeline=[{"$match": {"col": {"$gte": 0, "$lt": 10}}}, {"$sort": "sort_col"}],
    )

    # Reading a remote MongoDB is the same.
    ds = ray.data.read_mongo(
        uri="mongodb://username:password@mongodb0.example.com:27017/?authSource=admin",
        database="my_db",
        collection="my_collection",
        pipeline=[{"$match": {"col": {"$gte": 0, "$lt": 10}}}, {"$sort": "sort_col"}],
    )

    # Write back to MongoDB.
    ds.write_mongo(
        MongoDatasource(),
        uri="mongodb://username:password@mongodb0.example.com:27017/?authSource=admin",
        database="my_db",
        collection="my_collection",
    )

Creating synthetic data
=======================

Synthetic datasets can be useful for testing and benchmarking.

.. tab-set::

    .. tab-item:: Int Range

        To create a synthetic :class:`~ray.data.Dataset` from a range of integers, call
        :func:`~ray.data.range`. Ray Data stores the integer range in a single column.

        .. testcode::

            import ray

            ds = ray.data.range(10000)

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            id      int64

    .. tab-item:: Tensor Range

        To create a synthetic :class:`~ray.data.Dataset` containing arrays, call
        :func:`~ray.data.range_tensor`. Ray Data packs an integer range into ndarrays of
        the provided shape.

        .. testcode::

            import ray

            ds = ray.data.range_tensor(10, shape=(64, 64))

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            data    numpy.ndarray(shape=(64, 64), dtype=int64)

Loading other datasources
==========================

If Ray Data can't load your data, subclass
:class:`~ray.data.Datasource`. Then, construct an instance of your custom
datasource and pass it to :func:`~ray.data.read_datasource`. To write results, you might
also need to subclass :class:`ray.data.Datasink`. Then, create an instance of your custom
datasink and pass it to :func:`~ray.data.Dataset.write_datasink`. For more details, see
:ref:`Advanced: Read and Write Custom File Types <custom_datasource>`.

.. testcode::
    :skipif: True

    # Read from a custom datasource.
    ds = ray.data.read_datasource(YourCustomDatasource(), **read_args)

    # Write to a custom datasink.
    ds.write_datasink(YourCustomDatasink())

Performance considerations
==========================

By default, the number of output blocks from all read tasks is dynamically decided
based on input data size and available resources. It should work well in most cases.
However, you can also override the default value by setting the ``override_num_blocks``
argument. Ray Data decides internally how many read tasks to run concurrently to best
utilize the cluster, ranging from ``1...override_num_blocks`` tasks. In other words,
the higher the ``override_num_blocks``, the smaller the data blocks in the Dataset and
hence more opportunities for parallel execution.

For more information on how to tune the number of output blocks and other suggestions
for optimizing read performance, see `Optimizing reads <performance-tips.html#optimizing-reads>`__.


.. _saving-data:

===========
Saving Data
===========

Ray Data lets you save data in files or other Python objects.

This guide shows you how to:

* `Write data to files <#writing-data-to-files>`_
* `Convert Datasets to other Python libraries <#converting-datasets-to-other-python-libraries>`_

Writing data to files
=====================

Ray Data writes to local disk and cloud storage.

Writing data to local disk
~~~~~~~~~~~~~~~~~~~~~~~~~~

To save your :class:`~ray.data.dataset.Dataset` to local disk, call a method
like :meth:`Dataset.write_parquet <ray.data.Dataset.write_parquet>`  and specify a local
directory with the `local://` scheme.

.. warning::

    If your cluster contains multiple nodes and you don't use `local://`, Ray Data
    writes different partitions of data to different nodes.

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

    ds.write_parquet("local:///tmp/iris/")

To write data to formats other than Parquet, read the
:ref:`Input/Output reference <input-output>`.

Writing data to cloud storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To save your :class:`~ray.data.dataset.Dataset` to cloud storage, authenticate all nodes
with your cloud service provider. Then, call a method like
:meth:`Dataset.write_parquet <ray.data.Dataset.write_parquet>` and specify a URI with
the appropriate scheme. URI can point to buckets or folders.

To write data to formats other than Parquet, read the :ref:`Input/Output reference <input-output>`.

.. tab-set::

    .. tab-item:: S3

        To save data to Amazon S3, specify a URI with the ``s3://`` scheme.

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

            ds.write_parquet("s3://my-bucket/my-folder")

        Ray Data relies on PyArrow for authenticaion with Amazon S3. For more on how to configure
        your credentials to be compatible with PyArrow, see their
        `S3 Filesytem docs <https://arrow.apache.org/docs/python/filesystems.html#s3>`_.

    .. tab-item:: GCS

        To save data to Google Cloud Storage, install the
        `Filesystem interface to Google Cloud Storage <https://gcsfs.readthedocs.io/en/latest/>`_

        .. code-block:: console

            pip install gcsfs

        Then, create a ``GCSFileSystem`` and specify a URI with the ``gcs://`` scheme.

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

            filesystem = gcsfs.GCSFileSystem(project="my-google-project")
            ds.write_parquet("gcs://my-bucket/my-folder", filesystem=filesystem)

        Ray Data relies on PyArrow for authenticaion with Google Cloud Storage. For more on how
        to configure your credentials to be compatible with PyArrow, see their
        `GCS Filesytem docs <https://arrow.apache.org/docs/python/filesystems.html#google-cloud-storage-file-system>`_.

    .. tab-item:: ABS

        To save data to Azure Blob Storage, install the
        `Filesystem interface to Azure-Datalake Gen1 and Gen2 Storage <https://pypi.org/project/adlfs/>`_

        .. code-block:: console

            pip install adlfs

        Then, create a ``AzureBlobFileSystem`` and specify a URI with the ``az://`` scheme.

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

            filesystem = adlfs.AzureBlobFileSystem(account_name="azureopendatastorage")
            ds.write_parquet("az://my-bucket/my-folder", filesystem=filesystem)

        Ray Data relies on PyArrow for authenticaion with Azure Blob Storage. For more on how
        to configure your credentials to be compatible with PyArrow, see their
        `fsspec-compatible filesystems docs <https://arrow.apache.org/docs/python/filesystems.html#using-fsspec-compatible-filesystems-with-arrow>`_.

Writing data to NFS
~~~~~~~~~~~~~~~~~~~

To save your :class:`~ray.data.dataset.Dataset` to NFS file systems, call a method
like :meth:`Dataset.write_parquet <ray.data.Dataset.write_parquet>` and specify a
mounted directory.

.. testcode::
    :skipif: True

    import ray

    ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

    ds.write_parquet("/mnt/cluster_storage/iris")

To write data to formats other than Parquet, read the
:ref:`Input/Output reference <input-output>`.

.. _changing-number-output-files:

Changing the number of output files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you call a write method, Ray Data writes your data to several files. To control the
number of output files, configure ``num_rows_per_file``.

.. note::

    ``num_rows_per_file`` is a hint, not a strict limit. Ray Data might write more or 
    fewer rows to each file.

.. testcode::

    import os
    import ray

    ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")
    ds.write_csv("/tmp/few_files/", num_rows_per_file=75)

    print(os.listdir("/tmp/few_files/"))

.. testoutput::
    :options: +MOCK

    ['0_000001_000000.csv', '0_000000_000000.csv', '0_000002_000000.csv']                                                          

Converting Datasets to other Python libraries
=============================================

Converting Datasets to pandas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert a :class:`~ray.data.dataset.Dataset` to a pandas DataFrame, call
:meth:`Dataset.to_pandas() <ray.data.Dataset.to_pandas>`. Your data must fit in memory
on the head node.

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

    df = ds.to_pandas()
    print(df)

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

         sepal length (cm)  sepal width (cm)  ...  petal width (cm)  target
    0                  5.1               3.5  ...               0.2       0
    1                  4.9               3.0  ...               0.2       0
    2                  4.7               3.2  ...               0.2       0
    3                  4.6               3.1  ...               0.2       0
    4                  5.0               3.6  ...               0.2       0
    ..                 ...               ...  ...               ...     ...
    145                6.7               3.0  ...               2.3       2
    146                6.3               2.5  ...               1.9       2
    147                6.5               3.0  ...               2.0       2
    148                6.2               3.4  ...               2.3       2
    149                5.9               3.0  ...               1.8       2
    <BLANKLINE>
    [150 rows x 5 columns]

Converting Datasets to distributed DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray Data interoperates with distributed data processing frameworks like
:ref:`Dask <dask-on-ray>`, :ref:`Spark <spark-on-ray>`, :ref:`Modin <modin-on-ray>`, and
:ref:`Mars <mars-on-ray>`.

.. tab-set::

    .. tab-item:: Dask

        To convert a :class:`~ray.data.dataset.Dataset` to a
        `Dask DataFrame <https://docs.dask.org/en/stable/dataframe.html>`__, call
        :meth:`Dataset.to_dask() <ray.data.Dataset.to_dask>`.

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

            df = ds.to_dask()

    .. tab-item:: Spark

        To convert a :class:`~ray.data.dataset.Dataset` to a `Spark DataFrame
        <https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html>`__,
        call :meth:`Dataset.to_spark() <ray.data.Dataset.to_spark>`.

        .. testcode::

            import ray
            import raydp

            spark = raydp.init_spark(
                app_name = "example",
                num_executors = 1,
                executor_cores = 4,
                executor_memory = "512M"
            )

            ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")
            df = ds.to_spark(spark)

        .. testcode::
            :hide:

            raydp.stop_spark()

    .. tab-item:: Modin

        To convert a :class:`~ray.data.dataset.Dataset` to a Modin DataFrame, call
        :meth:`Dataset.to_modin() <ray.data.Dataset.to_modin>`.

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

            mdf = ds.to_modin()

    .. tab-item:: Mars

        To convert a :class:`~ray.data.dataset.Dataset` from a Mars DataFrame, call
        :meth:`Dataset.to_mars() <ray.data.Dataset.to_mars>`.

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

            mdf = ds.to_mars()

.. _custom_datasource:

Advanced: Read and Write Custom File Types 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. vale off

.. Ignoring Vale because of future tense.

This guide shows you how to extend Ray Data to read and write file types that aren't 
natively supported. This is an advanced guide, and you'll use unstable internal APIs.

.. vale on

Images are already supported with the :func:`~ray.data.read_images` 
and :meth:`~ray.data.Dataset.write_images` APIs, but this example shows you how to 
implement them for illustrative purposes.

Read data from files
--------------------

.. tip::
    If you're not contributing to Ray Data, you don't need to create a 
    :class:`~ray.data.Datasource`. Instead, you can call 
    :func:`~ray.data.read_binary_files` and decode files with 
    :meth:`~ray.data.Dataset.map`.

The core abstraction for reading files is :class:`~ray.data.datasource.FileBasedDatasource`.
It provides file-specific functionality on top of the 
:class:`~ray.data.Datasource` interface.

To subclass :class:`~ray.data.datasource.FileBasedDatasource`, implement the constructor 
and ``_read_stream``.

Implement the constructor
=========================

Call the superclass constructor and specify the files you want to read.
Optionally, specify valid file extensions. Ray Data ignores files with other extensions.

.. literalinclude:: doc_code/custom_datasource_example.py
    :language: python
    :start-after: __datasource_constructor_start__
    :end-before: __datasource_constructor_end__

Implement ``_read_stream``
==========================

``_read_stream`` is a generator that yields one or more blocks of data from a file. 

`Blocks <https://github.com/ray-project/ray/blob/23d3bfcb9dd97ea666b7b4b389f29b9cc0810121/python/ray/data/block.py#L54>`_ 
are a Data-internal abstraction for a collection of rows. They can be PyArrow tables, 
pandas DataFrames, or dictionaries of NumPy arrays. 

Don't create a block directly. Instead, add rows of data to a 
`DelegatingBlockBuilder <https://github.com/ray-project/ray/blob/23d3bfcb9dd97ea666b7b4b389f29b9cc0810121/python/ray/data/_internal/delegating_block_builder.py#L10>`_.

.. literalinclude:: doc_code/custom_datasource_example.py
    :language: python
    :start-after: __read_stream_start__
    :end-before: __read_stream_end__

Read your data
==============

Once you've implemented ``ImageDatasource``, call :func:`~ray.data.read_datasource` to 
read images into a :class:`~ray.data.Dataset`. Ray Data reads your files in parallel.

.. literalinclude:: doc_code/custom_datasource_example.py
    :language: python
    :start-after: __read_datasource_start__
    :end-before: __read_datasource_end__

Write data to files
-------------------

.. note::
    The write interface is under active development and might change in the future. If
    you have feature requests,
    `open a GitHub Issue <https://github.com/ray-project/ray/issues/new?assignees=&labels=enhancement%2Ctriage&projects=&template=feature-request.yml&title=%5B%3CRay+component%3A+Core%7CRLlib%7Cetc...%3E%5D+>`_.

The core abstractions for writing data to files are :class:`~ray.data.datasource.RowBasedFileDatasink` and 
:class:`~ray.data.datasource.BlockBasedFileDatasink`. They provide file-specific functionality on top of the
:class:`~ray.data.Datasink` interface.

If you want to write one row per file, subclass :class:`~ray.data.datasource.RowBasedFileDatasink`. 
Otherwise, subclass :class:`~ray.data.datasource.BlockBasedFileDatasink`.

.. vale off

.. Ignoring Vale because of future tense.

In this example, you'll write one image per file, so you'll subclass 
:class:`~ray.data.datasource.RowBasedFileDatasink`. To subclass 
:class:`~ray.data.datasource.RowBasedFileDatasink`, implement the constructor and 
:meth:`~ray.data.datasource.RowBasedFileDatasink.write_row_to_file`.

.. vale on

Implement the constructor
=========================

Call the superclass constructor and specify the folder to write to. Optionally, specify
a string representing the file format (for example, ``"png"``). Ray Data uses the
file format as the file extension.

.. literalinclude:: doc_code/custom_datasource_example.py
    :language: python
    :start-after: __datasink_constructor_start__
    :end-before: __datasink_constructor_end__

Implement ``write_row_to_file``
===============================

``write_row_to_file`` writes a row of data to a file. Each row is a dictionary that maps
column names to values. 

.. literalinclude:: doc_code/custom_datasource_example.py
    :language: python
    :start-after: __write_row_to_file_start__
    :end-before: __write_row_to_file_end__

Write your data
===============

Once you've implemented ``ImageDatasink``, call :meth:`~ray.data.Dataset.write_datasink`
to write images to files. Ray Data writes to multiple files in parallel.

.. literalinclude:: doc_code/custom_datasource_example.py
    :language: python
    :start-after: __write_datasink_start__
    :end-before: __write_datasink_end__


.. _execution-options-api:

ExecutionOptions API
====================

.. currentmodule:: ray.data

Constructor
-----------

.. autosummary::
   :nosignatures:
   :toctree: doc/
   :template: autosummary/class_without_autosummary.rst

   ExecutionOptions

Resource Options
----------------

.. autosummary::
   :nosignatures:
   :toctree: doc/
   :template: autosummary/class_without_autosummary.rst

   ExecutionResources


.. _data-api:

Ray Data API
================

.. toctree::
    :maxdepth: 2

    input_output.rst
    dataset.rst
    data_iterator.rst
    execution_options.rst
    grouped_data.rst
    data_context.rst
    utility.rst
    preprocessor.rst
    from_other_data_libs.rst


.. _input-output:

Input/Output
============

.. currentmodule:: ray.data

Synthetic Data
--------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   range
   range_tensor

Python Objects
--------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   from_items

Parquet
-------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_parquet
   read_parquet_bulk
   Dataset.write_parquet

CSV
---

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_csv
   Dataset.write_csv

JSON
----

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_json
   Dataset.write_json

Text
----

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_text

Avro
----

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_avro

Images
------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_images
   Dataset.write_images

Binary
------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_binary_files

TFRecords
---------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_tfrecords
   Dataset.write_tfrecords


Pandas
------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   from_pandas
   from_pandas_refs
   Dataset.to_pandas
   Dataset.to_pandas_refs

NumPy
-----

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_numpy
   from_numpy
   from_numpy_refs
   Dataset.write_numpy
   Dataset.to_numpy_refs

Arrow
-----

.. autosummary::
   :nosignatures:
   :toctree: doc/

   from_arrow
   from_arrow_refs
   Dataset.to_arrow_refs

MongoDB
-------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_mongo
   Dataset.write_mongo

BigQuery
--------

.. autosummary::
   :toctree: doc/

   read_bigquery
   Dataset.write_bigquery

SQL Databases
-------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_sql
   Dataset.write_sql

Databricks
----------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_databricks_tables

Lance
-----

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_lance

Dask
----

.. autosummary::
   :nosignatures:
   :toctree: doc/

   from_dask
   Dataset.to_dask

Spark
-----

.. autosummary::
   :nosignatures:
   :toctree: doc/

   from_spark
   Dataset.to_spark

Modin
-----

.. autosummary::
   :nosignatures:
   :toctree: doc/

   from_modin
   Dataset.to_modin

Mars
----

.. autosummary::
   :nosignatures:
   :toctree: doc/

   from_mars
   Dataset.to_mars

Torch
-----

.. autosummary::
   :nosignatures:
   :toctree: doc/

   from_torch

Hugging Face
------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   from_huggingface

TensorFlow
----------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   from_tf

WebDataset
----------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_webdataset

.. _data_source_api:

Datasource API
--------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   read_datasource
   Datasource
   ReadTask
   datasource.FilenameProvider

Datasink API
------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   Dataset.write_datasink
   Datasink
   datasource.RowBasedFileDatasink
   datasource.BlockBasedFileDatasink
   datasource.FileBasedDatasource

Partitioning API
----------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   datasource.Partitioning
   datasource.PartitionStyle
   datasource.PathPartitionParser
   datasource.PathPartitionFilter

.. _metadata_provider:

MetadataProvider API
--------------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   datasource.FileMetadataProvider
   datasource.BaseFileMetadataProvider
   datasource.ParquetMetadataProvider
   datasource.DefaultFileMetadataProvider
   datasource.DefaultParquetMetadataProvider
   datasource.FastFileMetadataProvider

   


.. _preprocessor-ref:

Preprocessor
============

Preprocessor Interface
------------------------

.. currentmodule:: ray.data

Constructor
~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~preprocessor.Preprocessor

Fit/Transform APIs
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~preprocessor.Preprocessor.fit
    ~preprocessor.Preprocessor.fit_transform
    ~preprocessor.Preprocessor.transform
    ~preprocessor.Preprocessor.transform_batch
    ~preprocessor.PreprocessorNotFittedException


Generic Preprocessors
---------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~preprocessors.Concatenator
    ~preprocessors.SimpleImputer

Categorical Encoders
--------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~preprocessors.Categorizer
    ~preprocessors.LabelEncoder
    ~preprocessors.MultiHotEncoder
    ~preprocessors.OneHotEncoder
    ~preprocessors.OrdinalEncoder

Feature Scalers
---------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~preprocessors.MaxAbsScaler
    ~preprocessors.MinMaxScaler
    ~preprocessors.Normalizer
    ~preprocessors.PowerTransformer
    ~preprocessors.RobustScaler
    ~preprocessors.StandardScaler

K-Bins Discretizers
-------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~preprocessors.CustomKBinsDiscretizer
    ~preprocessors.UniformKBinsDiscretizer


.. _api-guide-for-users-from-other-data-libs:

API Guide for Users from Other Data Libraries
=============================================

Ray Data is a data loading and preprocessing library for ML. It shares certain
similarities with other ETL data processing libraries, but also has its own focus.
This guide provides API mappings for users who come from those data
libraries, so you can quickly map what you may already know to Ray Data APIs.

.. note::

  - This is meant to map APIs that perform comparable but not necessarily identical operations.
    Select the API reference for exact semantics and usage.
  - This list may not be exhaustive: Ray Data isn't a traditional ETL data processing library, so not all data processing APIs can map to Datasets.
    In addition, this list focuses on common APIs or APIs that are less obvious to see a connection.

.. _api-guide-for-pandas-users:

For Pandas Users
----------------

.. list-table:: Pandas DataFrame vs. Ray Data APIs
   :header-rows: 1

   * - Pandas DataFrame API
     - Ray Data API
   * - df.head()
     - :meth:`ds.show() <ray.data.Dataset.show>`, :meth:`ds.take() <ray.data.Dataset.take>`, or :meth:`ds.take_batch() <ray.data.Dataset.take_batch>`
   * - df.dtypes
     - :meth:`ds.schema() <ray.data.Dataset.schema>`
   * - len(df) or df.shape[0]
     - :meth:`ds.count() <ray.data.Dataset.count>`
   * - df.truncate()
     - :meth:`ds.limit() <ray.data.Dataset.limit>`
   * - df.iterrows()
     - :meth:`ds.iter_rows() <ray.data.Dataset.iter_rows>`
   * - df.drop()
     - :meth:`ds.drop_columns() <ray.data.Dataset.drop_columns>`
   * - df.transform()
     - :meth:`ds.map_batches() <ray.data.Dataset.map_batches>` or :meth:`ds.map() <ray.data.Dataset.map>`
   * - df.groupby()
     - :meth:`ds.groupby() <ray.data.Dataset.groupby>`
   * - df.groupby().apply()
     - :meth:`ds.groupby().map_groups() <ray.data.grouped_data.GroupedData.map_groups>`
   * - df.sample()
     - :meth:`ds.random_sample() <ray.data.Dataset.random_sample>`
   * - df.sort_values()
     - :meth:`ds.sort() <ray.data.Dataset.sort>`
   * - df.append()
     - :meth:`ds.union() <ray.data.Dataset.union>`
   * - df.aggregate()
     - :meth:`ds.aggregate() <ray.data.Dataset.aggregate>`
   * - df.min()
     - :meth:`ds.min() <ray.data.Dataset.min>`
   * - df.max()
     - :meth:`ds.max() <ray.data.Dataset.max>`
   * - df.sum()
     - :meth:`ds.sum() <ray.data.Dataset.sum>`
   * - df.mean()
     - :meth:`ds.mean() <ray.data.Dataset.mean>`
   * - df.std()
     - :meth:`ds.std() <ray.data.Dataset.std>`

.. _api-guide-for-pyarrow-users:

For PyArrow Users
-----------------

.. list-table:: PyArrow Table vs. Ray Data APIs
   :header-rows: 1

   * - PyArrow Table API
     - Ray Data API
   * - ``pa.Table.schema``
     - :meth:`ds.schema() <ray.data.Dataset.schema>`
   * - ``pa.Table.num_rows``
     - :meth:`ds.count() <ray.data.Dataset.count>`
   * - ``pa.Table.filter()``
     - :meth:`ds.filter() <ray.data.Dataset.filter>`
   * - ``pa.Table.drop()``
     - :meth:`ds.drop_columns() <ray.data.Dataset.drop_columns>`
   * - ``pa.Table.add_column()``
     - :meth:`ds.add_column() <ray.data.Dataset.add_column>`
   * - ``pa.Table.groupby()``
     - :meth:`ds.groupby() <ray.data.Dataset.groupby>`
   * - ``pa.Table.sort_by()``
     - :meth:`ds.sort() <ray.data.Dataset.sort>`


For PyTorch Dataset & DataLoader Users
--------------------------------------

For more details, see the :ref:`Migrating from PyTorch to Ray Data <migrate_pytorch>`.


.. _data-context-api:

Global configuration
====================

.. currentmodule:: ray.data

.. autoclass:: DataContext

.. autosummary::
   :nosignatures:
   :toctree: doc/

   DataContext.get_current


.. _dataset-iterator-api:

DataIterator API
================

.. currentmodule:: ray.data

.. autoclass:: DataIterator

.. autosummary::
   :nosignatures:
   :toctree: doc/

   DataIterator.iter_batches
   DataIterator.iter_torch_batches
   DataIterator.to_tf
   DataIterator.materialize
   DataIterator.stats


.. _dataset-api:

Dataset API
==============

.. currentmodule:: ray.data

Constructor
-----------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   Dataset

Basic Transformations
---------------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   Dataset.map
   Dataset.map_batches
   Dataset.flat_map
   Dataset.filter
   Dataset.add_column
   Dataset.drop_columns
   Dataset.select_columns
   Dataset.random_sample
   Dataset.limit

Sorting, Shuffling, Repartitioning
----------------------------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   Dataset.sort
   Dataset.random_shuffle
   Dataset.randomize_block_order
   Dataset.repartition

Splitting and Merging Datasets
---------------------------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   Dataset.split
   Dataset.split_at_indices
   Dataset.split_proportionately
   Dataset.streaming_split
   Dataset.train_test_split
   Dataset.union
   Dataset.zip

Grouped and Global Aggregations
-------------------------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   Dataset.groupby
   Dataset.unique
   Dataset.aggregate
   Dataset.sum
   Dataset.min
   Dataset.max
   Dataset.mean
   Dataset.std

Consuming Data
---------------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   Dataset.show
   Dataset.take
   Dataset.take_batch
   Dataset.take_all
   Dataset.iterator
   Dataset.iter_rows
   Dataset.iter_batches
   Dataset.iter_torch_batches
   Dataset.iter_tf_batches

I/O and Conversion
------------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   Dataset.write_parquet
   Dataset.write_json
   Dataset.write_csv
   Dataset.write_numpy
   Dataset.write_tfrecords
   Dataset.write_webdataset
   Dataset.write_mongo
   Dataset.write_datasource
   Dataset.to_torch
   Dataset.to_tf
   Dataset.to_dask
   Dataset.to_mars
   Dataset.to_modin
   Dataset.to_spark
   Dataset.to_pandas
   Dataset.to_pandas_refs
   Dataset.to_numpy_refs
   Dataset.to_arrow_refs

Inspecting Metadata
-------------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   Dataset.count
   Dataset.columns
   Dataset.schema
   Dataset.num_blocks
   Dataset.size_bytes
   Dataset.input_files
   Dataset.stats
   Dataset.get_internal_block_refs

Execution
---------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    Dataset.materialize

.. _block-api:

Internals
---------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   block.Block
   block.BlockExecStats
   block.BlockMetadata
   block.BlockAccessor


.. _grouped-dataset-api:

GroupedData API
===============

.. currentmodule:: ray.data

GroupedData objects are returned by groupby call: 
:meth:`Dataset.groupby() <ray.data.Dataset.groupby>`.

Constructor
-----------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   grouped_data.GroupedData

Computations / Descriptive Stats
--------------------------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   grouped_data.GroupedData.count
   grouped_data.GroupedData.sum
   grouped_data.GroupedData.min
   grouped_data.GroupedData.max
   grouped_data.GroupedData.mean
   grouped_data.GroupedData.std

Function Application
--------------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   grouped_data.GroupedData.aggregate
   grouped_data.GroupedData.map_groups

Aggregate Function
------------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   aggregate.AggregateFn
   aggregate.Count
   aggregate.Sum
   aggregate.Max
   aggregate.Mean
   aggregate.Std
   aggregate.AbsMax


.. _data-utility:

Utility
=======

.. currentmodule:: ray.data

.. autosummary::
   :nosignatures:
   :toctree: doc/

   set_progress_bars


