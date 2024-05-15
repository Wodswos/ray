.. _tune-faq:

Ray Tune FAQ
------------

Here we try to answer questions that come up often.
If you still have questions after reading this FAQ, let us know!

.. contents::
    :local:
    :depth: 1


What are Hyperparameters?
~~~~~~~~~~~~~~~~~~~~~~~~~

What are *hyperparameters?* And how are they different from *model parameters*?

In supervised learning, we train a model with labeled data so the model can properly identify new data values.
Everything about the model is defined by a set of parameters, such as the weights in a linear regression. These
are *model parameters*; they are learned during training.

.. image:: /images/hyper-model-parameters.png

In contrast, the *hyperparameters* define structural details about the kind of model itself, like whether or not
we are using a linear regression or classification, what architecture is best for a neural network,
how many layers, what kind of filters, etc. They are defined before training, not learned.

.. image:: /images/hyper-network-params.png

Other quantities considered *hyperparameters* include learning rates, discount rates, etc. If we want our training
process and resulting model to work well, we first need to determine the optimal or near-optimal set of *hyperparameters*.

How do we determine the optimal *hyperparameters*? The most direct approach is to perform a loop where we pick
a candidate set of values from some reasonably inclusive list of possible values, train a model, compare the results
achieved with previous loop iterations, and pick the set that performed best. This process is called
*Hyperparameter Tuning* or *Optimization* (HPO). And *hyperparameters* are specified over a configured and confined
search space, collectively defined for each *hyperparameter* in a ``config`` dictionary.


.. TODO: We *really* need to improve this section.

Which search algorithm/scheduler should I choose?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ray Tune offers :ref:`many different search algorithms <tune-search-alg>`
and :ref:`schedulers <tune-schedulers>`.
Deciding on which to use mostly depends on your problem:

* Is it a small or large problem (how long does it take to train? How costly
  are the resources, like GPUs)? Can you run many trials in parallel?
* How many hyperparameters would you like to tune?
* What values are valid for hyperparameters?

**If your model returns incremental results** (eg. results per epoch in deep learning,
results per each added tree in GBDTs, etc.) using early stopping usually allows for sampling
more configurations, as unpromising trials are pruned before they run their full course.
Please note that not all search algorithms can use information from pruned trials.
Early stopping cannot be used without incremental results - in case of the functional API,
that means that ``session.report()`` has to be called more than once - usually in a loop.

**If your model is small**, you can usually try to run many different configurations.
A **random search** can be used to generate configurations. You can also grid search
over some values. You should probably still use
:ref:`ASHA for early termination of bad trials <tune-scheduler-hyperband>` (if your problem
supports early stopping).

**If your model is large**, you can try to either use
**Bayesian Optimization-based search algorithms** like :ref:`BayesOpt <bayesopt>`
to get good parameter configurations after few
trials. :ref:`Ax <tune-ax>` is similar but more robust to noisy data.
Please note that these algorithms only work well with **a small number of hyperparameters**.
Alternatively, you can use :ref:`Population Based Training <tune-scheduler-pbt>` which
works well with few trials, e.g. 8 or even 4. However, this will output a hyperparameter *schedule* rather
than one fixed set of hyperparameters.

**If you have a small number of hyperparameters**, Bayesian Optimization methods
work well. Take a look at :ref:`BOHB <tune-scheduler-bohb>` or :ref:`Optuna <tune-optuna>`
with the :ref:`ASHA <tune-scheduler-hyperband>` scheduler to combine the
benefits of Bayesian Optimization with early stopping.

**If you only have continuous values for hyperparameters** this will work well
with most Bayesian Optimization methods. Discrete or categorical variables still
work, but less good with an increasing number of categories.

**If you have many categorical values for hyperparameters**, consider using random search,
or a TPE-based Bayesian Optimization algorithm such as :ref:`Optuna <tune-optuna>` or
:ref:`HyperOpt <tune-hyperopt>`.

**Our go-to solution** is usually to use **random search** with
:ref:`ASHA for early stopping <tune-scheduler-hyperband>` for smaller problems.
Use :ref:`BOHB <tune-scheduler-bohb>` for **larger problems** with a **small number of hyperparameters**
and :ref:`Population Based Training <tune-scheduler-pbt>` for **larger problems** with a
**large number of hyperparameters** if a learning schedule is acceptable.


How do I choose hyperparameter ranges?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A good start is to look at the papers that introduced the algorithms, and also
to see what other people are using.

Most algorithms also have sensible defaults for some of their parameters.
For instance, `XGBoost's parameter overview <https://xgboost.readthedocs.io/en/latest/parameter.html>`_
reports to use ``max_depth=6`` for the maximum decision tree depth. Here, anything
between 2 and 10 might make sense (though that naturally depends on your problem).

For **learning rates**, we suggest using a **loguniform distribution** between
**1e-5** and **1e-1**: ``tune.loguniform(1e-5, 1e-1)``.

For **batch sizes**, we suggest trying **powers of 2**, for instance, 2, 4, 8,
16, 32, 64, 128, 256, etc. The magnitude depends on your problem. For easy
problems with lots of data, use higher batch sizes, for harder problems with
not so much data, use lower batch sizes.

For **layer sizes** we also suggest trying **powers of 2**. For small problems
(e.g. Cartpole), use smaller layer sizes. For larger problems, try larger ones.

For **discount factors** in reinforcement learning we suggest sampling uniformly
between 0.9 and 1.0. Depending on the problem, a much stricter range above 0.97
or oeven above 0.99 can make sense (e.g. for Atari).


How can I use nested/conditional search spaces?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sometimes you might need to define parameters whose value depend on the value
of other parameters. Ray Tune offers some methods to define these.

Nested spaces
'''''''''''''
You can nest hyperparameter definition in sub dictionaries:

.. literalinclude:: doc_code/faq.py
    :language: python
    :start-after: __basic_config_start__
    :end-before: __basic_config_end__

The trial config will be nested exactly like the input config.


Conditional spaces
''''''''''''''''''
:ref:`Custom and conditional search spaces are explained in detail here <tune_custom-search>`.
In short, you can pass custom functions to ``tune.sample_from()`` that can
return values that depend on other values:

.. literalinclude:: doc_code/faq.py
    :language: python
    :start-after: __conditional_spaces_start__
    :end-before: __conditional_spaces_end__


Conditional grid search
'''''''''''''''''''''''
If you would like to grid search over two parameters that depend on each other,
this might not work out of the box. For instance say that *a* should be a value
between 5 and 10 and *b* should be a value between 0 and a. In this case, we
cannot use ``tune.sample_from`` because it doesn't support grid searching.

The solution here is to create a list of valid *tuples* with the help of a
helper function, like this:

.. literalinclude:: doc_code/faq.py
    :language: python
    :start-after: __iter_start__
    :end-before: __iter_end__


Your trainable then can do something like ``a, b = config["ab"]`` to split
the a and b variables and use them afterwards.


How does early termination (e.g. Hyperband/ASHA) work?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Early termination algorithms look at the intermediately reported values,
e.g. what is reported to them via ``session.report()`` after each training
epoch. After a certain number of steps, they then remove the worst
performing trials and keep only the best performing trials. Goodness of a trial
is determined by ordering them by the objective metric, for instance accuracy
or loss.

In ASHA, you can decide how many trials are early terminated.
``reduction_factor=4`` means that only 25% of all trials are kept each
time they are reduced. With ``grace_period=n`` you can force ASHA to
train each trial at least for ``n`` epochs.


Why are all my trials returning "1" iteration?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**This is most likely applicable for the Tune function API.**

Ray Tune counts iterations internally every time ``session.report()`` is
called. If you only call ``session.report()`` once at the end of the training,
the counter has only been incremented once. If you're using the class API,
the counter is increased after calling ``step()``.

Note that it might make sense to report metrics more often than once. For
instance, if you train your algorithm for 1000 timesteps, consider reporting
intermediate performance values every 100 steps. That way, schedulers
like Hyperband/ASHA can terminate bad performing trials early.


What are all these extra outputs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You'll notice that Ray Tune not only reports hyperparameters (from the
``config``) or metrics (passed to ``session.report()``), but also some other
outputs.

.. code-block:: bash

    Result for easy_objective_c64c9112:
      date: 2020-10-07_13-29-18
      done: false
      experiment_id: 6edc31257b564bf8985afeec1df618ee
      experiment_tag: 7_activation=tanh,height=-53.116,steps=100,width=13.885
      hostname: ubuntu
      iterations: 0
      iterations_since_restore: 1
      mean_loss: 4.688385317424468
      neg_mean_loss: -4.688385317424468
      node_ip: 192.168.1.115
      pid: 5973
      time_since_restore: 7.605552673339844e-05
      time_this_iter_s: 7.605552673339844e-05
      time_total_s: 7.605552673339844e-05
      timestamp: 1602102558
      timesteps_since_restore: 0
      training_iteration: 1
      trial_id: c64c9112

See the :ref:`tune-autofilled-metrics` section for a glossary.

How do I set resources?
~~~~~~~~~~~~~~~~~~~~~~~
If you want to allocate specific resources to a trial, you can use the
``tune.with_resources`` and wrap it around you trainable together with
a dict or a :class:`PlacementGroupFactory <ray.tune.execution.placement_groups.PlacementGroupFactory>` object:

.. literalinclude:: doc_code/faq.py
    :dedent:
    :language: python
    :start-after: __resources_start__
    :end-before: __resources_end__

The example above showcases three things:

1. The `cpu` and `gpu` options set how many CPUs and GPUs are available for
   each trial, respectively. **Trials cannot request more resources** than these
   (exception: see 3).
2. It is possible to request **fractional GPUs**. A value of 0.5 means that
   half of the memory of the GPU is made available to the trial. You will have
   to make sure yourself that your model still fits on the fractional memory.
3. You can request custom resources you supplied to Ray when starting the cluster.
   Trials will only be scheduled on single nodes that can provide all resources you
   requested.

One important thing to keep in mind is that each Ray worker (and thus each
Ray Tune Trial) will only be scheduled on **one machine**. That means if
you for instance request 2 GPUs for your trial, but your cluster consists
of 4 machines with 1 GPU each, the trial will never be scheduled.

In other words, you will have to make sure that your Ray cluster
has machines that can actually fulfill your resource requests.

In some cases your trainable might want to start other remote actors, for instance if you're
leveraging distributed training via Ray Train. In these cases, you can use
:ref:`placement groups <ray-placement-group-doc-ref>` to request additional resources:

.. literalinclude:: doc_code/faq.py
    :dedent:
    :language: python
    :start-after: __resources_pgf_start__
    :end-before: __resources_pgf_end__

Here, you're requesting 2 additional CPUs for remote tasks. These two additional
actors do not necessarily have to live on the same node as your main trainable.
In fact, you can control this via the ``strategy`` parameter. In this example, ``PACK``
will try to schedule the actors on the same node, but allows them to be scheduled
on other nodes as well. Please refer to the
:ref:`placement groups documentation <ray-placement-group-doc-ref>` to learn more
about these placement strategies.

You can also allocate specific resources to a trial based on a custom rule via lambda functions.
For instance, if you want to allocate GPU resources to trials based on a setting in your param space:

.. literalinclude:: doc_code/faq.py
    :dedent:
    :language: python
    :start-after: __resources_lambda_start__
    :end-before: __resources_lambda_end__


Why is my training stuck and Ray reporting that pending actor or tasks cannot be scheduled?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is usually caused by Ray actors or tasks being started by the
trainable without the trainable resources accounting for them, leading to a deadlock.
This can also be "stealthly" caused by using other libraries in the trainable that are
based on Ray, such as Modin. In order to fix the issue, request additional resources for
the trial using :ref:`placement groups <ray-placement-group-doc-ref>`, as outlined in
the section above.

For example, if your trainable is using Modin dataframes, operations on those will spawn
Ray tasks. By allocating an additional CPU bundle to the trial, those tasks will be able
to run without being starved of resources.

.. literalinclude:: doc_code/faq.py
    :dedent:
    :language: python
    :start-after: __modin_start__
    :end-before: __modin_end__


How can I pass further parameter values to my trainable?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray Tune expects your trainable functions to accept only up to two parameters,
``config`` and ``checkpoint_dir``. But sometimes there are cases where
you want to pass constant arguments, like the number of epochs to run,
or a dataset to train on. Ray Tune offers a wrapper function to achieve
just that, called :func:`tune.with_parameters() <ray.tune.with_parameters>`:

.. literalinclude:: doc_code/faq.py
    :language: python
    :start-after: __huge_data_start__
    :end-before: __huge_data_end__


This function works similarly to ``functools.partial``, but it stores
the parameters directly in the Ray object store. This means that you
can pass even huge objects like datasets, and Ray makes sure that these
are efficiently stored and retrieved on your cluster machines.

:func:`tune.with_parameters() <ray.tune.with_parameters>`
also works with class trainables. Please see
:func:`tune.with_parameters() <ray.tune.with_parameters>` for more details and examples.


How can I reproduce experiments?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reproducing experiments and experiment results means that you get the exact same
results when running an experiment again and again. To achieve this, the
conditions have to be exactly the same each time you run the exeriment.
In terms of ML training and tuning, this mostly concerns
the random number generators that are used for sampling in various places of the
training and tuning lifecycle.

Random number generators are used to create randomness, for instance to sample a hyperparameter
value for a parameter you defined. There is no true randomness in computing, rather
there are sophisticated algorithms that generate numbers that *seem* to be random and
fulfill all properties of a random distribution. These algorithms can be *seeded* with
an initial state, after which the generated random numbers are always the same.

.. literalinclude:: doc_code/faq.py
    :language: python
    :start-after: __seeded_1_start__
    :end-before: __seeded_1_end__

The most commonly used random number generators from Python libraries are those in the
native ``random`` submodule and the ``numpy.random`` module.

.. literalinclude:: doc_code/faq.py
    :language: python
    :start-after: __seeded_2_start__
    :end-before: __seeded_2_end__

In your tuning and training run, there are several places where randomness occurs, and
at all these places we will have to introduce seeds to make sure we get the same behavior.

* **Search algorithm**: Search algorithms have to be seeded to generate the same
  hyperparameter configurations in each run. Some search algorithms can be explicitly instantiated with a
  random seed (look for a ``seed`` parameter in the constructor). For others, try to use
  the above code block.
* **Schedulers**: Schedulers like Population Based Training rely on resampling some
  of the parameters, requiring randomness. Use the code block above to set the initial
  seeds.
* **Training function**: In addition to initializing the configurations, the training
  functions themselves have to use seeds. This could concern e.g. the data splitting.
  You should make sure to set the seed at the start of your training function.

PyTorch and TensorFlow use their own RNGs, which have to be initialized, too:

.. literalinclude:: doc_code/faq.py
    :language: python
    :start-after: __torch_tf_seeds_start__
    :end-before: __torch_tf_seeds_end__

You should thus seed both Ray Tune's schedulers and search algorithms, and the
training code. The schedulers and search algorithms should always be seeded with the
same seed. This is also true for the training code, but often it is beneficial that
the seeds differ *between different training runs*.

Here's a blueprint on how to do all this in your training code:

.. literalinclude:: doc_code/faq.py
    :language: python
    :start-after: __torch_seed_example_start__
    :end-before: __torch_seed_example_end__


**Please note** that it is not always possible to control all sources of non-determinism.
For instance, if you use schedulers like ASHA or PBT, some trials might finish earlier
than other trials, affecting the behavior of the schedulers. Which trials finish first
can however depend on the current system load, network communication, or other factors
in the envrionment that we cannot control with random seeds. This is also true for search
algorithms such as Bayesian Optimization, which take previous results into account when
sampling new configurations. This can be tackled by
using the **synchronous modes** of PBT and Hyperband, where the schedulers wait for all trials to
finish an epoch before deciding which trials to promote.

We strongly advise to try reproduction on smaller toy problems first before relying
on it for larger experiments.


.. _tune-bottlenecks:

How can I avoid bottlenecks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sometimes you might run into a message like this:

.. code-block::

    The `experiment_checkpoint` operation took 2.43 seconds to complete, which may be a performance bottleneck

Most commonly, the ``experiment_checkpoint`` operation is throwing this warning, but it might be something else,
like ``process_trial_result``.

These operations should usually take less than 500ms to complete. When it consistently takes longer, this might
indicate a problem or inefficiencies. To get rid of this message, it is important to understand where it comes
from.

These are the main reasons this problem comes up:

**The Trial config is very large**

This is the case if you e.g. try to pass a dataset or other large object via the ``config`` parameter.
If this is the case, the dataset is serialized and written to disk repeatedly during experiment
checkpointing, which takes a long time.

**Solution**: Use :func:`tune.with_parameters <ray.tune.with_parameters>` to pass large objects to
function trainables via the objects store. For class trainables you can do this manually via ``ray.put()``
and ``ray.get()``. If you need to pass a class definition, consider passing an
indicator (e.g. a string) instead and let the trainable select the class instead. Generally, your config
dictionary should only contain primitive types, like numbers or strings.

**The Trial result is very large**

This is the case if you return objects, data, or other large objects via the return value of ``step()`` in
your class trainable or to ``session.report()`` in your function trainable. The effect is the same as above:
The results are repeatedly serialized and written to disk, and this can take a long time.

**Solution**: Use checkpoint by writing data to the trainable's current working directory instead. There are various ways
to do that depending on whether you are using class or functional Trainable API. 

**You are training a large number of trials on a cluster, or you are saving huge checkpoints**

**Solution**: You can use :ref:`cloud checkpointing <tune-cloud-checkpointing>` to save logs and checkpoints to a specified `storage_path`.
This is the preferred way to deal with this. All syncing will be taken care of automatically, as all nodes
are able to access the cloud storage. Additionally, your results will be safe, so even when you're working on
pre-emptible instances, you won't lose any of your data.

**You are reporting results too often**

Each result is processed by the search algorithm, trial scheduler, and callbacks (including loggers and the
trial syncer). If you're reporting a large number of results per trial (e.g. multiple results per second),
this can take a long time.

**Solution**: The solution here is obvious: Just don't report results that often. In class trainables, ``step()``
should maybe process a larger chunk of data. In function trainables, you can report only every n-th iteration
of the training loop. Try to balance the number of results you really need to make scheduling or searching
decisions. If you need more fine grained metrics for logging or tracking, consider using a separate logging
mechanism for this instead of the Ray Tune-provided progress logging of results.

How can I develop and test Tune locally?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, follow the instructions in :ref:`python-develop` to develop Tune without compiling Ray.
After Ray is set up, run ``pip install -r ray/python/ray/tune/requirements-dev.txt`` to install all packages
required for Tune development. Now, to run all Tune tests simply run:

.. code-block:: shell

    pytest ray/python/ray/tune/tests/

If you plan to submit a pull request, we recommend you to run unit tests locally beforehand to speed up the review process.
Even though we have hooks to run unit tests automatically for each pull request, it's usually quicker to run them
on your machine first to avoid any obvious mistakes.


How can I get started contributing to Tune?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use Github to track issues, feature requests, and bugs. Take a look at the
ones labeled `"good first issue" <https://github.com/ray-project/ray/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`__ and `"help wanted" <https://github.com/ray-project/ray/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`__ for a place to start.
Look for issues with "[tune]" in the title.

.. note::

    If raising a new issue or PR related to Tune, be sure to include "[tune]" in the title and add a ``tune`` label.

For project organization, Tune maintains a relatively up-to-date organization of
issues on the `Tune Github Project Board <https://github.com/ray-project/ray/projects/4>`__.
Here, you can track and identify how issues are organized.


.. _tune-reproducible:

How can I make my Tune experiments reproducible?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exact reproducibility of machine learning runs is hard to achieve. This
is even more true in a distributed setting, as more non-determinism is
introduced. For instance, if two trials finish at the same time, the
convergence of the search algorithm might be influenced by which trial
result is processed first. This depends on the searcher - for random search,
this shouldn't make a difference, but for most other searchers it will.

If you try to achieve some amount of reproducibility, there are two
places where you'll have to set random seeds:

1. On the driver program, e.g. for the search algorithm. This will ensure
   that at least the initial configurations suggested by the search
   algorithms are the same.

2. In the trainable (if required). Neural networks are usually initialized
   with random numbers, and many classical ML algorithms, like GBDTs, make use of
   randomness. Thus you'll want to make sure to set a seed here
   so that the initialization is always the same.

Here is an example that will always produce the same result (except for trial
runtimes).

.. literalinclude:: doc_code/faq.py
    :language: python
    :start-after: __reproducible_start__
    :end-before: __reproducible_end__


Some searchers use their own random states to sample new configurations.
These searchers usually accept a ``seed`` parameter that can be passed on
initialization. Other searchers use Numpy's ``np.random`` interface -
these seeds can be then set with ``np.random.seed()``. We don't offer an
interface to do this in the searcher classes as setting a random seed
globally could have side effects. For instance, it could influence the
way your dataset is split. Thus, we leave it up to the user to make
these global configuration changes.


How can I use large datasets in Tune?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You often will want to compute a large object (e.g., training data, model weights) on the driver and use that
object within each trial.

Tune provides a wrapper function ``tune.with_parameters()`` that allows you to broadcast large objects to your trainable.
Objects passed with this wrapper will be stored on the :ref:`Ray object store <objects-in-ray>` and will
be automatically fetched and passed to your trainable as a parameter.


.. tip:: If the objects are small in size or already exist in the :ref:`Ray Object Store <objects-in-ray>`, there's no need to use ``tune.with_parameters()``. You can use `partials <https://docs.python.org/3/library/functools.html#functools.partial>`__ or pass in directly to ``config`` instead.

.. literalinclude:: doc_code/faq.py
    :language: python
    :start-after: __large_data_start__
    :end-before: __large_data_end__


.. _tune-cloud-syncing:

How can I upload my Tune results to cloud storage?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`tune-cloud-checkpointing`.

Make sure that worker nodes have the write access to the cloud storage.
Failing to do so would cause error messages like ``Error message (1): fatal error: Unable to locate credentials``.
For AWS set up, this involves adding an IamInstanceProfile configuration for worker nodes.
Please :ref:`see here for more tips <aws-cluster-s3>`.


.. _tune-kubernetes:

How can I use Tune with Kubernetes?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You should configure shared storage. See this user guide: :ref:`tune-storage-options`.

.. _tune-docker:

How can I use Tune with Docker?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You should configure shared storage. See this user guide: :ref:`tune-storage-options`.


.. _tune-default-search-space:

How do I configure search spaces?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can specify a grid search or sampling distribution via the dict passed into ``Tuner(param_space=...)``.

.. literalinclude:: doc_code/faq.py
    :dedent:
    :language: python
    :start-after: __grid_search_start__
    :end-before: __grid_search_end__

By default, each random variable and grid search point is sampled once.
To take multiple random samples, add ``num_samples: N`` to the experiment config.
If `grid_search` is provided as an argument, the grid will be repeated ``num_samples`` of times.

.. literalinclude:: doc_code/faq.py
    :emphasize-lines: 16
    :language: python
    :start-after: __grid_search_2_start__
    :end-before: __grid_search_2_end__

Note that search spaces may not be interoperable across different search algorithms.
For example, for many search algorithms, you will not be able to use a ``grid_search`` or ``sample_from`` parameters.
Read about this in the :ref:`Search Space API <tune-search-space>` page.

.. _tune-working-dir:

How do I access relative filepaths in my Tune training function?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say you launch a Tune experiment with ``my_script.py`` from inside ``~/code``.
By default, Tune changes the working directory of each worker to its corresponding trial
directory (e.g. ``~/ray_results/exp_name/trial_0000x``). This guarantees separate working
directories for each worker process, avoiding conflicts when saving trial-specific outputs.

You can configure this by setting the `RAY_CHDIR_TO_TRIAL_DIR=0` environment variable.
This explicitly tells Tune to not change the working directory
to the trial directory, giving access to paths relative to the original working directory.
One caveat is that the working directory is now shared between workers, so the
:meth:`train.get_context().get_trial_dir() <ray.train.context.TrainContext.get_.get_trial_dir>`
API should be used to get the path for saving trial-specific outputs.

.. literalinclude:: doc_code/faq.py
    :dedent:
    :emphasize-lines: 3, 10, 11, 12, 16
    :language: python
    :start-after: __no_chdir_start__
    :end-before: __no_chdir_end__

.. warning::

    The `TUNE_ORIG_WORKING_DIR` environment variable was the original workaround for
    accessing paths relative to the original working directory. This environment
    variable is deprecated, and the `RAY_CHDIR_TO_TRIAL_DIR` environment variable above
    should be used instead.


.. _tune-multi-tenancy:

How can I run multiple Ray Tune jobs on the same cluster at the same time (multi-tenancy)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running multiple Ray Tune runs on the same cluster at the same
time is not officially supported. We do not test this workflow and we recommend
using a separate cluster for each tuning job.

The reasons for this are:

1. When multiple Ray Tune jobs run at the same time, they compete for resources.
   One job could run all its trials at the same time, while the other job waits
   for a long time until it gets resources to run the first trial.
2. If it is easy to start a new Ray cluster on your infrastructure, there is often
   no cost benefit to running one large cluster instead of multiple smaller
   clusters. For instance, running one cluster of 32 instances incurs almost the same
   cost as running 4 clusters with 8 instances each.
3. Concurrent jobs are harder to debug. If a trial of job A fills the disk,
   trials from job B on the same node are impacted. In practice, it's hard
   to reason about these conditions from the logs if something goes wrong.

Previously, some internal implementations in Ray Tune assumed that you only have one job
running at a time. A symptom was when trials from job A used parameters specified in job B,
leading to unexpected results.

Please refer to
[this github issue](https://github.com/ray-project/ray/issues/30091#issuecomment-1431676976)
for more context and a workaround if you run into this issue.

.. _tune-iterative-experimentation:

How can I continue training a completed Tune experiment for longer and with new configurations (iterative experimentation)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say that I have a Tune experiment that has completed with the following configurations:

.. literalinclude:: /tune/doc_code/faq.py
    :language: python
    :start-after: __iter_experimentation_initial_start__
    :end-before: __iter_experimentation_initial_end__

Now, I want to continue training from a checkpoint (e.g., the best one) generated by the previous experiment,
and search over a new hyperparameter search space, for another ``10`` epochs.

:ref:`tune-fault-tolerance-ref` explains that the usage of :meth:`Tuner.restore <ray.tune.Tuner.restore>`
is meant for resuming an *unfinished* experiment that was interrupted in the middle,
according to the *exact configuration* that was supplied in the initial training run.

Therefore, ``Tuner.restore`` is not suitable for our desired behavior.
This style of "iterative experimentation" should be done with *new* Tune experiments
rather than restoring a single experiment over and over and modifying the experiment spec.

See the following for an example of how to create a new experiment that builds off of the old one:

.. literalinclude:: /tune/doc_code/faq.py
    :language: python
    :start-after: __iter_experimentation_resume_start__
    :end-before: __iter_experimentation_resume_end__


.. _tune-main:

Ray Tune: Hyperparameter Tuning
===============================

.. toctree::
    :hidden:

    Getting Started <getting-started>
    Key Concepts <key-concepts>
    tutorials/overview
    examples/index
    faq
    api/api

.. image:: images/tune_overview.png
    :scale: 50%
    :align: center

Tune is a Python library for experiment execution and hyperparameter tuning at any scale.
You can tune your favorite machine learning framework (:ref:`PyTorch <tune-pytorch-cifar-ref>`, :ref:`XGBoost <tune-xgboost-ref>`, :doc:`TensorFlow and Keras <examples/tune_mnist_keras>`, and :doc:`more <examples/index>`) by running state of the art algorithms such as :ref:`Population Based Training (PBT) <tune-scheduler-pbt>` and :ref:`HyperBand/ASHA <tune-scheduler-hyperband>`.
Tune further integrates with a wide range of additional hyperparameter optimization tools, including :doc:`Ax <examples/ax_example>`, :doc:`BayesOpt <examples/bayesopt_example>`, :doc:`BOHB <examples/bohb_example>`, :doc:`Nevergrad <examples/nevergrad_example>`, and :doc:`Optuna <examples/optuna_example>`.

**Click on the following tabs to see code examples for various machine learning frameworks**:

.. tab-set::

    .. tab-item:: Quickstart

        To run this example, install the following: ``pip install "ray[tune]"``.

        In this quick-start example you `minimize` a simple function of the form ``f(x) = a**2 + b``, our `objective` function.
        The closer ``a`` is to zero and the smaller ``b`` is, the smaller the total value of ``f(x)``.
        We will define a so-called `search space` for  ``a`` and ``b`` and let Ray Tune explore the space for good values.

        .. callout::

            .. literalinclude:: ../../../python/ray/tune/tests/example.py
               :language: python
               :start-after: __quick_start_begin__
               :end-before: __quick_start_end__

            .. annotations::
                <1> Define an objective function.

                <2> Define a search space.

                <3> Start a Tune run and print the best result.


    .. tab-item:: Keras+Hyperopt

        To tune your Keras models with Hyperopt, you wrap your model in an objective function whose ``config`` you
        can access for selecting hyperparameters.
        In the example below we only tune the ``activation`` parameter of the first layer of the model, but you can
        tune any parameter of the model you want.
        After defining the search space, you can simply initialize the ``HyperOptSearch`` object and pass it to ``run``.
        It's important to tell Ray Tune which metric you want to optimize and whether you want to maximize or minimize it.

        .. callout::

            .. literalinclude:: doc_code/keras_hyperopt.py
                :language: python
                :start-after: __keras_hyperopt_start__
                :end-before: __keras_hyperopt_end__

            .. annotations::
                <1> Wrap a Keras model in an objective function.

                <2> Define a search space and initialize the search algorithm.

                <3> Start a Tune run that maximizes accuracy.

    .. tab-item:: PyTorch+Optuna

        To tune your PyTorch models with Optuna, you wrap your model in an objective function whose ``config`` you
        can access for selecting hyperparameters.
        In the example below we only tune the ``momentum`` and learning rate (``lr``) parameters of the model's optimizer,
        but you can tune any other model parameter you want.
        After defining the search space, you can simply initialize the ``OptunaSearch`` object and pass it to ``run``.
        It's important to tell Ray Tune which metric you want to optimize and whether you want to maximize or minimize it.
        We stop tuning this training run after ``5`` iterations, but you can easily define other stopping rules as well.


        .. callout::

            .. literalinclude:: doc_code/pytorch_optuna.py
                :language: python
                :start-after: __pytorch_optuna_start__
                :end-before: __pytorch_optuna_end__

            .. annotations::
                <1> Wrap a PyTorch model in an objective function.

                <2> Define a search space and initialize the search algorithm.

                <3> Start a Tune run that maximizes mean accuracy and stops after 5 iterations.

With Tune you can also launch a multi-node :ref:`distributed hyperparameter sweep <tune-distributed-ref>`
in less than 10 lines of code.
And you can move your models from training to serving on the same infrastructure with `Ray Serve`_.

.. _`Ray Serve`: ../serve/index.html


.. grid:: 1 2 3 4
    :gutter: 1
    :class-container: container pb-3

    .. grid-item-card::

        **Getting Started**
        ^^^

        In our getting started tutorial you will learn how to tune a PyTorch model
        effectively with Tune.

        +++
        .. button-ref:: tune-tutorial
            :color: primary
            :outline:
            :expand:

            Get Started with Tune

    .. grid-item-card::

        **Key Concepts**
        ^^^

        Understand the key concepts behind Ray Tune.
        Learn about tune runs, search algorithms, schedulers and other features.

        +++
        .. button-ref:: tune-60-seconds
            :color: primary
            :outline:
            :expand:

            Tune's Key Concepts

    .. grid-item-card::

        **User Guides**
        ^^^

        Our guides teach you about key features of Tune,
        such as distributed training or early stopping.


        +++
        .. button-ref:: tune-guides
            :color: primary
            :outline:
            :expand:

            Learn How To Use Tune

    .. grid-item-card::

        **Examples**
        ^^^

        In our examples you can find practical tutorials for using frameworks such as
        scikit-learn, Keras, TensorFlow, PyTorch, and mlflow, and state of the art search algorithm integrations.

        +++
        .. button-ref::  tune-examples-ref
            :color: primary
            :outline:
            :expand:

            Ray Tune Examples

    .. grid-item-card::

        **Ray Tune FAQ**
        ^^^

        Find answers to commonly asked questions in our detailed FAQ.

        +++
        .. button-ref:: tune-faq
            :color: primary
            :outline:
            :expand:

            Ray Tune FAQ

    .. grid-item-card::

        **Ray Tune API**
        ^^^

        Get more in-depth information about the Ray Tune API, including all about search spaces,
        algorithms and training configurations.

        +++
        .. button-ref:: tune-api-ref
            :color: primary
            :outline:
            :expand:

            Read the API Reference


Why choose Tune?
----------------

There are many other hyperparameter optimization libraries out there.
If you're new to Tune, you're probably wondering, "what makes Tune different?"

.. dropdown:: Cutting-Edge Optimization Algorithms
    :animate: fade-in-slide-down

    As a user, you're probably looking into hyperparameter optimization because you want to quickly increase your
    model performance.

    Tune enables you to leverage a variety of these cutting edge optimization algorithms, reducing the cost of tuning
    by `terminating bad runs early <tune-scheduler-hyperband>`_,
    :ref:`choosing better parameters to evaluate <tune-search-alg>`, or even
    :ref:`changing the hyperparameters during training <tune-scheduler-pbt>` to optimize schedules.

.. dropdown:: First-class Developer Productivity
    :animate: fade-in-slide-down

    A key problem with many hyperparameter optimization frameworks is the need to restructure
    your code to fit the framework.
    With Tune, you can optimize your model just by :ref:`adding a few code snippets <tune-tutorial>`.

    Also, Tune removes boilerplate from your code training workflow,
    supporting :ref:`multiple storage options for experiment results (NFS, cloud storage) <tune-storage-options>` and
    :ref:`logs results to tools <tune-logging>` such as MLflow and TensorBoard, while also being highly customizable.

.. dropdown:: Multi-GPU & Distributed Training Out Of The Box
    :animate: fade-in-slide-down

    Hyperparameter tuning is known to be highly time-consuming, so it is often necessary to parallelize this process.
    Most other tuning frameworks require you to implement your own multi-process framework or build your own
    distributed system to speed up hyperparameter tuning.

    However, Tune allows you to transparently :ref:`parallelize across multiple GPUs and multiple nodes <tune-parallelism>`.
    Tune even has seamless :ref:`fault tolerance and cloud support <tune-distributed-ref>`, allowing you to scale up
    your hyperparameter search by 100x while reducing costs by up to 10x by using cheap preemptible instances.

.. dropdown:: Coming From Another Hyperparameter Optimization Tool?
    :animate: fade-in-slide-down

    You might be already using an existing hyperparameter tuning tool such as HyperOpt or Bayesian Optimization.

    In this situation, Tune actually allows you to power up your existing workflow.
    Tune's :ref:`Search Algorithms <tune-search-alg>` integrate with a variety of popular hyperparameter tuning
    libraries (see :ref:`examples <tune-examples-ref>`) and allow you to seamlessly scale up your optimization
    process - without sacrificing performance.

Projects using Tune
-------------------

Here are some of the popular open source repositories and research projects that leverage Tune.
Feel free to submit a pull-request adding (or requesting a removal!) of a listed project.

- `Softlearning <https://github.com/rail-berkeley/softlearning>`_: Softlearning is a reinforcement learning framework for training maximum entropy policies in continuous domains. Includes the official implementation of the Soft Actor-Critic algorithm.
- `Flambe <https://github.com/asappresearch/flambe>`_: An ML framework to accelerate research and its path to production. See `flambe.ai <https://flambe.ai>`_.
- `Population Based Augmentation <https://github.com/arcelien/pba>`_: Population Based Augmentation (PBA) is a algorithm that quickly and efficiently learns data augmentation functions for neural network training. PBA matches state-of-the-art results on CIFAR with one thousand times less compute.
- `Fast AutoAugment by Kakao <https://github.com/kakaobrain/fast-autoaugment>`_: Fast AutoAugment (Accepted at NeurIPS 2019) learns augmentation policies using a more efficient search strategy based on density matching.
- `Allentune <https://github.com/allenai/allentune>`_: Hyperparameter Search for AllenNLP from AllenAI.
- `machinable <https://github.com/frthjf/machinable>`_: A modular configuration system for machine learning research. See `machinable.org <https://machinable.org>`_.
- `NeuroCard <https://github.com/neurocard/neurocard>`_: NeuroCard (Accepted at VLDB 2021) is a neural cardinality estimator for multi-table join queries. It uses state of the art deep density models to learn correlations across relational database tables.



Learn More About Ray Tune
-------------------------

Below you can find blog posts and talks about Ray Tune:

- [blog] `Tune: a Python library for fast hyperparameter tuning at any scale <https://towardsdatascience.com/fast-hyperparameter-tuning-at-scale-d428223b081c>`_
- [blog] `Cutting edge hyperparameter tuning with Ray Tune <https://medium.com/riselab/cutting-edge-hyperparameter-tuning-with-ray-tune-be6c0447afdf>`_
- [blog] `Simple hyperparameter and architecture search in tensorflow with Ray Tune <http://louiskirsch.com/ai/ray-tune>`_
- [slides] `Talk given at RISECamp 2019 <https://docs.google.com/presentation/d/1v3IldXWrFNMK-vuONlSdEuM82fuGTrNUDuwtfx4axsQ/edit?usp=sharing>`_
- [video] `Talk given at RISECamp 2018 <https://www.youtube.com/watch?v=38Yd_dXW51Q>`_
- [video] `A Guide to Modern Hyperparameter Optimization (PyData LA 2019) <https://www.youtube.com/watch?v=10uz5U3Gy6E>`_ (`slides <https://speakerdeck.com/richardliaw/a-modern-guide-to-hyperparameter-optimization>`_)

Citing Tune
-----------

If Tune helps you in your academic research, you are encouraged to cite `our paper <https://arxiv.org/abs/1807.05118>`__.
Here is an example bibtex:

.. code-block:: tex

    @article{liaw2018tune,
        title={Tune: A Research Platform for Distributed Model Selection and Training},
        author={Liaw, Richard and Liang, Eric and Nishihara, Robert
                and Moritz, Philipp and Gonzalez, Joseph E and Stoica, Ion},
        journal={arXiv preprint arXiv:1807.05118},
        year={2018}
    }


.. _tune-tutorial:

.. TODO: make this an executable notebook later on.

Getting Started with Ray Tune
=============================

This tutorial will walk you through the process of setting up a Tune experiment.
To get started, we take a PyTorch model and show you how to leverage Ray Tune to
optimize the hyperparameters of this model.
Specifically, we'll leverage early stopping and Bayesian Optimization via HyperOpt to do so.

.. tip:: If you have suggestions on how to improve this tutorial,
    please `let us know <https://github.com/ray-project/ray/issues/new/choose>`_!

To run this example, you will need to install the following:

.. code-block:: bash

    $ pip install "ray[tune]" torch torchvision

Setting Up a Pytorch Model to Tune
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To start off, let's first import some dependencies.
We import some PyTorch and TorchVision modules to help us create a model and train it.
Also, we'll import Ray Tune to help us optimize the model.
As you can see we use a so-called scheduler, in this case the ``ASHAScheduler``
that we will use for tuning the model later in this tutorial.

.. literalinclude:: /../../python/ray/tune/tests/tutorial.py
   :language: python
   :start-after: __tutorial_imports_begin__
   :end-before: __tutorial_imports_end__

Then, let's define a simple PyTorch model that we'll be training.
If you're not familiar with PyTorch, the simplest way to define a model is to implement a ``nn.Module``.
This requires you to set up your model with ``__init__`` and then implement a ``forward`` pass.
In this example we're using a small convolutional neural network consisting of one 2D convolutional layer, a fully
connected layer, and a softmax function.

.. literalinclude:: /../../python/ray/tune/tests/tutorial.py
   :language: python
   :start-after:  __model_def_begin__
   :end-before:  __model_def_end__

Below, we have implemented functions for training and evaluating your Pytorch model.
We define a ``train`` and a ``test`` function for that purpose.
If you know how to do this, skip ahead to the next section.

.. dropdown:: Training and evaluating the model

    .. literalinclude:: /../../python/ray/tune/tests/tutorial.py
       :language: python
       :start-after: __train_def_begin__
       :end-before: __train_def_end__

.. _tutorial-tune-setup:

Setting up a ``Tuner`` for a Training Run with Tune
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below, we define a function that trains the Pytorch model for multiple epochs.
This function will be executed on a separate :ref:`Ray Actor (process) <actor-guide>` underneath the hood,
so we need to communicate the performance of the model back to Tune (which is on the main Python process).

To do this, we call :func:`train.report() <ray.train.report>` in our training function,
which sends the performance value back to Tune. Since the function is executed on the separate process,
make sure that the function is :ref:`serializable by Ray <serialization-guide>`.

.. literalinclude:: /../../python/ray/tune/tests/tutorial.py
   :language: python
   :start-after: __train_func_begin__
   :end-before: __train_func_end__

Let's run one trial by calling :ref:`Tuner.fit <tune-run-ref>` and :ref:`randomly sample <tune-search-space>`
from a uniform distribution for learning rate and momentum.

.. literalinclude:: /../../python/ray/tune/tests/tutorial.py
   :language: python
   :start-after: __eval_func_begin__
   :end-before: __eval_func_end__

``Tuner.fit`` returns an :ref:`ResultGrid object <tune-analysis-docs>`.
You can use this to plot the performance of this trial.

.. literalinclude:: /../../python/ray/tune/tests/tutorial.py
   :language: python
   :start-after: __plot_begin__
   :end-before: __plot_end__

.. note:: Tune will automatically run parallel trials across all available cores/GPUs on your machine or cluster.
    To limit the number of concurrent trials, use the :ref:`ConcurrencyLimiter <limiter>`.


Early Stopping with Adaptive Successive Halving (ASHAScheduler)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's integrate early stopping into our optimization process. Let's use :ref:`ASHA <tune-scheduler-hyperband>`, a scalable algorithm for `principled early stopping`_.

.. _`principled early stopping`: https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/

On a high level, ASHA terminates trials that are less promising and allocates more time and resources to more promising trials.
As our optimization process becomes more efficient, we can afford to **increase the search space by 5x**, by adjusting the parameter ``num_samples``.

ASHA is implemented in Tune as a "Trial Scheduler".
These Trial Schedulers can early terminate bad trials, pause trials, clone trials, and alter hyperparameters of a running trial.
See :ref:`the TrialScheduler documentation <tune-schedulers>` for more details of available schedulers and library integrations.

.. literalinclude:: /../../python/ray/tune/tests/tutorial.py
   :language: python
   :start-after: __run_scheduler_begin__
   :end-before: __run_scheduler_end__

You can run the below in a Jupyter notebook to visualize trial progress.

.. literalinclude:: /../../python/ray/tune/tests/tutorial.py
   :language: python
   :start-after: __plot_scheduler_begin__
   :end-before: __plot_scheduler_end__

.. image:: /images/tune-df-plot.png
    :scale: 50%
    :align: center

You can also use :ref:`TensorBoard <tensorboard>` for visualizing results.

.. code:: bash

    $ tensorboard --logdir {logdir}


Using Search Algorithms in Tune
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to :ref:`TrialSchedulers <tune-schedulers>`, you can further optimize your hyperparameters
by using an intelligent search technique like Bayesian Optimization.
To do this, you can use a Tune :ref:`Search Algorithm <tune-search-alg>`.
Search Algorithms leverage optimization algorithms to intelligently navigate the given hyperparameter space.

Note that each library has a specific way of defining the search space.

.. literalinclude:: /../../python/ray/tune/tests/tutorial.py
   :language: python
   :start-after: __run_searchalg_begin__
   :end-before: __run_searchalg_end__

.. note:: Tune allows you to use some search algorithms in combination with different trial schedulers. See :ref:`this page for more details <tune-schedulers>`.

Evaluating Your Model after Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can evaluate best trained model using the :ref:`ExperimentAnalysis object <tune-analysis-docs>` to retrieve the best model:

.. literalinclude:: /../../python/ray/tune/tests/tutorial.py
   :language: python
   :start-after: __run_analysis_begin__
   :end-before: __run_analysis_end__


Next Steps
----------

* Check out the :ref:`Tune tutorials <tune-guides>` for guides on using Tune with your preferred machine learning library.
* Browse our :doc:`gallery of examples <examples/other-examples>` to see how to use Tune with PyTorch, XGBoost, Tensorflow, etc.
* `Let us know <https://github.com/ray-project/ray/issues>`__ if you ran into issues or have any questions by opening an issue on our Github.
* To check how your application is doing, you can use the :ref:`Ray dashboard <observability-getting-started>`.


.. _tune-60-seconds:

========================
Key Concepts of Ray Tune
========================

.. TODO: should we introduce checkpoints as well?
.. TODO: should we at least mention "Stopper" classes here?

Let's quickly walk through the key concepts you need to know to use Tune.
If you want to see practical tutorials right away, go visit our :ref:`user guides <tune-guides>`.
In essence, Tune has six crucial components that you need to understand.

First, you define the hyperparameters you want to tune in a `search space` and pass them into a `trainable`
that specifies the objective you want to tune.
Then you select a `search algorithm` to effectively optimize your parameters and optionally use a
`scheduler` to stop searches early and speed up your experiments.
Together with other configuration, your `trainable`, search algorithm, and scheduler are passed into ``Tuner``,
which runs your experiments and creates `trials`.
The `Tuner` returns a `ResultGrid` to inspect your experiment results.
The following figure shows an overview of these components, which we cover in detail in the next sections.

.. image:: images/tune_flow.png

.. _tune_60_seconds_trainables:

Ray Tune Trainables
-------------------

In short, a :ref:`Trainable <trainable-docs>` is an object that you can pass into a Tune run.
Ray Tune has two ways of defining a `trainable`, namely the :ref:`Function API <tune-function-api>`
and the :ref:`Class API <tune-class-api>`.
Both are valid ways of defining a `trainable`, but the Function API is generally recommended and is used
throughout the rest of this guide.

Let's say we want to optimize a simple objective function like ``a (x ** 2) + b`` in which ``a`` and ``b`` are the
hyperparameters we want to tune to `minimize` the objective.
Since the objective also has a variable ``x``, we need to test for different values of ``x``.
Given concrete choices for ``a``, ``b`` and ``x`` we can evaluate the objective function and get a `score` to minimize.

.. tab-set::

    .. tab-item:: Function API

        With the :ref:`the function-based API <tune-function-api>` you create a function (here called ``trainable``) that
        takes in a dictionary of hyperparameters.
        This function computes a ``score`` in a "training loop" and `reports` this score back to Tune:

        .. literalinclude:: doc_code/key_concepts.py
            :language: python
            :start-after: __function_api_start__
            :end-before: __function_api_end__

        Note that we use ``session.report(...)`` to report the intermediate ``score`` in the training loop, which can be useful
        in many machine learning tasks.
        If you just want to report the final ``score`` outside of this loop, you can simply return the score at the
        end of the ``trainable`` function with ``return {"score": score}``.
        You can also use ``yield {"score": score}`` instead of ``session.report()``.

    .. tab-item:: Class API

        Here's an example of specifying the objective function using the :ref:`class-based API <tune-class-api>`:

        .. literalinclude:: doc_code/key_concepts.py
            :language: python
            :start-after: __class_api_start__
            :end-before: __class_api_end__

        .. tip:: ``session.report`` can't be used within a ``Trainable`` class.

Learn more about the details of :ref:`Trainables here <trainable-docs>`
and :doc:`have a look at our examples <examples/other-examples>`.
Next, let's have a closer look at what the ``config`` dictionary is that you pass into your trainables.

.. _tune-key-concepts-search-spaces:

Tune Search Spaces
------------------

To optimize your *hyperparameters*, you have to define a *search space*.
A search space defines valid values for your hyperparameters and can specify
how these values are sampled (e.g. from a uniform distribution or a normal
distribution).

Tune offers various functions to define search spaces and sampling methods.
:ref:`You can find the documentation of these search space definitions here <tune-search-space>`.

Here's an example covering all search space functions. Again,
:ref:`here is the full explanation of all these functions <tune-search-space>`.

.. literalinclude:: doc_code/key_concepts.py
    :language: python
    :start-after: __config_start__
    :end-before: __config_end__

.. _tune_60_seconds_trials:

Tune Trials
-----------

You use :ref:`Tuner.fit <tune-run-ref>` to execute and manage hyperparameter tuning and generate your `trials`.
At a minimum, your ``Tuner`` call takes in a trainable as first argument, and a ``param_space`` dictionary
to define the search space.

The ``Tuner.fit()`` function also provides many features such as :ref:`logging <tune-logging>`,
:ref:`checkpointing <tune-trial-checkpoint>`, and :ref:`early stopping <tune-stopping-ref>`.
In the example, minimizing ``a (x ** 2) + b``, a simple Tune run with a simplistic search space for ``a`` and ``b``
looks like this:

.. literalinclude:: doc_code/key_concepts.py
    :language: python
    :start-after: __run_tunable_start__
    :end-before: __run_tunable_end__

``Tuner.fit`` will generate a couple of hyperparameter configurations from its arguments,
wrapping them into :ref:`Trial objects <trial-docstring>`.

Trials contain a lot of information.
For instance, you can get the hyperparameter configuration using (``trial.config``), the trial ID (``trial.trial_id``),
the trial's resource specification (``resources_per_trial`` or ``trial.placement_group_factory``) and many other values.

By default ``Tuner.fit`` will execute until all trials stop or error.
Here's an example output of a trial run:

.. TODO: how to make sure this doesn't get outdated?
.. code-block:: bash

    == Status ==
    Memory usage on this node: 11.4/16.0 GiB
    Using FIFO scheduling algorithm.
    Resources requested: 1/12 CPUs, 0/0 GPUs, 0.0/3.17 GiB heap, 0.0/1.07 GiB objects
    Result logdir: /Users/foo/ray_results/myexp
    Number of trials: 1 (1 RUNNING)
    +----------------------+----------+---------------------+-----------+--------+--------+----------------+-------+
    | Trial name           | status   | loc                 |         a |      b |  score | total time (s) |  iter |
    |----------------------+----------+---------------------+-----------+--------+--------+----------------+-------|
    | Trainable_a826033a | RUNNING  | 10.234.98.164:31115 | 0.303706  | 0.0761 | 0.1289 |        7.54952 |    15 |
    +----------------------+----------+---------------------+-----------+--------+--------+----------------+-------+


You can also easily run just 10 trials by specifying the number of samples (``num_samples``).
Tune automatically :ref:`determines how many trials will run in parallel <tune-parallelism>`.
Note that instead of the number of samples, you can also specify a time budget in seconds through ``time_budget_s``,
if you set ``num_samples=-1``.

.. literalinclude:: doc_code/key_concepts.py
    :language: python
    :start-after: __run_tunable_samples_start__
    :end-before: __run_tunable_samples_end__


Finally, you can use more interesting search spaces to optimize your hyperparameters
via Tune's :ref:`search space API <tune-default-search-space>`, like using random samples or grid search.
Here's an example of uniformly sampling between ``[0, 1]`` for ``a`` and ``b``:

.. literalinclude:: doc_code/key_concepts.py
    :language: python
    :start-after: __search_space_start__
    :end-before: __search_space_end__

To learn more about the various ways of configuring your Tune runs,
check out the :ref:`Tuner API reference <tune-run-ref>`.

.. _search-alg-ref:

Tune Search Algorithms
----------------------

To optimize the hyperparameters of your training process, you use
a :ref:`Search Algorithm <tune-search-alg>` which suggests hyperparameter configurations.
If you don't specify a search algorithm, Tune will use random search by default, which can provide you
with a good starting point for your hyperparameter optimization.

For instance, to use Tune with simple Bayesian optimization through the ``bayesian-optimization`` package
(make sure to first run ``pip install bayesian-optimization``), we can define an ``algo`` using ``BayesOptSearch``.
Simply pass in a ``search_alg`` argument to ``tune.TuneConfig``, which is taken in by ``Tuner``:

.. literalinclude:: doc_code/key_concepts.py
    :language: python
    :start-after: __bayes_start__
    :end-before: __bayes_end__

Tune has Search Algorithms that integrate with many popular **optimization** libraries,
such as :ref:`HyperOpt <tune-hyperopt>` or :ref:`Optuna <tune-optuna>`.
Tune automatically converts the provided search space into the search
spaces the search algorithms and underlying libraries expect.
See the :ref:`Search Algorithm API documentation <tune-search-alg>` for more details.

Here's an overview of all available search algorithms in Tune:

.. list-table::
   :widths: 5 5 2 10
   :header-rows: 1

   * - SearchAlgorithm
     - Summary
     - Website
     - Code Example
   * - :ref:`Random search/grid search <tune-basicvariant>`
     - Random search/grid search
     -
     - :doc:`/tune/examples/includes/tune_basic_example`
   * - :ref:`AxSearch <tune-ax>`
     - Bayesian/Bandit Optimization
     - [`Ax <https://ax.dev/>`__]
     - :doc:`/tune/examples/includes/ax_example`
   * - :ref:`HyperOptSearch <tune-hyperopt>`
     - Tree-Parzen Estimators
     - [`HyperOpt <http://hyperopt.github.io/hyperopt>`__]
     - :doc:`/tune/examples/hyperopt_example`
   * - :ref:`BayesOptSearch <bayesopt>`
     - Bayesian Optimization
     - [`BayesianOptimization <https://github.com/fmfn/BayesianOptimization>`__]
     - :doc:`/tune/examples/includes/bayesopt_example`
   * - :ref:`TuneBOHB <suggest-TuneBOHB>`
     - Bayesian Opt/HyperBand
     - [`BOHB <https://github.com/automl/HpBandSter>`__]
     - :doc:`/tune/examples/includes/bohb_example`
   * - :ref:`NevergradSearch <nevergrad>`
     - Gradient-free Optimization
     - [`Nevergrad <https://github.com/facebookresearch/nevergrad>`__]
     - :doc:`/tune/examples/includes/nevergrad_example`
   * - :ref:`OptunaSearch <tune-optuna>`
     - Optuna search algorithms
     - [`Optuna <https://optuna.org/>`__]
     - :doc:`/tune/examples/optuna_example`

.. note:: Unlike :ref:`Tune's Trial Schedulers <tune-schedulers>`,
    Tune Search Algorithms cannot affect or stop training processes.
    However, you can use them together to early stop the evaluation of bad trials.

In case you want to implement your own search algorithm, the interface is easy to implement,
you can :ref:`read the instructions here <byo-algo>`.

Tune also provides helpful utilities to use with Search Algorithms:

 * :ref:`repeater`: Support for running each *sampled hyperparameter* with multiple random seeds.
 * :ref:`limiter`: Limits the amount of concurrent trials when running optimization.
 * :ref:`shim`: Allows creation of the search algorithm object given a string.

Note that in the example above we  tell Tune to ``stop`` after ``20`` training iterations.
This way of stopping trials with explicit rules is useful, but in many cases we can do even better with
`schedulers`.

.. _schedulers-ref:

Tune Schedulers
---------------

To make your training process more efficient, you can use a :ref:`Trial Scheduler <tune-schedulers>`.
For instance, in our ``trainable`` example minimizing a function in a training loop, we used ``session.report()``.
This reported `incremental` results, given a hyperparameter configuration selected by a search algorithm.
Based on these reported results, a Tune scheduler can decide whether to stop the trial early or not.
If you don't specify a scheduler, Tune will use a first-in-first-out (FIFO) scheduler by default, which simply
passes through the trials selected by your search algorithm in the order they were picked and does not perform any early stopping.

In short, schedulers can stop, pause, or tweak the
hyperparameters of running trials, potentially making your hyperparameter tuning process much faster.
Unlike search algorithms, :ref:`Trial Scheduler <tune-schedulers>` do not select which hyperparameter
configurations to evaluate.

Here's a quick example of using the so-called ``HyperBand`` scheduler to tune an experiment.
All schedulers take in a ``metric``, which is the value reported by your trainable.
The ``metric`` is then maximized or minimized according to the ``mode`` you provide.
To use a scheduler, just pass in a ``scheduler`` argument to ``tune.TuneConfig``, which is taken in by ``Tuner``:

.. literalinclude:: doc_code/key_concepts.py
    :language: python
    :start-after: __hyperband_start__
    :end-before: __hyperband_end__


Tune includes distributed implementations of early stopping algorithms such as
`Median Stopping Rule <https://research.google.com/pubs/pub46180.html>`__, `HyperBand <https://arxiv.org/abs/1603.06560>`__,
and `ASHA <https://openreview.net/forum?id=S1Y7OOlRZ>`__.
Tune also includes a distributed implementation of `Population Based Training (PBT) <https://www.deepmind.com/blog/population-based-training-of-neural-networks>`__
and `Population Based Bandits (PB2) <https://arxiv.org/abs/2002.02518>`__.

.. tip:: The easiest scheduler to start with is the ``ASHAScheduler`` which will aggressively terminate low-performing trials.

When using schedulers, you may face compatibility issues, as shown in the below compatibility matrix.
Certain schedulers cannot be used with search algorithms,
and certain schedulers require that you implement :ref:`checkpointing <tune-trial-checkpoint>`.

Schedulers can dynamically change trial resource requirements during tuning.
This is implemented in :ref:`ResourceChangingScheduler<tune-resource-changing-scheduler>`,
which can wrap around any other scheduler.

.. list-table:: Scheduler Compatibility Matrix
   :header-rows: 1

   * - Scheduler
     - Need Checkpointing?
     - SearchAlg Compatible?
     - Example
   * - :ref:`ASHA <tune-scheduler-hyperband>`
     - No
     - Yes
     - :doc:`Link </tune/examples/includes/async_hyperband_example>`
   * - :ref:`Median Stopping Rule <tune-scheduler-msr>`
     - No
     - Yes
     - :ref:`Link <tune-scheduler-msr>`
   * - :ref:`HyperBand <tune-original-hyperband>`
     - Yes
     - Yes
     - :doc:`Link </tune/examples/includes/hyperband_example>`
   * - :ref:`BOHB <tune-scheduler-bohb>`
     - Yes
     - Only TuneBOHB
     - :doc:`Link </tune/examples/includes/bohb_example>`
   * - :ref:`Population Based Training <tune-scheduler-pbt>`
     - Yes
     - Not Compatible
     - :doc:`Link </tune/examples/includes/pbt_function>`
   * - :ref:`Population Based Bandits <tune-scheduler-pb2>`
     - Yes
     - Not Compatible
     - :doc:`Basic Example </tune/examples/includes/pb2_example>`, :doc:`PPO example </tune/examples/includes/pb2_ppo_example>`

Learn more about trial schedulers in :ref:`the scheduler API documentation <schedulers-ref>`.

.. _tune-concepts-analysis:

Tune ResultGrid
---------------

``Tuner.fit()`` returns an :ref:`ResultGrid <tune-analysis-docs>` object which has methods you can use for
analyzing your training.
The following example shows you how to access various metrics from an ``ResultGrid`` object, like the best available
trial, or the best hyperparameter configuration for that trial:

.. literalinclude:: doc_code/key_concepts.py
    :language: python
    :start-after: __analysis_start__
    :end-before: __analysis_end__

This object can also retrieve all training runs as dataframes,
allowing you to do ad-hoc data analysis over your results.

.. literalinclude:: doc_code/key_concepts.py
    :language: python
    :start-after: __results_start__
    :end-before: __results_end__

See the :ref:`result analysis user guide <tune-analysis-guide>` for more usage examples.

What's Next?
-------------

Now that you have a working understanding of Tune, check out:

* :ref:`tune-guides`: Tutorials for using Tune with your preferred machine learning library.
* :doc:`/tune/examples/index`: End-to-end examples and templates for using Tune with your preferred machine learning library.
* :doc:`/tune/getting-started`: A simple tutorial that walks you through the process of setting up a Tune experiment.


Further Questions or Issues?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: /_includes/_help.rst


.. _tune-search-space-tutorial:

Working with Tune Search Spaces
===============================

Tune has a native interface for specifying search spaces.
You can specify the search space via ``Tuner(param_space=...)``.

Thereby, you can either use the ``tune.grid_search`` primitive to use grid search:

.. code-block:: python

    tuner = tune.Tuner(
        trainable,
        param_space={"bar": tune.grid_search([True, False])})
    results = tuner.fit()


Or you can use one of the random sampling primitives to specify distributions (:doc:`/tune/api/search_space`):

.. code-block:: python

    tuner = tune.Tuner(
        trainable,
        param_space={
            "param1": tune.choice([True, False]),
            "bar": tune.uniform(0, 10),
            "alpha": tune.sample_from(lambda _: np.random.uniform(100) ** 2),
            "const": "hello"  # It is also ok to specify constant values.
        })
    results = tuner.fit()

.. caution:: If you use a SearchAlgorithm, you may not be able to specify lambdas or grid search with this
    interface, as some search algorithms may not be compatible.


To sample multiple times/run multiple trials, specify ``tune.RunConfig(num_samples=N``.
If ``grid_search`` is provided as an argument, the *same* grid will be repeated ``N`` times.

.. code-block:: python

    # 13 different configs.
    tuner = tune.Tuner(trainable, tune_config=tune.TuneConfig(num_samples=13), param_space={
        "x": tune.choice([0, 1, 2]),
        }
    )
    tuner.fit()

    # 13 different configs.
    tuner = tune.Tuner(trainable, tune_config=tune.TuneConfig(num_samples=13), param_space={
        "x": tune.choice([0, 1, 2]),
        "y": tune.randn([0, 1, 2]),
        }
    )
    tuner.fit()

    # 4 different configs.
    tuner = tune.Tuner(trainable, tune_config=tune.TuneConfig(num_samples=1), param_space={"x": tune.grid_search([1, 2, 3, 4])})
    tuner.fit()

    # 3 different configs.
    tuner = tune.Tuner(trainable, tune_config=tune.TuneConfig(num_samples=1), param_space={"x": grid_search([1, 2, 3])})
    tuner.fit()

    # 6 different configs.
    tuner = tune.Tuner(trainable, tune_config=tune.TuneConfig(num_samples=2), param_space={"x": tune.grid_search([1, 2, 3])})
    tuner.fit()

    # 9 different configs.
    tuner = tune.Tuner(trainable, tune_config=tune.TuneConfig(num_samples=1), param_space={
        "x": tune.grid_search([1, 2, 3]),
        "y": tune.grid_search([a, b, c])}
    )
    tuner.fit()

    # 18 different configs.
    tuner = tune.Tuner(trainable, tune_config=tune.TuneConfig(num_samples=2), param_space={
        "x": tune.grid_search([1, 2, 3]),
        "y": tune.grid_search([a, b, c])}
    )
    tuner.fit()

    # 45 different configs.
    tuner = tune.Tuner(trainable, tune_config=tune.TuneConfig(num_samples=5), param_space={
        "x": tune.grid_search([1, 2, 3]),
        "y": tune.grid_search([a, b, c])}
    )
    tuner.fit()



Note that grid search and random search primitives are inter-operable.
Each can be used independently or in combination with each other.

.. code-block:: python

    # 6 different configs.
    tuner = tune.Tuner(trainable, tune_config=tune.TuneConfig(num_samples=2), param_space={
        "x": tune.sample_from(...),
        "y": tune.grid_search([a, b, c])
        }
    )
    tuner.fit()

In the below example, ``num_samples=10`` repeats the 3x3 grid search 10 times,
for a total of 90 trials, each with randomly sampled values of ``alpha`` and ``beta``.

.. code-block:: python
   :emphasize-lines: 12

    tuner = tune.Tuner(
        my_trainable,
        run_config=RunConfig(name="my_trainable"),
        # num_samples will repeat the entire config 10 times.
        tune_config=tune.TuneConfig(num_samples=10),
        param_space={
            # ``sample_from`` creates a generator to call the lambda once per trial.
            "alpha": tune.sample_from(lambda spec: np.random.uniform(100)),
            # ``sample_from`` also supports "conditional search spaces"
            "beta": tune.sample_from(lambda spec: spec.config.alpha * np.random.normal()),
            "nn_layers": [
                # tune.grid_search will make it so that all values are evaluated.
                tune.grid_search([16, 64, 256]),
                tune.grid_search([16, 64, 256]),
            ],
        },
    )
    tuner.fit()

.. tip::

    Avoid passing large objects as values in the search space, as that will incur a performance overhead.
    Use :func:`tune.with_parameters <ray.tune.with_parameters>` to pass large objects in or load them inside your trainable
    from disk (making sure that all nodes have access to the files) or cloud storage.
    See :ref:`tune-bottlenecks` for more information.

Note that when using Ray Train with Ray Tune, certain config objects can also be included
as part of the search space, thereby allowing you to tune things like number of workers for a trainer.

.. _tune_custom-search:

How to use Custom and Conditional Search Spaces in Tune?
--------------------------------------------------------

You'll often run into awkward search spaces (i.e., when one hyperparameter depends on another).
Use ``tune.sample_from(func)`` to provide a **custom** callable function for generating a search space.

The parameter ``func`` should take in a ``spec`` object, which has a ``config`` namespace
from which you can access other hyperparameters.
This is useful for conditional distributions:

.. code-block:: python

    tuner = tune.Tuner(
        ...,
        param_space={
            # A random function
            "alpha": tune.sample_from(lambda _: np.random.uniform(100)),
            # Use the `spec.config` namespace to access other hyperparameters
            "beta": tune.sample_from(lambda spec: spec.config.alpha * np.random.normal())
        }
    )
    tuner.fit()

Here's an example showing a grid search over two nested parameters combined with random sampling from
two lambda functions, generating 9 different trials.
Note that the value of ``beta`` depends on the value of ``alpha``,
which is represented by referencing ``spec.config.alpha`` in the lambda function.
This lets you specify conditional parameter distributions.

.. code-block:: python
   :emphasize-lines: 4-11

    tuner = tune.Tuner(
        my_trainable,
        run_config=RunConfig(name="my_trainable"),
        param_space={
            "alpha": tune.sample_from(lambda spec: np.random.uniform(100)),
            "beta": tune.sample_from(lambda spec: spec.config.alpha * np.random.normal()),
            "nn_layers": [
                tune.grid_search([16, 64, 256]),
                tune.grid_search([16, 64, 256]),
            ],
        }
    )

.. note::

    This format is not supported by every SearchAlgorithm, and only some SearchAlgorithms, like :ref:`HyperOpt <tune-hyperopt>`
    and :ref:`Optuna <tune-optuna>`, handle conditional search spaces at all.

    In order to use conditional search spaces with :ref:`HyperOpt <tune-hyperopt>`,
    a `Hyperopt search space <http://hyperopt.github.io/hyperopt/getting-started/search_spaces/>`_ isnecessary.
    :ref:`Optuna <tune-optuna>` supports conditional search spaces through its define-by-run
    interface (:doc:`/tune/examples/optuna_example`).


.. _tune-storage-options:

How to Configure Persistent Storage in Ray Tune
===============================================

.. seealso::

    Before diving into storage options, one can take a look at
    :ref:`the different types of data stored by Tune <tune-persisted-experiment-data>`.

Tune allows you to configure persistent storage options to enable following use cases in a distributed Ray cluster:

- **Trial-level fault tolerance**: When trials are restored (e.g. after a node failure or when the experiment was paused),
  they may be scheduled on different nodes, but still would need access to their latest checkpoint.
- **Experiment-level fault tolerance**: For an entire experiment to be restored (e.g. if the cluster crashes unexpectedly),
  Tune needs to be able to access the latest experiment state, along with all trial
  checkpoints to start from where the experiment left off.
- **Post-experiment analysis**: A consolidated location storing data from all trials is useful for post-experiment analysis
  such as accessing the best checkpoints and hyperparameter configs after the cluster has already been terminated.
- **Bridge with downstream serving/batch inference tasks**: With a configured storage, you can easily access the models
  and artifacts generated by trials, share them with others or use them in downstream tasks.


Storage Options in Tune
-----------------------

Tune provides support for three scenarios:

1. When using cloud storage (e.g. AWS S3 or Google Cloud Storage) accessible by all machines in the cluster.
2. When using a network filesystem (NFS) mounted to all machines in the cluster.
3. When running Tune on a single node and using the local filesystem as the persistent storage location.

.. note::

    A network filesystem or cloud storage can be configured for single-node
    experiments. This can be useful to persist your experiment results in external storage
    if, for example, the instance you run your experiment on clears its local storage
    after termination.

.. seealso::

    See :class:`~ray.train.SyncConfig` for the full set of configuration options as well as more details.


.. _tune-cloud-checkpointing:

Configuring Tune with cloud storage (AWS S3, Google Cloud Storage)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If all nodes in a Ray cluster have access to cloud storage, e.g. AWS S3 or Google Cloud Storage (GCS),
then all experiment outputs can be saved in a shared cloud bucket.

We can configure cloud storage by telling Ray Tune to **upload to a remote** ``storage_path``:

.. code-block:: python

    from ray import tune
    from ray.train import RunConfig

    tuner = tune.Tuner(
        trainable,
        run_config=RunConfig(
            name="experiment_name",
            storage_path="s3://bucket-name/sub-path/",
        )
    )
    tuner.fit()

In this example, all experiment results can be found in the shared storage at ``s3://bucket-name/sub-path/experiment_name`` for further processing.

.. note::

    The head node will not have access to all experiment results locally. If you want to process
    e.g. the best checkpoint further, you will first have to fetch it from the cloud storage.

    Experiment restoration should also be done using the experiment directory at the cloud storage
    URI, rather than the local experiment directory on the head node. See :ref:`here for an example <tune-syncing-restore-from-uri>`.



Configuring Tune with a network filesystem (NFS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If all Ray nodes have access to a network filesystem, e.g. AWS EFS or Google Cloud Filestore,
they can all write experiment outputs to this directory.

All we need to do is **set the shared network filesystem as the path to save results**.

.. code-block:: python

    from ray import train, tune

    tuner = tune.Tuner(
        trainable,
        run_config=train.RunConfig(
            name="experiment_name",
            storage_path="/mnt/path/to/shared/storage/",
        )
    )
    tuner.fit()

In this example, all experiment results can be found in the shared storage at ``/path/to/shared/storage/experiment_name`` for further processing.


.. _tune-default-syncing:

Configure Tune without external persistent storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a single-node cluster
************************

If you're just running an experiment on a single node (e.g., on a laptop), Tune will use the
local filesystem as the default storage location for checkpoints and other artifacts.
Results are saved to ``~/ray_results`` in a sub-directory with a unique auto-generated name by default,
unless you customize this with ``storage_path`` and ``name`` in :class:`~ray.train.RunConfig`.

.. code-block:: python

    from ray import tune
    from ray.train import RunConfig

    tuner = tune.Tuner(
        trainable,
        run_config=RunConfig(
            storage_path="/tmp/custom/storage/path",
            name="experiment_name",
        )
    )
    tuner.fit()

In this example, all experiment results can found locally at ``/tmp/custom/storage/path/experiment_name`` for further processing.


On a multi-node cluster (Deprecated)
************************************

.. warning::

    When running on multiple nodes, using the local filesystem of the head node as the persistent storage location is *deprecated*.
    If you save trial checkpoints and run on a multi-node cluster, Tune will raise an error by default, if NFS or cloud storage is not setup.
    See `this issue <https://github.com/ray-project/ray/issues/37177>`_ for more information.


Examples
--------

Let's show some examples of configuring storage location and synchronization options.
We'll also show how to resume the experiment for each of the examples, in the case that your experiment gets interrupted.
See :ref:`tune-fault-tolerance-ref` for more information on resuming experiments.

In each example, we'll give a practical explanation of how *trial checkpoints* are saved
across the cluster and the external storage location (if one is provided).
See :ref:`tune-persisted-experiment-data` for an overview of other experiment data that Tune needs to persist.


Example: Running Tune with cloud storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's assume that you're running this example script from your Ray cluster's head node.

In the example below, ``my_trainable`` is a Tune :ref:`trainable <trainable-docs>`
that implements saving and loading checkpoints.

.. code-block:: python

    import os
    import ray
    from ray import train, tune
    from your_module import my_trainable


    tuner = tune.Tuner(
        my_trainable,
        run_config=train.RunConfig(
            # Name of your experiment
            name="my-tune-exp",
            # Configure how experiment data and checkpoints are persisted.
            # We recommend cloud storage checkpointing as it survives the cluster when
            # instances are terminated and has better performance.
            storage_path="s3://my-checkpoints-bucket/path/",
            checkpoint_config=train.CheckpointConfig(
                # We'll keep the best five checkpoints at all times
                # (with the highest AUC scores, a metric reported by the trainable)
                checkpoint_score_attribute="max-auc",
                checkpoint_score_order="max",
                num_to_keep=5,
            ),
        ),
    )
    # This starts the run!
    results = tuner.fit()

In this example, trial checkpoints will be saved to: ``s3://my-checkpoints-bucket/path/my-tune-exp/<trial_name>/checkpoint_<step>``

.. _tune-syncing-restore-from-uri:

If this run stopped for any reason (ex: user CTRL+C, terminated due to out of memory issues),
you can resume it any time starting from the experiment state saved in the cloud:

.. code-block:: python

    from ray import tune
    tuner = tune.Tuner.restore(
        "s3://my-checkpoints-bucket/path/my-tune-exp",
        trainable=my_trainable,
        resume_errored=True,
    )
    tuner.fit()


There are a few options for restoring an experiment:
``resume_unfinished``, ``resume_errored`` and ``restart_errored``.
Please see the documentation of
:meth:`Tuner.restore() <ray.tune.tuner.Tuner.restore>` for more details.


Advanced configuration
----------------------

See :ref:`Ray Train's section on advanced storage configuration <train-storage-advanced>`.
All of the configurations also apply to Ray Tune.

Running Basic Tune Experiments
==============================

The most common way to use Tune is also the simplest: as a parallel experiment runner. If you can define experiment trials in a Python function, you can use Tune to run hundreds to thousands of independent trial instances in a cluster. Tune manages trial execution, status reporting, and fault tolerance.

Running Independent Tune Trials in Parallel
-------------------------------------------

As a general example, let's consider executing ``N`` independent model training trials using Tune as a simple grid sweep. Each trial can execute different code depending on a passed-in config dictionary.

**Step 1:** First, we define the model training function that we want to run variations of. The function takes in a config dictionary as argument, and returns a simple dict output. Learn more about logging Tune results at :ref:`tune-logging`.

.. literalinclude:: ../doc_code/tune.py
    :language: python
    :start-after: __step1_begin__
    :end-before: __step1_end__

**Step 2:** Next, define the space of trials to run. Here, we define a simple grid sweep from ``0..NUM_MODELS``, which will generate the config dicts to be passed to each model function. Learn more about what features Tune offers for defining spaces at :ref:`tune-search-space-tutorial`.

.. literalinclude:: ../doc_code/tune.py
    :language: python
    :start-after: __step2_begin__
    :end-before: __step2_end__

**Step 3:** Optionally, configure the resources allocated per trial. Tune uses this resources allocation to control the parallelism. For example, if each trial was configured to use 4 CPUs, and the cluster had only 32 CPUs, then Tune will limit the number of concurrent trials to 8 to avoid overloading the cluster. For more information, see :ref:`tune-parallelism`.

.. literalinclude:: ../doc_code/tune.py
    :language: python
    :start-after: __step3_begin__
    :end-before: __step3_end__

**Step 4:** Run the trial with Tune. Tune will report on experiment status, and after the experiment finishes, you can inspect the results. Tune can retry failed trials automatically, as well as entire experiments; see :ref:`tune-stopping-guide`.

.. literalinclude:: ../doc_code/tune.py
    :language: python
    :start-after: __step4_begin__
    :end-before: __step4_end__

**Step 5:** Inspect results. They will look something like this. Tune periodically prints a status summary to stdout showing the ongoing experiment status, until it finishes:

.. code::

    == Status ==
    Current time: 2022-09-21 10:19:34 (running for 00:00:04.54)
    Memory usage on this node: 6.9/31.1 GiB
    Using FIFO scheduling algorithm.
    Resources requested: 0/8 CPUs, 0/0 GPUs, 0.0/16.13 GiB heap, 0.0/8.06 GiB objects
    Result logdir: /home/ubuntu/ray_results/train_model_2022-09-21_10-19-26
    Number of trials: 100/100 (100 TERMINATED)
    +-------------------------+------------+----------------------+------------+--------+------------------+
    | Trial name              | status     | loc                  | model_id   |   iter |   total time (s) |
    |-------------------------+------------+----------------------+------------+--------+------------------|
    | train_model_8d627_00000 | TERMINATED | 192.168.1.67:2381731 | model_0    |      1 |      8.46386e-05 |
    | train_model_8d627_00001 | TERMINATED | 192.168.1.67:2381761 | model_1    |      1 |      0.000126362 |
    | train_model_8d627_00002 | TERMINATED | 192.168.1.67:2381763 | model_2    |      1 |      0.000112772 |
    ...
    | train_model_8d627_00097 | TERMINATED | 192.168.1.67:2381731 | model_97   |      1 |      5.57899e-05 |
    | train_model_8d627_00098 | TERMINATED | 192.168.1.67:2381767 | model_98   |      1 |      6.05583e-05 |
    | train_model_8d627_00099 | TERMINATED | 192.168.1.67:2381763 | model_99   |      1 |      6.69956e-05 |
    +-------------------------+------------+----------------------+------------+--------+------------------+

    2022-09-21 10:19:35,159	INFO tune.py:762 -- Total run time: 5.06 seconds (4.46 seconds for the tuning loop).

The final result objects contain finished trial metadata:

.. code::

    Result(metrics={'score': 'model_0', 'other_data': Ellipsis, 'done': True, 'trial_id': '8d627_00000', 'experiment_tag': '0_model_id=model_0'}, error=None, log_dir=PosixPath('/home/ubuntu/ray_results/train_model_2022-09-21_10-19-26/train_model_8d627_00000_0_model_id=model_0_2022-09-21_10-19-30'))
    Result(metrics={'score': 'model_1', 'other_data': Ellipsis, 'done': True, 'trial_id': '8d627_00001', 'experiment_tag': '1_model_id=model_1'}, error=None, log_dir=PosixPath('/home/ubuntu/ray_results/train_model_2022-09-21_10-19-26/train_model_8d627_00001_1_model_id=model_1_2022-09-21_10-19-31'))
    Result(metrics={'score': 'model_2', 'other_data': Ellipsis, 'done': True, 'trial_id': '8d627_00002', 'experiment_tag': '2_model_id=model_2'}, error=None, log_dir=PosixPath('/home/ubuntu/ray_results/train_model_2022-09-21_10-19-26/train_model_8d627_00002_2_model_id=model_2_2022-09-21_10-19-31'))

How does Tune compare  to using Ray Core (``ray.remote``)?
----------------------------------------------------------

You might be wondering how Tune differs from simply using :ref:`ray-remote-functions` for parallel trial execution. Indeed, the above example could be re-written similarly as:

.. literalinclude:: ../doc_code/tune.py
    :language: python
    :start-after: __tasks_begin__
    :end-before: __tasks_end__

Compared to using Ray tasks, Tune offers the following additional functionality:

* Status reporting and tracking, including integrations and callbacks to common monitoring tools.
* Checkpointing of trials for fine-grained fault-tolerance.
* Gang scheduling of multi-worker trials.

In short, consider using Tune if you need status tracking or support for more advanced ML workloads.


.. _tune-distributed-ref:

Running Distributed Experiments with Ray Tune
==============================================

Tune is commonly used for large-scale distributed hyperparameter optimization. This page will overview how to setup and launch a distributed experiment along with :ref:`commonly used commands <tune-distributed-common>` for Tune when running distributed experiments.

.. contents::
    :local:
    :backlinks: none

Summary
-------

To run a distributed experiment with Tune, you need to:

1. First, :ref:`start a Ray cluster <cluster-index>` if you have not already.
2. Run the script on the head node, or use :ref:`ray submit <ray-submit-doc>`, or use :ref:`Ray Job Submission <jobs-overview>`.

.. tune-distributed-cloud:

Example: Distributed Tune on AWS VMs
------------------------------------

Follow the instructions below to launch nodes on AWS (using the Deep Learning AMI). See the :ref:`cluster setup documentation <cluster-index>`. Save the below cluster configuration (``tune-default.yaml``):

.. literalinclude:: /../../python/ray/tune/examples/tune-default.yaml
   :language: yaml
   :name: tune-default.yaml

``ray up`` starts Ray on the cluster of nodes.

.. code-block:: bash

    ray up tune-default.yaml

``ray submit --start`` starts a cluster as specified by the given cluster configuration YAML file, uploads ``tune_script.py`` to the cluster, and runs ``python tune_script.py [args]``.

.. code-block:: bash

    ray submit tune-default.yaml tune_script.py --start -- --ray-address=localhost:6379

.. image:: /images/tune-upload.png
    :scale: 50%
    :align: center

Analyze your results on TensorBoard by starting TensorBoard on the remote head machine.

.. code-block:: bash

    # Go to http://localhost:6006 to access TensorBoard.
    ray exec tune-default.yaml 'tensorboard --logdir=~/ray_results/ --port 6006' --port-forward 6006


Note that you can customize the directory of results by specifying: ``RunConfig(storage_path=..)``, taken in by ``Tuner``. You can then point TensorBoard to that directory to visualize results. You can also use `awless <https://github.com/wallix/awless>`_ for easy cluster management on AWS.


Running a Distributed Tune Experiment
-------------------------------------

Running a distributed (multi-node) experiment requires Ray to be started already.
You can do this on local machines or on the cloud.

Across your machines, Tune will automatically detect the number of GPUs and CPUs without you needing to manage ``CUDA_VISIBLE_DEVICES``.

To execute a distributed experiment, call ``ray.init(address=XXX)`` before ``Tuner.fit()``, where ``XXX`` is the Ray address, which defaults to ``localhost:6379``. The Tune python script should be executed only on the head node of the Ray cluster.

One common approach to modifying an existing Tune experiment to go distributed is to set an ``argparse`` variable so that toggling between distributed and single-node is seamless.

.. code-block:: python

    import ray
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--address")
    args = parser.parse_args()
    ray.init(address=args.address)

    tuner = tune.Tuner(...)
    tuner.fit()

.. code-block:: bash

    # On the head node, connect to an existing ray cluster
    $ python tune_script.py --ray-address=localhost:XXXX

If you used a cluster configuration (starting a cluster with ``ray up`` or ``ray submit --start``), use:

.. code-block:: bash

    ray submit tune-default.yaml tune_script.py -- --ray-address=localhost:6379

.. tip::

    1. In the examples, the Ray address commonly used is ``localhost:6379``.
    2. If the Ray cluster is already started, you should not need to run anything on the worker nodes.


Storage Options in a Distributed Tune Run
-----------------------------------------

In a distributed experiment, you should try to use :ref:`cloud checkpointing <tune-cloud-checkpointing>` to
reduce synchronization overhead. For this, you just have to specify a remote ``storage_path`` in the
:class:`RunConfig <ray.train.RunConfig>`.

`my_trainable` is a user-defined :ref:`Tune Trainable <tune_60_seconds_trainables>` in the following example:

.. code-block:: python

    from ray import train, tune
    from my_module import my_trainable

    tuner = tune.Tuner(
        my_trainable,
        run_config=train.RunConfig(
            name="experiment_name",
            storage_path="s3://bucket-name/sub-path/",
        )
    )
    tuner.fit()

For more details or customization, see our
:ref:`guide on configuring storage in a distributed Tune experiment <tune-storage-options>`.


.. _tune-distributed-spot:

Tune Runs on preemptible instances
-----------------------------------

Running on spot instances (or preemptible instances) can reduce the cost of your experiment.
You can enable spot instances in AWS via the following configuration modification:

.. code-block:: yaml

    # Provider-specific config for worker nodes, e.g. instance type.
    worker_nodes:
        InstanceType: m5.large
        ImageId: ami-0b294f219d14e6a82 # Deep Learning AMI (Ubuntu) Version 21.0

        # Run workers on spot by default. Comment this out to use on-demand.
        InstanceMarketOptions:
            MarketType: spot
            SpotOptions:
                MaxPrice: 1.0  # Max Hourly Price

In GCP, you can use the following configuration modification:

.. code-block:: yaml

    worker_nodes:
        machineType: n1-standard-2
        disks:
          - boot: true
            autoDelete: true
            type: PERSISTENT
            initializeParams:
              diskSizeGb: 50
              # See https://cloud.google.com/compute/docs/images for more images
              sourceImage: projects/deeplearning-platform-release/global/images/family/tf-1-13-cpu

        # Run workers on preemtible instances.
        scheduling:
          - preemptible: true

Spot instances may be pre-empted suddenly while trials are still running.
Tune allows you to mitigate the effects of this by preserving the progress of your model training through
:ref:`checkpointing <tune-trial-checkpoint>`.

.. literalinclude:: /../../python/ray/tune/tests/tutorial.py
    :language: python
    :start-after: __trainable_run_begin__
    :end-before: __trainable_run_end__


Example for Using Tune with Spot instances (AWS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is an example for running Tune on spot instances. This assumes your AWS credentials have already been setup (``aws configure``):

1. Download a full example Tune experiment script here. This includes a Trainable with checkpointing: :download:`mnist_pytorch_trainable.py </../../python/ray/tune/examples/mnist_pytorch_trainable.py>`. To run this example, you will need to install the following:

.. code-block:: bash

    $ pip install ray torch torchvision filelock

2. Download an example cluster yaml here: :download:`tune-default.yaml </../../python/ray/tune/examples/tune-default.yaml>`
3. Run ``ray submit`` as below to run Tune across them. Append ``[--start]`` if the cluster is not up yet. Append ``[--stop]`` to automatically shutdown your nodes after running.

.. code-block:: bash

    ray submit tune-default.yaml mnist_pytorch_trainable.py --start -- --ray-address=localhost:6379


4. Optionally for testing on AWS or GCP, you can use the following to kill a random worker node after all the worker nodes are up

.. code-block:: bash

    $ ray kill-random-node tune-default.yaml --hard

To summarize, here are the commands to run:

.. code-block:: bash

    wget https://raw.githubusercontent.com/ray-project/ray/master/python/ray/tune/examples/mnist_pytorch_trainable.py
    wget https://raw.githubusercontent.com/ray-project/ray/master/python/ray/tune/tune-default.yaml
    ray submit tune-default.yaml mnist_pytorch_trainable.py --start -- --ray-address=localhost:6379

    # wait a while until after all nodes have started
    ray kill-random-node tune-default.yaml --hard

You should see Tune eventually continue the trials on a different worker node. See the :ref:`Fault Tolerance <tune-fault-tol>` section for more details.

You can also specify ``storage_path=...``, as part of ``RunConfig``, which is taken in by ``Tuner``, to upload results to cloud storage like S3, allowing you to persist results in case you want to start and stop your cluster automatically.

.. _tune-fault-tol:

Fault Tolerance of Tune Runs
----------------------------

Tune automatically restarts trials in the case of trial failures (if ``max_failures != 0``),
both in the single node and distributed setting.

For example, let's say a node is pre-empted or crashes while a trial is still executing on that node.
Assuming that a checkpoint for this trial exists (and in the distributed setting,
:ref:`some form of persistent storage is configured to access the trial's checkpoint <tune-storage-options>`),
Tune waits until available resources are available to begin executing the trial again from where it left off.
If no checkpoint is found, the trial will restart from scratch.
See :ref:`here for information on checkpointing <tune-trial-checkpoint>`.


If the trial or actor is then placed on a different node, Tune automatically pushes the previous checkpoint file
to that node and restores the remote trial actor state, allowing the trial to resume from the latest checkpoint
even after failure.

Recovering From Failures
~~~~~~~~~~~~~~~~~~~~~~~~

Tune automatically persists the progress of your entire experiment (a ``Tuner.fit()`` session), so if an experiment crashes or is otherwise cancelled, it can be resumed through :meth:`Tuner.restore() <ray.tune.tuner.Tuner.restore>`.

.. _tune-distributed-common:

Common Tune Commands
--------------------

Below are some commonly used commands for submitting experiments. Please see the :ref:`Clusters page <cluster-index>` to see find more comprehensive documentation of commands.

.. code-block:: bash

    # Upload `tune_experiment.py` from your local machine onto the cluster. Then,
    # run `python tune_experiment.py --address=localhost:6379` on the remote machine.
    $ ray submit CLUSTER.YAML tune_experiment.py -- --address=localhost:6379

    # Start a cluster and run an experiment in a detached tmux session,
    # and shut down the cluster as soon as the experiment completes.
    # In `tune_experiment.py`, set `RunConfig(storage_path="s3://...")`
    # to persist results
    $ ray submit CLUSTER.YAML --tmux --start --stop tune_experiment.py -- --address=localhost:6379

    # To start or update your cluster:
    $ ray up CLUSTER.YAML [-y]

    # Shut-down all instances of your cluster:
    $ ray down CLUSTER.YAML [-y]

    # Run TensorBoard and forward the port to your own machine.
    $ ray exec CLUSTER.YAML 'tensorboard --logdir ~/ray_results/ --port 6006' --port-forward 6006

    # Run Jupyter Lab and forward the port to your own machine.
    $ ray exec CLUSTER.YAML 'jupyter lab --port 6006' --port-forward 6006

    # Get a summary of all the experiments and trials that have executed so far.
    $ ray exec CLUSTER.YAML 'tune ls ~/ray_results'

    # Upload and sync file_mounts up to the cluster with this command.
    $ ray rsync-up CLUSTER.YAML

    # Download the results directory from your cluster head node to your local machine on ``~/cluster_results``.
    $ ray rsync-down CLUSTER.YAML '~/ray_results' ~/cluster_results

    # Launching multiple clusters using the same configuration.
    $ ray up CLUSTER.YAML -n="cluster1"
    $ ray up CLUSTER.YAML -n="cluster2"
    $ ray up CLUSTER.YAML -n="cluster3"

Troubleshooting
---------------

Sometimes, your program may freeze.
Run this to restart the Ray cluster without running any of the installation commands.

.. code-block:: bash

    $ ray up CLUSTER.YAML --restart-only


.. _tune-guides:

===========
User Guides
===========

.. toctree::
    :hidden:

    Running Basic Experiments <tune-run>
    tune-output
    Setting Trial Resources <tune-resources>
    Using Search Spaces <tune-search-spaces>
    tune-stopping
    tune-trial-checkpoints
    tune-storage
    tune-fault-tolerance
    Using Callbacks and Metrics <tune-metrics>
    tune_get_data_in_and_out
    ../examples/tune_analyze_results
    ../examples/pbt_guide
    Deploying Tune in the Cloud <tune-distributed>
    Tune Architecture <tune-lifecycle>
    Scalability Benchmarks <tune-scalability>


.. tip:: We'd love to hear your feedback on using Tune - `get in touch <https://forms.gle/PTRvGLbKRdUfuzQo9>`_!

In this section, you can find material on how to use Tune and its various features.
You can follow our :ref:`Tune Feature Guides <tune-feature-guides>`, but can also  look into our
:ref:`Practical Examples <tune-recipes>`, or go through some :doc:`Exercises <../examples/exercises>` to get started.

.. _tune-feature-guides:

Tune Feature Guides
-------------------


.. grid:: 1 2 3 4
    :gutter: 1
    :class-container: container pb-3

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-run

            Running Basic Experiments

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-output

            Logging Tune Runs

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-resources

            Setting Trial Resources

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-search-space-tutorial

            Using Search Spaces

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-stopping

            How to Define Stopping Criteria for a Ray Tune Experiment

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-trial-checkpoints

            How to Save and Load Trial Checkpoints

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-storage

            How to Configure Storage Options for a Distributed Tune Experiment

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-fault-tolerance

            How to Enable Fault Tolerance in Ray Tune

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-metrics

            Using Callbacks and Metrics

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: ../tutorials/tune_get_data_in_and_out

            Getting Data in and out of Tune

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-analysis-guide

            Analyzing Tune Experiment Results

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: ../examples/pbt_guide

            A Guide to Population-Based Training

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-distributed

            Deploying Tune in the Cloud

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-lifecycle

            Tune Architecture

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-scalability

            Scalability Benchmarks


.. _tune-trial-checkpoint:

How to Save and Load Trial Checkpoints
======================================

Trial checkpoints are one of :ref:`the three types of data stored by Tune <tune-persisted-experiment-data>`.
These are user-defined and are meant to snapshot your training progress!

Trial-level checkpoints are saved via the :ref:`Tune Trainable <tune-60-seconds>` API: this is how you define your
custom training logic, and it's also where you'll define which trial state to checkpoint.
In this guide, we will show how to save and load checkpoints for Tune's Function Trainable and Class Trainable APIs,
as well as walk you through configuration options.

.. _tune-function-trainable-checkpointing:

Function API Checkpointing
--------------------------

If using Ray Tune's Function API, one can save and load checkpoints in the following manner.
To create a checkpoint, use the :meth:`~ray.train.Checkpoint.from_directory` APIs.

.. literalinclude:: /tune/doc_code/trial_checkpoint.py
    :language: python
    :start-after: __function_api_checkpointing_from_dir_start__
    :end-before: __function_api_checkpointing_from_dir_end__

In the above code snippet:

- We implement *checkpoint saving* with :meth:`train.report(..., checkpoint=checkpoint) <ray.train.report>`. Note that every checkpoint must be reported alongside a set of metrics -- this way, checkpoints can be ordered with respect to a specified metric.
- The saved checkpoint during training iteration `epoch` is saved to the path ``<storage_path>/<exp_name>/<trial_name>/checkpoint_<epoch>`` on the node on which training happens and can be further synced to a consolidated storage location depending on the :ref:`storage configuration <tune-storage-options>`.
- We implement *checkpoint loading* with :meth:`train.get_checkpoint() <ray.train.get_checkpoint>`. This will be populated with a trial's latest checkpoint whenever Tune restores a trial. This happens when (1) a trial is configured to retry after encountering a failure, (2) the experiment is being restored, and (3) the trial is being resumed after a pause (ex: :doc:`PBT </tune/examples/pbt_guide>`).

  .. TODO: for (1), link to tune fault tolerance guide. For (2), link to tune restore guide.

.. note::
    ``checkpoint_frequency`` and ``checkpoint_at_end`` will not work with Function API checkpointing.
    These are configured manually with Function Trainable. For example, if you want to checkpoint every three
    epochs, you can do so through:

    .. literalinclude:: /tune/doc_code/trial_checkpoint.py
        :language: python
        :start-after: __function_api_checkpointing_periodic_start__
        :end-before: __function_api_checkpointing_periodic_end__


See :class:`here for more information on creating checkpoints <ray.train.Checkpoint>`.


.. _tune-class-trainable-checkpointing:

Class API Checkpointing
-----------------------

You can also implement checkpoint/restore using the Trainable Class API:

.. literalinclude:: /tune/doc_code/trial_checkpoint.py
    :language: python
    :start-after: __class_api_checkpointing_start__
    :end-before: __class_api_checkpointing_end__

You can checkpoint with three different mechanisms: manually, periodically, and at termination.

Manual Checkpointing
~~~~~~~~~~~~~~~~~~~~

A custom Trainable can manually trigger checkpointing by returning ``should_checkpoint: True``
(or ``tune.result.SHOULD_CHECKPOINT: True``) in the result dictionary of `step`.
This can be especially helpful in spot instances:

.. literalinclude:: /tune/doc_code/trial_checkpoint.py
    :language: python
    :start-after: __class_api_manual_checkpointing_start__
    :end-before: __class_api_manual_checkpointing_end__

In the above example, if ``detect_instance_preemption`` returns True, manual checkpointing can be triggered.


Periodic Checkpointing
~~~~~~~~~~~~~~~~~~~~~~

This can be enabled by setting ``checkpoint_frequency=N`` to checkpoint trials every *N* iterations, e.g.:

.. literalinclude:: /tune/doc_code/trial_checkpoint.py
    :language: python
    :start-after: __class_api_periodic_checkpointing_start__
    :end-before: __class_api_periodic_checkpointing_end__


Checkpointing at Termination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The checkpoint_frequency may not coincide with the exact end of an experiment.
If you want a checkpoint to be created at the end of a trial, you can additionally set the ``checkpoint_at_end=True``:

.. literalinclude:: /tune/doc_code/trial_checkpoint.py
    :language: python
    :start-after: __class_api_end_checkpointing_start__
    :end-before: __class_api_end_checkpointing_end__


Configurations
--------------
Checkpointing can be configured through :class:`CheckpointConfig <ray.train.CheckpointConfig>`.
Some of the configurations do not apply to Function Trainable API, since checkpointing frequency
is determined manually within the user-defined training loop. See the compatibility matrix below.

.. list-table::
   :header-rows: 1

   * -
     - Class API
     - Function API
   * - ``num_to_keep``
     - ✅
     - ✅
   * - ``checkpoint_score_attribute``
     - ✅
     - ✅
   * - ``checkpoint_score_order``
     - ✅
     - ✅
   * - ``checkpoint_frequency``
     - ✅
     - ❌
   * - ``checkpoint_at_end``
     - ✅
     - ❌



Summary
=======

In this user guide, we covered how to save and load trial checkpoints in Tune. Once checkpointing is enabled,
move onto one of the following guides to find out how to:

- :ref:`Extract checkpoints from Tune experiment results <tune-analysis-guide>`
- :ref:`Configure persistent storage options <tune-storage-options>` for a :ref:`distributed Tune experiment <tune-distributed-ref>`

.. _tune-persisted-experiment-data:

Appendix: Types of data stored by Tune
--------------------------------------

Experiment Checkpoints
~~~~~~~~~~~~~~~~~~~~~~

Experiment-level checkpoints save the experiment state. This includes the state of the searcher,
the list of trials and their statuses (e.g., PENDING, RUNNING, TERMINATED, ERROR), and
metadata pertaining to each trial (e.g., hyperparameter configuration, some derived trial results
(min, max, last), etc).

The experiment-level checkpoint is periodically saved by the driver on the head node.
By default, the frequency at which it is saved is automatically
adjusted so that at most 5% of the time is spent saving experiment checkpoints,
and the remaining time is used for handling training results and scheduling.
This time can also be adjusted with the
:ref:`TUNE_GLOBAL_CHECKPOINT_S environment variable <tune-env-vars>`.

Trial Checkpoints
~~~~~~~~~~~~~~~~~

Trial-level checkpoints capture the per-trial state. This often includes the model and optimizer states.
Following are a few uses of trial checkpoints:

- If the trial is interrupted for some reason (e.g., on spot instances), it can be resumed from the last state. No training time is lost.
- Some searchers or schedulers pause trials to free up resources for other trials to train in the meantime. This only makes sense if the trials can then continue training from the latest state.
- The checkpoint can be later used for other downstream tasks like batch inference.

Learn how to save and load trial checkpoints :ref:`here <tune-trial-checkpoint>`.

Trial Results
~~~~~~~~~~~~~

Metrics reported by trials are saved and logged to their respective trial directories.
This is the data stored in CSV, JSON or Tensorboard (events.out.tfevents.*) formats.
that can be inspected by Tensorboard and used for post-experiment analysis.

How does Tune work?
===================

This page provides an overview of Tune's inner workings.
We describe in detail what happens when you call ``Tuner.fit()``, what the lifecycle of a Tune trial looks like
and what the architectural components of Tune are.

.. tip:: Before you continue, be sure to have read :ref:`the Tune Key Concepts page <tune-60-seconds>`.

What happens in ``Tuner.fit``?
------------------------------

When calling the following:

.. code-block:: python

    space = {"x": tune.uniform(0, 1)}
    tuner = tune.Tuner(
        my_trainable, 
        param_space=space, 
        tune_config=tune.TuneConfig(num_samples=10),
    )
    results = tuner.fit()

The provided ``my_trainable`` is evaluated multiple times in parallel
with different hyperparameters (sampled from ``uniform(0, 1)``).

Every Tune run consists of "driver process" and many "worker processes".
The driver process is the python process that calls ``Tuner.fit()`` (which calls ``ray.init()`` underneath the hood).
The Tune driver process runs on the node where you run your script (which calls ``Tuner.fit()``),
while Ray Tune trainable "actors" run on any node (either on the same node or on worker nodes (distributed Ray only)).

.. note:: :ref:`Ray Actors <actor-guide>` allow you to parallelize an instance of a class in Python.
    When you instantiate a class that is a Ray actor, Ray will start a instance of that class on a separate process
    either on the same machine (or another distributed machine, if running a Ray cluster).
    This actor can then asynchronously execute method calls and maintain its own internal state.

The driver spawns parallel worker processes (:ref:`Ray actors <actor-guide>`)
that are responsible for evaluating each trial using its hyperparameter configuration and the provided trainable.

While the Trainable is executing (:ref:`trainable-execution`), the Tune Driver communicates with each actor
via actor methods to receive intermediate training results and pause/stop actors (see :ref:`trial-lifecycle`).

When the Trainable terminates (or is stopped), the actor is also terminated.

.. _trainable-execution:

The execution of a trainable in Tune
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tune uses :ref:`Ray actors <actor-guide>` to parallelize the evaluation of multiple hyperparameter configurations.
Each actor is a Python process that executes an instance of the user-provided Trainable.

The definition of the user-provided Trainable will be
:ref:`serialized via cloudpickle <serialization-guide>`) and sent to each actor process.
Each Ray actor will start an instance of the Trainable to be executed.

If the Trainable is a class, it will be executed iteratively by calling ``train/step``.
After each invocation, the driver is notified that a "result dict" is ready.
The driver will then pull the result via ``ray.get``.

If the trainable is a callable or a function, it will be executed on the Ray actor process on a separate execution thread.
Whenever ``session.report`` is called, the execution thread is paused and waits for the driver to pull a
result (see `function_trainable.py <https://github.com/ray-project/ray/blob/master/python/ray/tune/trainable/function_trainable.py>`__.
After pulling, the actor’s execution thread will automatically resume.


Resource Management in Tune
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before running a trial, the Ray Tune driver will check whether there are available
resources on the cluster (see :ref:`resource-requirements`).
It will compare the available resources with the resources required by the trial.

If there is space on the cluster, then the Tune Driver will start a Ray actor (worker).
This actor will be scheduled and executed on some node where the resources are available.
See :doc:`tune-resources` for more information.

.. _trial-lifecycle:

Lifecycle of a Tune Trial
-------------------------

A trial's life cycle consists of 6 stages:

* **Initialization** (generation): A trial is first generated as a hyperparameter sample,
  and its parameters are configured according to what was provided in ``Tuner``.
  Trials are then placed into a queue to be executed (with status PENDING).

* **PENDING**: A pending trial is a trial to be executed on the machine.
  Every trial is configured with resource values. Whenever the trial’s resource values are available,
  Tune will run the trial (by starting a ray actor holding the config and the training function.

* **RUNNING**: A running trial is assigned a Ray Actor. There can be multiple running trials in parallel.
  See the :ref:`trainable execution <trainable-execution>` section for more details.

* **ERRORED**: If a running trial throws an exception, Tune will catch that exception and mark the trial as errored.
  Note that exceptions can be propagated from an actor to the main Tune driver process.
  If max_retries is set, Tune will set the trial back into "PENDING" and later start it from the last checkpoint.

* **TERMINATED**: A trial is terminated if it is stopped by a Stopper/Scheduler.
  If using the Function API, the trial is also terminated when the function stops.

* **PAUSED**: A trial can be paused by a Trial scheduler. This means that the trial’s actor will be stopped.
  A paused trial can later be resumed from the most recent checkpoint.


Tune's Architecture
-------------------

.. image:: ../../images/tune-arch.png

The blue boxes refer to internal components, while green boxes are public-facing.

Tune's main components consist of
the :class:`~ray.tune.execution.tune_controller.TuneController`,
:class:`~ray.tune.experiment.trial.Trial` objects,
a :class:`~ray.tune.search.search_algorithm.SearchAlgorithm`,
a :class:`~ray.tune.schedulers.trial_scheduler.TrialScheduler`,
and a :class:`~ray.tune.trainable.trainable.Trainable`,

.. _trial-runner-flow:

This is an illustration of the high-level training flow and how some of the components interact:

*Note: This figure is horizontally scrollable*

.. figure:: ../../images/tune-trial-runner-flow-horizontal.png
    :class: horizontal-scroll


TuneController
~~~~~~~~~~~~~~

[`source code <https://github.com/ray-project/ray/blob/master/python/ray/tune/execution/tune_controller.py>`__]
This is the main driver of the training loop. This component
uses the TrialScheduler to prioritize and execute trials,
queries the SearchAlgorithm for new
configurations to evaluate, and handles the fault tolerance logic.

**Fault Tolerance**: The TuneController executes checkpointing if ``checkpoint_freq``
is set, along with automatic trial restarting in case of trial failures (if ``max_failures`` is set).
For example, if a node is lost while a trial (specifically, the corresponding
Trainable of the trial) is still executing on that node and checkpointing
is enabled, the trial will then be reverted to a ``"PENDING"`` state and resumed
from the last available checkpoint when it is run.
The TuneController is also in charge of checkpointing the entire experiment execution state
upon each loop iteration. This allows users to restart their experiment
in case of machine failure.

See the docstring at :class:`~ray.tune.execution.tune_controller.TuneController`.

Trial objects
~~~~~~~~~~~~~

[`source code <https://github.com/ray-project/ray/blob/master/python/ray/tune/experiment/trial.py>`__]
This is an internal data structure that contains metadata about each training run. Each Trial
object is mapped one-to-one with a Trainable object but are not themselves
distributed/remote. Trial objects transition among
the following states: ``"PENDING"``, ``"RUNNING"``, ``"PAUSED"``, ``"ERRORED"``, and
``"TERMINATED"``.

See the docstring at :ref:`trial-docstring`.

SearchAlg
~~~~~~~~~
[`source code <https://github.com/ray-project/ray/tree/master/python/ray/tune/search>`__]
The SearchAlgorithm is a user-provided object
that is used for querying new hyperparameter configurations to evaluate.

SearchAlgorithms will be notified every time a trial finishes
executing one training step (of ``train()``), every time a trial
errors, and every time a trial completes.

TrialScheduler
~~~~~~~~~~~~~~
[`source code <https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers>`__]
TrialSchedulers operate over a set of possible trials to run,
prioritizing trial execution given available cluster resources.

TrialSchedulers are given the ability to kill or pause trials,
and also are given the ability to reorder/prioritize incoming trials.

Trainables
~~~~~~~~~~
[`source code <https://github.com/ray-project/ray/blob/master/python/ray/tune/trainable/trainable.py>`__]
These are user-provided objects that are used for
the training process. If a class is provided, it is expected to conform to the
Trainable interface. If a function is provided. it is wrapped into a
Trainable class, and the function itself is executed on a separate thread.

Trainables will execute one step of ``train()`` before notifying the TrialRunner.


.. _tune-stopping-guide:
.. _tune-stopping-ref:

How to Define Stopping Criteria for a Ray Tune Experiment
=========================================================

When running a Tune experiment, it can be challenging to determine the ideal duration of training beforehand. Stopping criteria in Tune can be useful for terminating training based on specific conditions.

For instance, one may want to set up the experiment to stop under the following circumstances:

1. Set up an experiment to end after ``N`` epochs or when the reported evaluation score surpasses a particular threshold, whichever occurs first.
2. Stop the experiment after ``T`` seconds.
3. Terminate when trials encounter runtime errors.
4. Stop underperforming trials early by utilizing Tune's early-stopping schedulers.

This user guide will illustrate how to achieve these types of stopping criteria in a Tune experiment.

For all the code examples, we use the following training function for demonstration:

.. literalinclude:: /tune/doc_code/stopping.py
    :language: python
    :start-after: __stopping_example_trainable_start__
    :end-before: __stopping_example_trainable_end__

Stop a Tune experiment manually
-------------------------------

If you send a ``SIGINT`` signal to the process running :meth:`Tuner.fit() <ray.tune.Tuner.fit>`
(which is usually what happens when you press ``Ctrl+C`` in the terminal), Ray Tune shuts
down training gracefully and saves the final experiment state.

.. note::

    Forcefully terminating a Tune experiment, for example, through multiple ``Ctrl+C``
    commands, will not give Tune the opportunity to snapshot the experiment state
    one last time. If you resume the experiment in the future, this could result
    in resuming with stale state.

Ray Tune also accepts the ``SIGUSR1`` signal to interrupt training gracefully. This
should be used when running Ray Tune in a remote Ray task
as Ray will filter out ``SIGINT`` and ``SIGTERM`` signals per default.


Stop using metric-based criteria
--------------------------------

In addition to manual stopping, Tune provides several ways to stop experiments programmatically. The simplest way is to use metric-based criteria. These are a fixed set of thresholds that determine when the experiment should stop.

You can implement the stopping criteria using either a dictionary, a function, or a custom :class:`Stopper <ray.tune.stopper.Stopper>`.

.. tab-set::

    .. tab-item:: Dictionary

        If a dictionary is passed in, the keys may be any field in the return result of ``session.report`` in the
        Function API or ``step()`` in the Class API.

        .. note::

            This includes :ref:`auto-filled metrics <tune-autofilled-metrics>` such as ``training_iteration``.

        In the example below, each trial will be stopped either when it completes ``10`` iterations or when it
        reaches a mean accuracy of ``0.8`` or more.

        These metrics are assumed to be **increasing**, so the trial will stop once the reported metric has exceeded the threshold specified in the dictionary.

        .. literalinclude:: /tune/doc_code/stopping.py
            :language: python
            :start-after: __stopping_dict_start__
            :end-before: __stopping_dict_end__

    .. tab-item:: User-defined Function

        For more flexibility, you can pass in a function instead.
        If a function is passed in, it must take ``(trial_id: str, result: dict)`` as arguments and return a boolean
        (``True`` if trial should be stopped and ``False`` otherwise).

        In the example below, each trial will be stopped either when it completes ``10`` iterations or when it
        reaches a mean accuracy of ``0.8`` or more.

        .. literalinclude:: /tune/doc_code/stopping.py
            :language: python
            :start-after: __stopping_fn_start__
            :end-before: __stopping_fn_end__

    .. tab-item:: Custom Stopper Class

        Finally, you can implement the :class:`~ray.tune.stopper.Stopper` interface for
        stopping individual trials or even entire experiments based on custom stopping
        criteria. For example, the following example stops all trials after the criteria
        is achieved by any individual trial and prevents new ones from starting:

        .. literalinclude:: /tune/doc_code/stopping.py
            :language: python
            :start-after: __stopping_cls_start__
            :end-before: __stopping_cls_end__

        In the example, once any trial reaches a ``mean_accuracy`` of 0.8 or more, all trials will stop.

        .. note::

            When returning ``True`` from ``stop_all``, currently running trials will not stop immediately.
            They will stop after finishing their ongoing training iteration (after ``session.report`` or ``step``).

        Ray Tune comes with a set of out-of-the-box stopper classes. See the :ref:`Stopper <tune-stoppers>` documentation.


Stop trials after a certain amount of time
------------------------------------------

There are two choices to stop a Tune experiment based on time: stopping trials individually
after a specified timeout, or stopping the full experiment after a certain amount of time.

Stop trials individually with a timeout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use a dictionary stopping criteria as described above, using the ``time_total_s`` metric that is auto-filled by Tune.

.. literalinclude:: /tune/doc_code/stopping.py
    :language: python
    :start-after: __stopping_trials_by_time_start__
    :end-before: __stopping_trials_by_time_end__

.. note::

    You need to include some intermediate reporting via :meth:`train.report <ray.train.report>`
    if using the :ref:`Function Trainable API <tune-function-api>`.
    Each report will automatically record the trial's ``time_total_s``, which allows Tune to stop based on time as a metric.

    If the training loop hangs somewhere, Tune will not be able to intercept the training and stop the trial for you.
    In this case, you can explicitly implement timeout logic in the training loop.


Stop the experiment with a timeout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``TuneConfig(time_budget_s)`` configuration to tell Tune to stop the experiment after ``time_budget_s`` seconds.

.. literalinclude:: /tune/doc_code/stopping.py
    :language: python
    :start-after: __stopping_experiment_by_time_start__
    :end-before: __stopping_experiment_by_time_end__

.. note::

    You need to include some intermediate reporting via :meth:`train.report <ray.train.report>`
    if using the :ref:`Function Trainable API <tune-function-api>`, for the same reason as above.


Stop on trial failures
----------------------

In addition to stopping trials based on their performance, you can also stop the entire experiment if any trial encounters a runtime error. To do this, you can use the :class:`ray.train.FailureConfig` class.

With this configuration, if any trial encounters an error, the entire experiment will stop immediately.

.. literalinclude:: /tune/doc_code/stopping.py
    :language: python
    :start-after: __stopping_on_trial_error_start__
    :end-before: __stopping_on_trial_error_end__

This is useful when you are debugging a Tune experiment with many trials.


Early stopping with Tune schedulers
-----------------------------------

Another way to stop Tune experiments is to use early stopping schedulers.
These schedulers monitor the performance of trials and stop them early if they are not making sufficient progress.

:class:`~ray.tune.schedulers.AsyncHyperBandScheduler` and :class:`~ray.tune.schedulers.HyperBandForBOHB` are examples of early stopping schedulers built into Tune.
See :ref:`the Tune scheduler API reference <tune-schedulers>` for a full list, as well as more realistic examples.

In the following example, we use both a dictionary stopping criteria along with an early-stopping criteria:

.. literalinclude:: /tune/doc_code/stopping.py
    :language: python
    :start-after: __early_stopping_start__
    :end-before: __early_stopping_end__

Summary
-------

In this user guide, we learned how to stop Tune experiments using metrics, trial errors,
and early stopping schedulers.

See the following resources for more information:

- :ref:`Tune Stopper API reference <tune-stoppers>`
- For an experiment that was manually interrupted or the cluster dies unexpectedly while trials are still running, it's possible to resume the experiment. See :ref:`tune-fault-tolerance-ref`.


.. _tune-fault-tolerance-ref:

How to Enable Fault Tolerance in Ray Tune
=========================================

Fault tolerance is an important feature for distributed machine learning experiments
that can help mitigate the impact of node failures due to out of memory and out of disk issues.

With fault tolerance, users can:

- **Save time and resources by preserving training progress** even if a node fails.
- **Access the cost savings of preemptible spot instance nodes** in the distributed setting.

.. seealso::

    In a *distributed* Tune experiment, a prerequisite to enabling fault tolerance
    is configuring some form of persistent storage where all trial results and
    checkpoints can be consolidated. See :ref:`tune-storage-options`.

In this guide, we will cover how to enable different types of fault tolerance offered by Ray Tune.


.. _tune-experiment-level-fault-tolerance:

Experiment-level Fault Tolerance in Tune
----------------------------------------

At the experiment level, :meth:`Tuner.restore <ray.tune.Tuner.restore>`
resumes a previously interrupted experiment from where it left off.

You should use :meth:`Tuner.restore <ray.tune.Tuner.restore>` in the following cases:

1. The driver script that calls :meth:`Tuner.fit() <ray.tune.Tuner.fit>` errors out (e.g., due to the head node running out of memory or out of disk).
2. The experiment is manually interrupted with ``Ctrl+C``.
3. The entire cluster, and the experiment along with it, crashes due to an ephemeral error such as the network going down or Ray object store memory filling up.

.. note::

    :meth:`Tuner.restore <ray.tune.Tuner.restore>` is *not* meant for resuming a terminated
    experiment and modifying hyperparameter search spaces or stopping criteria.
    Rather, experiment restoration is meant to resume and complete the *exact job*
    that was previously submitted via :meth:`Tuner.fit <ray.tune.Tuner.fit>`.

    For example, consider a Tune experiment configured to run for ``10`` training iterations,
    where all trials have already completed.
    :meth:`Tuner.restore <ray.tune.Tuner.restore>` cannot be used to restore the experiment,
    change the number of training iterations to ``20``, then continue training.

    Instead, this should be achieved by starting a *new* experiment and initializing
    your model weights with a checkpoint from the previous experiment.
    See :ref:`this FAQ post <tune-iterative-experimentation>` for an example.


.. note::

    Bugs in your user-defined training loop cannot be fixed with restoration. Instead, the issue
    that caused the experiment to crash in the first place should be *ephemeral*,
    meaning that the retry attempt after restoring can succeed the next time.


.. _tune-experiment-restore-example:

Restore a Tune Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say your initial Tune experiment is configured as follows.
The actual training loop is just for demonstration purposes: the important detail is that
:ref:`saving and loading checkpoints has been implemented in the trainable <tune-trial-checkpoint>`.

.. literalinclude:: /tune/doc_code/fault_tolerance.py
    :language: python
    :start-after: __ft_initial_run_start__
    :end-before: __ft_initial_run_end__

The results and checkpoints of the experiment are saved to ``~/ray_results/tune_fault_tolerance_guide``,
as configured by :class:`~ray.train.RunConfig`.
If the experiment has been interrupted due to one of the reasons listed above, use this path to resume:

.. literalinclude:: /tune/doc_code/fault_tolerance.py
    :language: python
    :start-after: __ft_restored_run_start__
    :end-before: __ft_restored_run_end__

.. tip::

    You can also restore the experiment from a cloud bucket path:

    .. code-block:: python

        tuner = tune.Tuner.restore(
            path="s3://cloud-bucket/tune_fault_tolerance_guide", trainable=trainable
        )

    See :ref:`tune-storage-options`.


Restore Configurations
~~~~~~~~~~~~~~~~~~~~~~

Tune allows configuring which trials should be resumed, based on their status when the experiment was interrupted:

- Unfinished trials left in the ``RUNNING`` state will be resumed by default.
- Trials that have ``ERRORED`` can be resumed or retried from scratch.
- ``TERMINATED`` trials *cannot* be resumed.

.. literalinclude:: /tune/doc_code/fault_tolerance.py
    :language: python
    :start-after: __ft_restore_options_start__
    :end-before: __ft_restore_options_end__


.. _tune-experiment-autoresume-example:

Auto-resume
~~~~~~~~~~~

When running in a production setting, one may want a *single script* that (1) launches the
initial training run in the beginning and (2) restores the experiment if (1) already happened.

Use the :meth:`Tuner.can_restore <ray.tune.Tuner.can_restore>` utility to accomplish this:

.. literalinclude:: /tune/doc_code/fault_tolerance.py
    :language: python
    :start-after: __ft_restore_multiplexing_start__
    :end-before: __ft_restore_multiplexing_end__

Running this script the first time will launch the initial training run.
Running this script the second time will attempt to resume from the outputs of the first run.


Tune Experiment Restoration with Ray Object References (Advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Experiment restoration often happens in a different Ray session than the original run,
in which case Ray object references are automatically garbage collected.
If object references are saved along with experiment state (e.g., within each trial's config),
then attempting to retrieve theses objects will not work properly after restoration:
the objects these references point to no longer exist.

To work around this, you must re-create these objects, put them in the Ray object store,
and then pass the new object references to Tune.

Example
*******

Let's say we have some large pre-trained model that we want to use in some way in our training loop.
For example, this could be a image classification model used to calculate an Inception Score
to evaluate the quality of a generative model.
We may have multiple models that we want to tune over, where each trial samples one of the models to use.

.. literalinclude:: /tune/doc_code/fault_tolerance.py
    :language: python
    :start-after: __ft_restore_objrefs_initial_start__
    :end-before: __ft_restore_objrefs_initial_end__

To restore, we just need to re-specify the ``param_space`` via :meth:`Tuner.restore <ray.tune.Tuner.restore>`:

.. literalinclude:: /tune/doc_code/fault_tolerance.py
    :language: python
    :start-after: __ft_restore_objrefs_restored_start__
    :end-before: __ft_restore_objrefs_restored_end__

.. note::

    If you're tuning over :ref:`Ray Data <data>`, you'll also need to re-specify them in the ``param_space``.
    Ray Data can contain object references, so the same problems described above apply.

    See below for an example:

    .. code-block:: python

        ds_1 = ray.data.from_items([{"x": i, "y": 2 * i} for i in range(128)])
        ds_2 = ray.data.from_items([{"x": i, "y": 3 * i} for i in range(128)])

        param_space = {
            "datasets": {"train": tune.grid_search([ds_1, ds_2])},
        }

        tuner = tune.Tuner.restore(..., param_space=param_space)

.. _tune-trial-level-fault-tolerance:

Trial-level Fault Tolerance in Tune
-----------------------------------

Trial-level fault tolerance deals with individual trial failures in the cluster, which can be caused by:

- Running with preemptible spot instances.
- Ephemeral network connection issues.
- Nodes running out of memory or out of disk space.

Ray Tune provides a way to configure failure handling of individual trials with the :class:`~ray.train.FailureConfig`.

Assuming that we're using the ``trainable`` from the previous example that implements
trial checkpoint saving and loading, here is how to configure :class:`~ray.train.FailureConfig`:

.. literalinclude:: /tune/doc_code/fault_tolerance.py
    :language: python
    :start-after: __ft_trial_failure_start__
    :end-before: __ft_trial_failure_end__

When a trial encounters a runtime error, the above configuration will re-schedule that trial
up to ``max_failures=3`` times.

Similarly, if a node failure occurs for node ``X`` (e.g., pre-empted or lost connection),
this configuration will reschedule all trials that lived on node ``X`` up to ``3`` times.


Summary
-------

In this user guide, we covered how to enable experiment-level and trial-level fault tolerance in Ray Tune.

See the following resources for more information:

- :ref:`tune-storage-options`
- :ref:`tune-distributed-ref`
- :ref:`tune-trial-checkpoint`


.. _tune-parallelism:

A Guide To Parallelism and Resources for Ray Tune
-------------------------------------------------

Parallelism is determined by per trial resources (defaulting to 1 CPU, 0 GPU per trial)
and the resources available to Tune (``ray.cluster_resources()``).

By default, Tune automatically runs `N` concurrent trials, where `N` is the number
of CPUs (cores) on your machine.

.. code-block:: python

    # If you have 4 CPUs on your machine, this will run 4 concurrent trials at a time.
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(num_samples=10)
    )
    results = tuner.fit()

You can override this per trial resources with :func:`tune.with_resources <ray.tune.with_resources>`. Here you can
specify your resource requests using either a dictionary or a
:class:`PlacementGroupFactory <ray.tune.execution.placement_groups.PlacementGroupFactory>`
object. In either case, Ray Tune will try to start a placement group for each trial.

.. code-block:: python

    # If you have 4 CPUs on your machine, this will run 2 concurrent trials at a time.
    trainable_with_resources = tune.with_resources(trainable, {"cpu": 2})
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(num_samples=10)
    )
    results = tuner.fit()

    # If you have 4 CPUs on your machine, this will run 1 trial at a time.
    trainable_with_resources = tune.with_resources(trainable, {"cpu": 4})
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(num_samples=10)
    )
    results = tuner.fit()

    # Fractional values are also supported, (i.e., {"cpu": 0.5}).
    # If you have 4 CPUs on your machine, this will run 8 concurrent trials at a time.
    trainable_with_resources = tune.with_resources(trainable, {"cpu": 0.5})
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(num_samples=10)
    )
    results = tuner.fit()

    # Custom resource allocation via lambda functions are also supported.
    # If you want to allocate gpu resources to trials based on a setting in your config
    trainable_with_resources = tune.with_resources(trainable,
        resources=lambda spec: {"gpu": 1} if spec.config.use_gpu else {"gpu": 0})
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(num_samples=10)
    )
    results = tuner.fit()


Tune will allocate the specified GPU and CPU as specified by ``tune.with_resources`` to each individual trial.
Even if the trial cannot be scheduled right now, Ray Tune will still try to start the respective placement group. If not enough resources are available, this will trigger
:ref:`autoscaling behavior <cluster-index>` if you're using the Ray cluster launcher.

.. warning::
    ``tune.with_resources`` cannot be used with :ref:`Ray Train Trainers <train-docs>`. If you are passing a Trainer to a Tuner, specify the resource requirements in the Trainer instance using :class:`~ray.train.ScalingConfig`. The general principles outlined below still apply.

It is also possible to specify memory (``"memory"``, in bytes) and custom resource requirements.

If your trainable function starts more remote workers, you will need to pass so-called placement group
factory objects to request these resources.
See the :class:`PlacementGroupFactory documentation <ray.tune.execution.placement_groups.PlacementGroupFactory>`
for further information.
This also applies if you are using other libraries making use of Ray, such as Modin.
Failure to set resources correctly may result in a deadlock, "hanging" the cluster.

.. note::
    The resources specified this way will only be allocated for scheduling Tune trials.
    These resources will not be enforced on your objective function (Tune trainable) automatically.
    You will have to make sure your trainable has enough resources to run (e.g. by setting ``n_jobs`` for a
    scikit-learn model accordingly).

How to leverage GPUs in Tune?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To leverage GPUs, you must set ``gpu`` in ``tune.with_resources(trainable, resources_per_trial)``.
This will automatically set ``CUDA_VISIBLE_DEVICES`` for each trial.

.. code-block:: python

    # If you have 8 GPUs, this will run 8 trials at once.
    trainable_with_gpu = tune.with_resources(trainable, {"gpu": 1})
    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(num_samples=10)
    )
    results = tuner.fit()

    # If you have 4 CPUs and 1 GPU on your machine, this will run 1 trial at a time.
    trainable_with_cpu_gpu = tune.with_resources(trainable, {"cpu": 2, "gpu": 1})
    tuner = tune.Tuner(
        trainable_with_cpu_gpu,
        tune_config=tune.TuneConfig(num_samples=10)
    )
    results = tuner.fit()

You can find an example of this in the :doc:`Keras MNIST example </tune/examples/tune_mnist_keras>`.

.. warning:: If ``gpu`` is not set, ``CUDA_VISIBLE_DEVICES`` environment variable will be set as empty, disallowing GPU access.

**Troubleshooting**: Occasionally, you may run into GPU memory issues when running a new trial. This may be
due to the previous trial not cleaning up its GPU state fast enough. To avoid this,
you can use :func:`tune.utils.wait_for_gpu <ray.tune.utils.wait_for_gpu>`.

.. _tune-dist-training:

How to run distributed training with Tune?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To tune distributed training jobs, you can use Ray Tune with Ray Train. Ray Tune will run multiple trials in parallel, with each trial running distributed training with Ray Train.

For more details, see :ref:`Ray Train Hyperparameter Optimization <train-tune>`.

How to limit concurrency in Tune?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To specifies the max number of trials to run concurrently, set `max_concurrent_trials` in :class:`TuneConfig <ray.tune.tune_config.TuneConfig>`.

Note that actual parallelism can be less than `max_concurrent_trials` and will be determined by how many trials
can fit in the cluster at once (i.e., if you have a trial that requires 16 GPUs, your cluster has 32 GPUs,
and `max_concurrent_trials=10`, the `Tuner` can only run 2 trials concurrently).

.. code-block:: python 

    from ray.tune import TuneConfig

    config = TuneConfig(
        # ...
        num_samples=100,
        max_concurrent_trials=10,
    )


:orphan:

Scalability and Overhead Benchmarks for Ray Tune
================================================

We conducted a series of micro-benchmarks where we evaluated the scalability of Ray Tune and analyzed the
performance overhead we observed. The results from these benchmarks are reflected in the documentation,
e.g. when we make suggestions on :ref:`how to remove performance bottlenecks <tune-bottlenecks>`.

This page gives an overview over the experiments we did. For each of these experiments, the goal was to
examine the total runtime of the experiment and address issues when the observed overhead compared to the
minimal theoretical time was too high (e.g. more than 20% overhead).

In some of the experiments we tweaked the default settings for maximum throughput, e.g. by disabling
trial synchronization or result logging. If this is the case, this is stated in the respective benchmark
description.


.. list-table:: Ray Tune scalability benchmarks overview
   :header-rows: 1

   * - Variable
     - # of trials
     - Results/second /trial
     - # of nodes
     - # CPUs/node
     - Trial length (s)
     - Observed runtime
   * - `Trial bookkeeping /scheduling overhead <https://github.com/ray-project/ray/blob/master/release/tune_tests/scalability_tests/workloads/test_bookkeeping_overhead.py>`_
     - 10,000
     - 1
     - 1
     - 16
     - 1
     - | 715.27
       | (625 minimum)
   * - `Result throughput (many trials) <https://github.com/ray-project/ray/blob/master/release/tune_tests/scalability_tests/workloads/test_result_throughput_cluster.py>`_
     - 1,000
     - 0.1
     - 16
     - 64
     - 100
     - 168.18
   * - `Result throughput (many results) <https://github.com/ray-project/ray/blob/master/release/tune_tests/scalability_tests/workloads/test_result_throughput_single_node.py>`_
     - 96
     - 10
     - 1
     - 96
     - 100
     - 168.94
   * - `Network communication overhead <https://github.com/ray-project/ray/blob/master/release/tune_tests/scalability_tests/workloads/test_network_overhead.py>`_
     - 200
     - 1
     - 200
     - 2
     - 300
     - 2280.82
   * - `Long running, 3.75 GB checkpoints <https://github.com/ray-project/ray/blob/master/release/tune_tests/scalability_tests/workloads/test_long_running_large_checkpoints.py>`_
     - 16
     - | Results: 1/60
       | Checkpoint: 1/900
     - 1
     - 16
     - 86,400
     - 88687.41
   * - `Durable trainable <https://github.com/ray-project/ray/blob/master/release/tune_tests/scalability_tests/workloads/test_durable_trainable.py>`_
     - 16
     - | 10/60
       | with 10MB CP
     - 16
     - 2
     - 300
     - 392.42


Below we discuss some insights on results where we observed much overhead.


Result throughput
-----------------

Result throughput describes the number of results Ray Tune can process in a given timeframe (e.g.
"results per second").
The higher the throughput, the more concurrent results can be processed without major delays.

Result throughput is limited by the time it takes to process results. When a trial reports results, it only
continues training once the trial executor re-triggered the remote training function. If many trials report
results at the same time, each subsequent remote training call is only triggered after handling that trial's
results.

To speed the process up, Ray Tune adaptively buffers results, so that trial training is continued earlier if
many trials are running in parallel and report many results at the same time. Still, processing hundreds of
results per trial for dozens or hundreds of trials can become a bottleneck.

**Main insight**: Ray Tune will throw a warning when trial processing becomes a bottleneck. If you notice
that this becomes a problem, please follow our guidelines outlined :ref:`in the FAQ <tune-bottlenecks>`.
Generally, it is advised to not report too many results at the same time. Consider increasing the report
intervals by a factor of 5-10x.

Below we present more detailed results on the result throughput performance.

Benchmarking many concurrent Tune trials
""""""""""""""""""""""""""""""""""""""""

In this setup, loggers (CSV, JSON, and TensorBoardX) and trial synchronization are disabled, except when
explicitly noted.

In this experiment, we're running many concurrent trials (up to 1,000) on a cluster. We then adjust the
reporting frequency (number of results per second) of the trials to measure the throughput limits.

It seems that around 500 total results/second seem to be the threshold for acceptable performance
when logging and synchronization are disabled. With logging enabled, around 50-100 results per second
can still be managed without too much overhead, but after that measures to decrease incoming results
should be considered.

+-------------+--------------------------+---------+---------------+------------------+---------+
| # of trials | Results / second / trial | # Nodes | # CPUs / Node | Length of trial. | Current |
+=============+==========================+=========+===============+==================+=========+
| 1,000       | 10                       | 16      | 64            | 100s             | 248.39  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 1,000       | 1                        | 16      | 64            | 100s             | 175.00  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 1,000       | 0.1 with logging         | 16      | 64            | 100s             | 168.18  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 384         | 10                       | 16      | 64            | 100s             | 125.17  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 256         | 50                       | 16      | 64            | 100s             | 307.02  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 256         | 20                       | 16      | 64            | 100s             | 146.20  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 256         | 10                       | 16      | 64            | 100s             | 113.40  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 256         | 10 with logging          | 16      | 64            | 100s             | 436.12  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 256         | 0.1 with logging         | 16      | 64            | 100s             | 106.75  |
+-------------+--------------------------+---------+---------------+------------------+---------+


Benchmarking many Tune results on a single node
"""""""""""""""""""""""""""""""""""""""""""""""

In this setup, loggers (CSV, JSON, and TensorBoardX) are disabled, except when
explicitly noted.

In this experiment, we're running 96 concurrent trials on a single node. We then adjust the
reporting frequency (number of results per second) of the trials to find the throughput limits.
Compared to the cluster experiment setup, we report much more often, as we're running less total trials in parallel.

On a single node, throughput seems to be a bit higher. With logging, handling 1000 results per second
seems acceptable in terms of overhead, though you should probably still target for a lower number.

+-------------+--------------------------+---------+---------------+------------------+---------+
| # of trials | Results / second / trial | # Nodes | # CPUs / Node | Length of trial. | Current |
+=============+==========================+=========+===============+==================+=========+
| 96          | 500                      | 1       | 96            | 100s             | 959.32  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 96          | 100                      | 1       | 96            | 100s             | 219.48  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 96          | 80                       | 1       | 96            | 100s             | 197.15  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 96          | 50                       | 1       | 96            | 100s             | 110.55  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 96          | 50 with logging          | 1       | 96            | 100s             | 702.64  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 96          | 10                       | 1       | 96            | 100s             | 103.51  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 96          | 10 with logging          | 1       | 96            | 100s             | 168.94  |
+-------------+--------------------------+---------+---------------+------------------+---------+


Network overhead in Ray Tune
----------------------------

Running Ray Tune on a distributed setup leads to network communication overhead. This is mostly due to
trial synchronization, where results and checkpoints are periodically synchronized and sent via the network.
Per default this happens via SSH, where connnection initialization can take between 1 and 2 seconds each time.
Since this is a blocking operation that happens on a per-trial basis, running many concurrent trials
quickly becomes bottlenecked by this synchronization.

In this experiment, we ran a number of trials on a cluster. Each trial was run on a separate node. We
varied the number of concurrent trials (and nodes) to see how much network communication affects
total runtime.

**Main insight**: When running many concurrent trials in a distributed setup, consider using
:ref:`cloud checkpointing <tune-cloud-checkpointing>` for checkpoint synchronization instead. Another option would
be to use a shared storage and disable syncing to driver. The best practices are described
:ref:`here for Kubernetes setups <tune-kubernetes>` but is applicable for any kind of setup.


In the table below we present more detailed results on the network communication overhead.

+-------------+--------------------------+---------+---------------+------------------+---------+
| # of trials | Results / second / trial | # Nodes | # CPUs / Node | Length of trial  | Current |
+=============+==========================+=========+===============+==================+=========+
| 200         | 1                        | 200     | 2             | 300s             | 2280.82 |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 100         | 1                        | 100     | 2             | 300s             | 1470    |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 100         | 0.01                     | 100     | 2             | 300s             | 473.41  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 50          | 1                        | 50      | 2             | 300s             | 474.30  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 50          | 0.1                      | 50      | 2             | 300s             | 441.54  |
+-------------+--------------------------+---------+---------------+------------------+---------+
| 10          | 1                        | 10      | 2             | 300s             | 334.37  |
+-------------+--------------------------+---------+---------------+------------------+---------+


A Guide To Callbacks & Metrics in Tune
======================================

.. _tune-callbacks:

How to work with Callbacks in Ray Tune?
---------------------------------------

Ray Tune supports callbacks that are called during various times of the training process.
Callbacks can be passed as a parameter to ``RunConfig``, taken in by ``Tuner``, and the sub-method you provide will be invoked automatically.

This simple callback just prints a metric each time a result is received:

.. code-block:: python

    from ray import train, tune
    from ray.train import RunConfig
    from ray.tune import Callback


    class MyCallback(Callback):
        def on_trial_result(self, iteration, trials, trial, result, **info):
            print(f"Got result: {result['metric']}")


    def train_fn(config):
        for i in range(10):
            train.report({"metric": i})


    tuner = tune.Tuner(
        train_fn,
        run_config=RunConfig(callbacks=[MyCallback()]))
    tuner.fit()

For more details and available hooks, please :ref:`see the API docs for Ray Tune callbacks <tune-callbacks-docs>`.


.. _tune-autofilled-metrics:

How to use log metrics in Tune?
-------------------------------

You can log arbitrary values and metrics in both Function and Class training APIs:

.. code-block:: python

    def trainable(config):
        for i in range(num_epochs):
            ...
            train.report({"acc": accuracy, "metric_foo": random_metric_1, "bar": metric_2})

    class Trainable(tune.Trainable):
        def step(self):
            ...
            # don't call report here!
            return dict(acc=accuracy, metric_foo=random_metric_1, bar=metric_2)


.. tip::
    Note that ``train.report()`` is not meant to transfer large amounts of data, like models or datasets.
    Doing so can incur large overheads and slow down your Tune run significantly.

Which Tune metrics get automatically filled in?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tune has the concept of auto-filled metrics.
During training, Tune will automatically log the below metrics in addition to any user-provided values.
All of these can be used as stopping conditions or passed as a parameter to Trial Schedulers/Search Algorithms.

* ``config``: The hyperparameter configuration
* ``date``: String-formatted date and time when the result was processed
* ``done``: True if the trial has been finished, False otherwise
* ``episodes_total``: Total number of episodes (for RLlib trainables)
* ``experiment_id``: Unique experiment ID
* ``experiment_tag``: Unique experiment tag (includes parameter values)
* ``hostname``: Hostname of the worker
* ``iterations_since_restore``: The number of times ``train.report`` has been
  called after restoring the worker from a checkpoint
* ``node_ip``: Host IP of the worker
* ``pid``: Process ID (PID) of the worker process
* ``time_since_restore``: Time in seconds since restoring from a checkpoint.
* ``time_this_iter_s``: Runtime of the current training iteration in seconds (i.e.
  one call to the trainable function or to ``_train()`` in the class API.
* ``time_total_s``: Total runtime in seconds.
* ``timestamp``: Timestamp when the result was processed
* ``timesteps_since_restore``: Number of timesteps since restoring from a checkpoint
* ``timesteps_total``: Total number of timesteps
* ``training_iteration``: The number of times ``train.report()`` has been
  called
* ``trial_id``: Unique trial ID

All of these metrics can be seen in the ``Trial.last_result`` dictionary.


Logging and Outputs in Tune
===========================

By default, Tune logs results for TensorBoard, CSV, and JSON formats.
If you need to log something lower level like model weights or gradients, see :ref:`Trainable Logging <trainable-logging>`.
You can learn more about logging and customizations here: :ref:`loggers-docstring`.


.. _tune-logging:

How to configure logging in Tune?
---------------------------------

Tune will log the results of each trial to a sub-folder under a specified local dir, which defaults to ``~/ray_results``.

.. code-block:: python

    # This logs to two different trial folders:
    # ~/ray_results/trainable_name/trial_name_1 and ~/ray_results/trainable_name/trial_name_2
    # trainable_name and trial_name are autogenerated.
    tuner = tune.Tuner(trainable, run_config=RunConfig(num_samples=2))
    results = tuner.fit()

You can specify the ``storage_path`` and ``trainable_name``:

.. code-block:: python

    # This logs to 2 different trial folders:
    # ./results/test_experiment/trial_name_1 and ./results/test_experiment/trial_name_2
    # Only trial_name is autogenerated.
    tuner = tune.Tuner(trainable,
        tune_config=tune.TuneConfig(num_samples=2),
        run_config=RunConfig(storage_path="./results", name="test_experiment"))
    results = tuner.fit()


To learn more about Trials, see its detailed API documentation: :ref:`trial-docstring`.

.. _tensorboard:

How to log your Tune runs to TensorBoard?
-----------------------------------------

Tune automatically outputs TensorBoard files during ``Tuner.fit()``.
To visualize learning in tensorboard, install tensorboardX:

.. code-block:: bash

    $ pip install tensorboardX

Then, after you run an experiment, you can visualize your experiment with TensorBoard by specifying
the output directory of your results.

.. code-block:: bash

    $ tensorboard --logdir=~/ray_results/my_experiment

If you are running Ray on a remote multi-user cluster where you do not have sudo access,
you can run the following commands to make sure tensorboard is able to write to the tmp directory:

.. code-block:: bash

    $ export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=~/ray_results

.. image:: ../images/ray-tune-tensorboard.png

If using TensorFlow ``2.x``, Tune also automatically generates TensorBoard HParams output, as shown below:

.. code-block:: python

    tuner = tune.Tuner(
        ...,
        param_space={
            "lr": tune.grid_search([1e-5, 1e-4]),
            "momentum": tune.grid_search([0, 0.9])
        }
    )
    results = tuner.fit()

.. image:: ../../images/tune-hparams.png


.. _tune-console-output:

How to control console output with Tune?
----------------------------------------

User-provided fields will be outputted automatically on a best-effort basis.
You can use a :ref:`Reporter <tune-reporter-doc>` object to customize the console output.

.. code-block:: bash

    == Status ==
    Memory usage on this node: 11.4/16.0 GiB
    Using FIFO scheduling algorithm.
    Resources requested: 4/12 CPUs, 0/0 GPUs, 0.0/3.17 GiB heap, 0.0/1.07 GiB objects
    Result logdir: /Users/foo/ray_results/myexp
    Number of trials: 4 (4 RUNNING)
    +----------------------+----------+---------------------+-----------+--------+--------+----------------+-------+
    | Trial name           | status   | loc                 |    param1 | param2 |    acc | total time (s) |  iter |
    |----------------------+----------+---------------------+-----------+--------+--------+----------------+-------|
    | MyTrainable_a826033a | RUNNING  | 10.234.98.164:31115 | 0.303706  | 0.0761 | 0.1289 |        7.54952 |    15 |
    | MyTrainable_a8263fc6 | RUNNING  | 10.234.98.164:31117 | 0.929276  | 0.158  | 0.4865 |        7.0501  |    14 |
    | MyTrainable_a8267914 | RUNNING  | 10.234.98.164:31111 | 0.068426  | 0.0319 | 0.9585 |        7.0477  |    14 |
    | MyTrainable_a826b7bc | RUNNING  | 10.234.98.164:31112 | 0.729127  | 0.0748 | 0.1797 |        7.05715 |    14 |
    +----------------------+----------+---------------------+-----------+--------+--------+----------------+-------+


.. _tune-log_to_file:

How to redirect Trainable logs to files in a Tune run?
---------------------------------------------------------

In Tune, Trainables are run as remote actors. By default, Ray collects actors' stdout and stderr and prints them to
the head process (see :ref:`ray worker logs <ray-worker-logs>` for more information).
Logging that happens within Tune Trainables follows this handling by default.
However, if you wish to collect Trainable logs in files for analysis, Tune offers the option
``log_to_file`` for this.
This applies to print statements, ``warnings.warn`` and ``logger.info`` etc.

By passing ``log_to_file=True`` to ``RunConfig``, which is taken in by ``Tuner``, stdout and stderr will be logged
to ``trial_logdir/stdout`` and ``trial_logdir/stderr``, respectively:

.. code-block:: python

    tuner = tune.Tuner(
        trainable,
        run_config=RunConfig(log_to_file=True)
    )
    results = tuner.fit()

If you would like to specify the output files, you can either pass one filename,
where the combined output will be stored, or two filenames, for stdout and stderr,
respectively:

.. code-block:: python

    tuner = tune.Tuner(
        trainable,
        run_config=RunConfig(log_to_file="std_combined.log")
    )
    tuner.fit()

    tuner = tune.Tuner(
        trainable,
        run_config=RunConfig(log_to_file=("my_stdout.log", "my_stderr.log")))
    results = tuner.fit()

The file names are relative to the trial's logdir. You can pass absolute paths,
too.

Caveats
^^^^^^^
Logging that happens in distributed training workers (if you happen to use Ray Tune together with Ray Train)
is not part of this ``log_to_file`` configuration.

Where to find ``log_to_file`` files?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If your Tune workload is configured with syncing to head node, then the corresponding ``log_to_file`` outputs
can be located under each trial folder.
If your Tune workload is instead configured with syncing to cloud, then the corresponding ``log_to_file``
outputs are *NOT* synced to cloud and can only be found in the worker nodes that the corresponding trial happens.

.. note::
    This can cause problems when the trainable is moved across different nodes throughout its lifetime.
    This can happen with some schedulers or with node failures.
    We may prioritize enabling this if there are enough user requests.
    If this impacts your workflow, consider commenting on
    [this ticket](https://github.com/ray-project/ray/issues/32142).


Leave us feedback on this feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We know that logging and observability can be a huge performance boost for your workflow. Let us know what is your
preferred way to interact with logging that happens in trainables. Leave you comments in
[this ticket](https://github.com/ray-project/ray/issues/32142).

.. _trainable-logging:

How do you log arbitrary files from a Tune Trainable?
-----------------------------------------------------

By default, Tune only logs the *training result dictionaries* and *checkpoints* from your Trainable.
However, you may want to save a file that visualizes the model weights or model graph,
or use a custom logging library that requires multi-process logging.
For example, you may want to do this if you're trying to log images to TensorBoard.
We refer to these saved files as **trial artifacts**.

.. note::

    If :class:`SyncConfig(sync_artifacts=True) <ray.train.SyncConfig>`, trial artifacts
    are uploaded periodically from each trial (or from each remote training worker for Ray Train)
    to the :class:`RunConfig(storage_path) <ray.train.RunConfig>`.

    See the :class:`~ray.train.SyncConfig` API reference for artifact syncing configuration options.

You can save trial artifacts directly in the trainable, as shown below:

.. tip:: Make sure that any logging calls or objects stay within scope of the Trainable.
    You may see pickling or other serialization errors or inconsistent logs otherwise.

.. tab-set::

    .. tab-item:: Function API

        .. code-block:: python

            import logging_library  # ex: mlflow, wandb
            from ray import train

            def trainable(config):
                logging_library.init(
                    name=trial_id,
                    id=trial_id,
                    resume=trial_id,
                    reinit=True,
                    allow_val_change=True)
                logging_library.set_log_path(os.getcwd())

                for step in range(100):
                    logging_library.log_model(...)
                    logging_library.log(results, step=step)

                    # You can also just write to a file directly.
                    # The working directory is set to the trial directory, so
                    # you don't need to worry about multiple workers saving
                    # to the same location.
                    with open(f"./artifact_{step}.txt", "w") as f:
                        f.write("Artifact Data")

                    train.report(results)


    .. tab-item:: Class API

        .. code-block:: python

            import logging_library  # ex: mlflow, wandb
            from ray import tune

            class CustomLogging(tune.Trainable)
                def setup(self, config):
                    trial_id = self.trial_id
                    logging_library.init(
                        name=trial_id,
                        id=trial_id,
                        resume=trial_id,
                        reinit=True,
                        allow_val_change=True
                    )
                    logging_library.set_log_path(os.getcwd())

                def step(self):
                    logging_library.log_model(...)

                    # You can also write to a file directly.
                    # The working directory is set to the trial directory, so
                    # you don't need to worry about multiple workers saving
                    # to the same location.
                    with open(f"./artifact_{self.iteration}.txt", "w") as f:
                        f.write("Artifact Data")

                def log_result(self, result):
                    res_dict = {
                        str(k): v
                        for k, v in result.items()
                        if (v and "config" not in k and not isinstance(v, str))
                    }
                    step = result["training_iteration"]
                    logging_library.log(res_dict, step=step)


In the code snippet above, ``logging_library`` refers to whatever 3rd party logging library you are using.
Note that ``logging_library.set_log_path(os.getcwd())`` is an imaginary API that we are using
for demonstation purposes, and it highlights that the third-party library
should be configured to log to the Trainable's *working directory.* By default,
the current working directory of both functional and class trainables is set to the
corresponding trial directory once it's been launched as a remote Ray actor.


How to Build Custom Tune Loggers?
---------------------------------

You can create a custom logger by inheriting the LoggerCallback interface (:ref:`logger-interface`):

.. code-block:: python

    from typing import Dict, List

    import json
    import os

    from ray.tune.logger import LoggerCallback


    class CustomLoggerCallback(LoggerCallback):
        """Custom logger interface"""

        def __init__(self, filename: str = "log.txt"):
            self._trial_files = {}
            self._filename = filename

        def log_trial_start(self, trial: "Trial"):
            trial_logfile = os.path.join(trial.logdir, self._filename)
            self._trial_files[trial] = open(trial_logfile, "at")

        def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
            if trial in self._trial_files:
                self._trial_files[trial].write(json.dumps(result))

        def on_trial_complete(self, iteration: int, trials: List["Trial"],
                              trial: "Trial", **info):
            if trial in self._trial_files:
                self._trial_files[trial].close()
                del self._trial_files[trial]


You can then pass in your own logger as follows:

.. code-block:: python

    from ray import tune
    from ray.train import RunConfig

    tuner = tune.Tuner(
        MyTrainableClass,
        run_config=RunConfig(name="experiment_name", callbacks=[CustomLoggerCallback("log_test.txt")])
    )
    results = tuner.fit()


Per default, Ray Tune creates JSON, CSV and TensorBoardX logger callbacks if you don't pass them yourself.
You can disable this behavior by setting the ``TUNE_DISABLE_AUTO_CALLBACK_LOGGERS`` environment variable to ``"1"``.

An example of creating a custom logger can be found in :doc:`/tune/examples/includes/logging_example`.


.. _tune-api-ref:

Ray Tune API
============

.. tip:: We'd love to hear your feedback on using Tune - `get in touch <https://forms.gle/PTRvGLbKRdUfuzQo9>`_!

This section contains a reference for the Tune API. If there is anything missing, please open an issue
on `Github`_.

.. _`GitHub`: https://github.com/ray-project/ray/issues

.. toctree::
    :maxdepth: 2

    execution.rst
    result_grid.rst
    trainable.rst
    search_space.rst
    suggestion.rst
    schedulers.rst
    stoppers.rst
    reporters.rst
    syncing.rst
    logging.rst
    callbacks.rst
    env.rst
    integration.rst
    internals.rst
    cli.rst


.. _tune-reporter-doc:


Tune Console Output (Reporters)
===============================

By default, Tune reports experiment progress periodically to the command-line as follows.

.. code-block:: bash

    == Status ==
    Memory usage on this node: 11.4/16.0 GiB
    Using FIFO scheduling algorithm.
    Resources requested: 4/12 CPUs, 0/0 GPUs, 0.0/3.17 GiB heap, 0.0/1.07 GiB objects
    Result logdir: /Users/foo/ray_results/myexp
    Number of trials: 4 (4 RUNNING)
    +----------------------+----------+---------------------+-----------+--------+--------+--------+--------+------------------+-------+
    | Trial name           | status   | loc                 |    param1 | param2 | param3 |    acc |   loss |   total time (s) |  iter |
    |----------------------+----------+---------------------+-----------+--------+--------+--------+--------+------------------+-------|
    | MyTrainable_a826033a | RUNNING  | 10.234.98.164:31115 | 0.303706  | 0.0761 | 0.4328 | 0.1289 | 1.8572 |          7.54952 |    15 |
    | MyTrainable_a8263fc6 | RUNNING  | 10.234.98.164:31117 | 0.929276  | 0.158  | 0.3417 | 0.4865 | 1.6307 |          7.0501  |    14 |
    | MyTrainable_a8267914 | RUNNING  | 10.234.98.164:31111 | 0.068426  | 0.0319 | 0.1147 | 0.9585 | 1.9603 |          7.0477  |    14 |
    | MyTrainable_a826b7bc | RUNNING  | 10.234.98.164:31112 | 0.729127  | 0.0748 | 0.1784 | 0.1797 | 1.7161 |          7.05715 |    14 |
    +----------------------+----------+---------------------+-----------+--------+--------+--------+--------+------------------+-------+

Note that columns will be hidden if they are completely empty. The output can be configured in various ways by
instantiating a ``CLIReporter`` instance (or ``JupyterNotebookReporter`` if you're using jupyter notebook).
Here's an example:

.. TODO: test these snippets

.. code-block:: python

    from ray.train import RunConfig
    from ray.tune import CLIReporter

    # Limit the number of rows.
    reporter = CLIReporter(max_progress_rows=10)
    # Add a custom metric column, in addition to the default metrics.
    # Note that this must be a metric that is returned in your training results.
    reporter.add_metric_column("custom_metric")
    tuner = tune.Tuner(my_trainable, run_config=RunConfig(progress_reporter=reporter))
    results = tuner.fit()

Extending ``CLIReporter`` lets you control reporting frequency. For example:

.. code-block:: python

    from ray.tune.experiment.trial import Trial

    class ExperimentTerminationReporter(CLIReporter):
        def should_report(self, trials, done=False):
            """Reports only on experiment termination."""
            return done

    tuner = tune.Tuner(my_trainable, run_config=RunConfig(progress_reporter=ExperimentTerminationReporter()))
    results = tuner.fit()

    class TrialTerminationReporter(CLIReporter):
        def __init__(self):
            super(TrialTerminationReporter, self).__init__()
            self.num_terminated = 0

        def should_report(self, trials, done=False):
            """Reports only on trial termination events."""
            old_num_terminated = self.num_terminated
            self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
            return self.num_terminated > old_num_terminated

    tuner = tune.Tuner(my_trainable, run_config=RunConfig(progress_reporter=TrialTerminationReporter()))
    results = tuner.fit()

The default reporting style can also be overridden more broadly by extending the ``ProgressReporter`` interface directly. Note that you can print to any output stream, file etc.

.. code-block:: python

    from ray.tune import ProgressReporter

    class CustomReporter(ProgressReporter):

        def should_report(self, trials, done=False):
            return True

        def report(self, trials, *sys_info):
            print(*sys_info)
            print("\n".join([str(trial) for trial in trials]))

    tuner = tune.Tuner(my_trainable, run_config=RunConfig(progress_reporter=CustomReporter()))
    results = tuner.fit()


.. currentmodule:: ray.tune

Reporter Interface (tune.ProgressReporter)
------------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ProgressReporter

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ProgressReporter.report
    ProgressReporter.should_report


Tune Built-in Reporters
-----------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    CLIReporter
    JupyterNotebookReporter



.. _tune-search-alg:

Tune Search Algorithms (tune.search)
====================================

Tune's Search Algorithms are wrappers around open-source optimization libraries for efficient hyperparameter selection.
Each library has a specific way of defining the search space - please refer to their documentation for more details.
Tune will automatically convert search spaces passed to ``Tuner`` to the library format in most cases.

You can utilize these search algorithms as follows:

.. code-block:: python

    from ray import train, tune
    from ray.train import RunConfig
    from ray.tune.search.optuna import OptunaSearch

    def train_fn(config):
        # This objective function is just for demonstration purposes
        train.report({"loss": config["param"]})

    tuner = tune.Tuner(
        train_fn,
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(),
            num_samples=100,
            metric="loss",
            mode="min",
        ),
        param_space={"param": tune.uniform(0, 1)},
    )
    results = tuner.fit()


Saving and Restoring Tune Search Algorithms
-------------------------------------------

.. TODO: what to do about this section? It doesn't really belong here and is not worth its own guide.
.. TODO: at least check that this pseudo-code runs.

Certain search algorithms have ``save/restore`` implemented,
allowing reuse of searchers that are fitted on the results of multiple tuning runs.

.. code-block:: python

    search_alg = HyperOptSearch()

    tuner_1 = tune.Tuner(
        train_fn,
        tune_config=tune.TuneConfig(search_alg=search_alg)
    )
    results_1 = tuner_1.fit()

    search_alg.save("./my-checkpoint.pkl")

    # Restore the saved state onto another search algorithm,
    # in a new tuning script

    search_alg2 = HyperOptSearch()
    search_alg2.restore("./my-checkpoint.pkl")

    tuner_2 = tune.Tuner(
        train_fn,
        tune_config=tune.TuneConfig(search_alg=search_alg2)
    )
    results_2 = tuner_2.fit()

Tune automatically saves searcher state inside the current experiment folder during tuning.
See ``Result logdir: ...`` in the output logs for this location.

Note that if you have two Tune runs with the same experiment folder,
the previous state checkpoint will be overwritten. You can
avoid this by making sure ``RunConfig(name=...)`` is set to a unique
identifier:

.. code-block:: python

    search_alg = HyperOptSearch()
    tuner_1 = tune.Tuner(
        train_fn,
        tune_config=tune.TuneConfig(
            num_samples=5,
            search_alg=search_alg,
        ),
        run_config=RunConfig(
            name="my-experiment-1",
            storage_path="~/my_results",
        )
    )
    results = tuner_1.fit()

    search_alg2 = HyperOptSearch()
    search_alg2.restore_from_dir(
      os.path.join("~/my_results", "my-experiment-1")
    )

.. _tune-basicvariant:

Random search and grid search (tune.search.basic_variant.BasicVariantGenerator)
-------------------------------------------------------------------------------

The default and most basic way to do hyperparameter search is via random and grid search.
Ray Tune does this through the :class:`BasicVariantGenerator <ray.tune.search.basic_variant.BasicVariantGenerator>`
class that generates trial variants given a search space definition.

The :class:`BasicVariantGenerator <ray.tune.search.basic_variant.BasicVariantGenerator>` is used per
default if no search algorithm is passed to
:func:`Tuner <ray.tune.Tuner>`.

.. currentmodule:: ray.tune.search

.. autosummary::
    :nosignatures:
    :toctree: doc/

    basic_variant.BasicVariantGenerator

.. _tune-ax:

Ax (tune.search.ax.AxSearch)
----------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ax.AxSearch

.. _bayesopt:

Bayesian Optimization (tune.search.bayesopt.BayesOptSearch)
-----------------------------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    bayesopt.BayesOptSearch

.. _suggest-TuneBOHB:

BOHB (tune.search.bohb.TuneBOHB)
--------------------------------

BOHB (Bayesian Optimization HyperBand) is an algorithm that both terminates bad trials
and also uses Bayesian Optimization to improve the hyperparameter search.
It is available from the `HpBandSter library <https://github.com/automl/HpBandSter>`_.

Importantly, BOHB is intended to be paired with a specific scheduler class: :ref:`HyperBandForBOHB <tune-scheduler-bohb>`.

In order to use this search algorithm, you will need to install ``HpBandSter`` and ``ConfigSpace``:

.. code-block:: bash

    $ pip install hpbandster ConfigSpace

See the `BOHB paper <https://arxiv.org/abs/1807.01774>`_ for more details.

.. autosummary::
    :nosignatures:
    :toctree: doc/

    bohb.TuneBOHB

.. _tune-hebo:

HEBO (tune.search.hebo.HEBOSearch)
----------------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    hebo.HEBOSearch

.. _tune-hyperopt:

HyperOpt (tune.search.hyperopt.HyperOptSearch)
----------------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    hyperopt.HyperOptSearch

.. _nevergrad:

Nevergrad (tune.search.nevergrad.NevergradSearch)
-------------------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/
    
    nevergrad.NevergradSearch

.. _tune-optuna:

Optuna (tune.search.optuna.OptunaSearch)
----------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    optuna.OptunaSearch


.. _zoopt:

ZOOpt (tune.search.zoopt.ZOOptSearch)
-------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    zoopt.ZOOptSearch

.. _repeater:

Repeated Evaluations (tune.search.Repeater)
-------------------------------------------

Use ``ray.tune.search.Repeater`` to average over multiple evaluations of the same
hyperparameter configurations. This is useful in cases where the evaluated
training procedure has high variance (i.e., in reinforcement learning).

By default, ``Repeater`` will take in a ``repeat`` parameter and a ``search_alg``.
The ``search_alg`` will suggest new configurations to try, and the ``Repeater``
will run ``repeat`` trials of the configuration. It will then average the
``search_alg.metric`` from the final results of each repeated trial.


.. warning:: It is recommended to not use ``Repeater`` with a TrialScheduler.
    Early termination can negatively affect the average reported metric.

.. autosummary::
    :nosignatures:
    :toctree: doc/

    Repeater

.. _limiter:

ConcurrencyLimiter (tune.search.ConcurrencyLimiter)
---------------------------------------------------

Use ``ray.tune.search.ConcurrencyLimiter`` to limit the amount of concurrency when using a search algorithm.
This is useful when a given optimization algorithm does not parallelize very well (like a naive Bayesian Optimization).

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ConcurrencyLimiter

.. _byo-algo:

Custom Search Algorithms (tune.search.Searcher)
-----------------------------------------------

If you are interested in implementing or contributing a new Search Algorithm, provide the following interface:

.. autosummary::
    :nosignatures:
    :toctree: doc/

    Searcher

.. autosummary::
    :nosignatures:
    :toctree: doc/

    Searcher.suggest
    Searcher.save
    Searcher.restore
    Searcher.on_trial_result
    Searcher.on_trial_complete

If contributing, make sure to add test cases and an entry in the function described below.

.. _shim:

Shim Instantiation (tune.create_searcher)
-----------------------------------------
There is also a shim function that constructs the search algorithm based on the provided string.
This can be useful if the search algorithm you want to use changes often
(e.g., specifying the search algorithm via a CLI option or config file).

.. autosummary::
    :nosignatures:
    :toctree: doc/

    create_searcher


Tune Execution (tune.Tuner)
===========================

.. _tune-run-ref:

Tuner
-----

.. currentmodule:: ray.tune

.. autosummary::
    :nosignatures:
    :toctree: doc/

    Tuner

.. autosummary::
    :nosignatures:
    :toctree: doc/

    Tuner.fit
    Tuner.get_results

Tuner Configuration
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    TuneConfig

.. seealso::

    The `Tuner` constructor also takes in a :class:`RunConfig <ray.train.RunConfig>`.

Restoring a Tuner
~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    Tuner.restore
    Tuner.can_restore


tune.run_experiments
--------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    run_experiments
    Experiment


.. _air-results-ref:
.. _tune-analysis-docs:

.. _result-grid-docstring:

Tune Experiment Results (tune.ResultGrid)
=========================================

ResultGrid (tune.ResultGrid)
----------------------------

.. currentmodule:: ray

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~tune.ResultGrid

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~tune.ResultGrid.get_best_result
    ~tune.ResultGrid.get_dataframe

.. _result-docstring:

Result (train.Result)
---------------------

.. autosummary::
    :nosignatures:
    :template: autosummary/class_without_autosummary.rst

    ~train.Result

.. _exp-analysis-docstring:


ExperimentAnalysis (tune.ExperimentAnalysis)
--------------------------------------------

.. note::

    An `ExperimentAnalysis` is the output of the ``tune.run`` API.
    It's now recommended to use :meth:`Tuner.fit <ray.tune.Tuner.fit>`,
    which outputs a `ResultGrid` object.

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~tune.ExperimentAnalysis


Syncing in Tune (train.SyncConfig)
==================================

.. seealso::

    See :doc:`this user guide </tune/tutorials/tune-storage>` for more details and examples.


.. _tune-sync-config:

Tune Syncing Configuration
--------------------------

.. autosummary::
    :nosignatures:

    ray.train.SyncConfig
        :noindex:


Tune Internals
==============

.. _raytrialexecutor-docstring:

TunerInternal
---------------

.. autoclass:: ray.tune.impl.tuner_internal.TunerInternal
    :members:


.. _trial-docstring:

Trial
-----

.. autoclass:: ray.tune.experiment.trial.Trial
    :members:

FunctionTrainable
-----------------

.. autoclass:: ray.tune.trainable.function_trainable.FunctionTrainable

.. autofunction:: ray.tune.trainable.function_trainable.wrap_function


Registry
--------

.. autofunction:: ray.tune.register_trainable

.. autofunction:: ray.tune.register_env


.. _tune-callbacks-docs:

Tune Callbacks (tune.Callback)
==============================

See :doc:`this user guide </tune/tutorials/tune-metrics>` for more details.

.. seealso::

    :doc:`Tune's built-in loggers </tune/api/logging>` use the ``Callback`` interface.


Callback Interface
------------------

Callback Initialization and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: ray.tune
.. autosummary::
    :nosignatures:
    :toctree: doc/

    Callback

.. autosummary::
    :nosignatures:
    :toctree: doc/

    Callback.setup


Callback Hooks
~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    Callback.on_checkpoint
    Callback.on_experiment_end
    Callback.on_step_begin
    Callback.on_step_end
    Callback.on_trial_complete
    Callback.on_trial_error
    Callback.on_trial_restore
    Callback.on_trial_result
    Callback.on_trial_save
    Callback.on_trial_start


Stateful Callbacks
~~~~~~~~~~~~~~~~~~

The following methods must be overriden for stateful callbacks to be saved/restored
properly by Tune.

.. autosummary::
    :nosignatures:
    :toctree: doc/

    Callback.get_state
    Callback.set_state


.. _loggers-docstring:

Tune Loggers (tune.logger)
==========================

Tune automatically uses loggers for TensorBoard, CSV, and JSON formats.
By default, Tune only logs the returned result dictionaries from the training function.

If you need to log something lower level like model weights or gradients,
see :ref:`Trainable Logging <trainable-logging>`.

.. note::

    Tune's per-trial ``Logger`` classes have been deprecated. Use the ``LoggerCallback`` interface instead.


.. currentmodule:: ray

.. _logger-interface:

LoggerCallback Interface (tune.logger.LoggerCallback)
-----------------------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~tune.logger.LoggerCallback

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~tune.logger.LoggerCallback.log_trial_start
    ~tune.logger.LoggerCallback.log_trial_restore
    ~tune.logger.LoggerCallback.log_trial_save
    ~tune.logger.LoggerCallback.log_trial_result
    ~tune.logger.LoggerCallback.log_trial_end


Tune Built-in Loggers
---------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    tune.logger.JsonLoggerCallback
    tune.logger.CSVLoggerCallback
    tune.logger.TBXLoggerCallback


MLFlow Integration
------------------

Tune also provides a logger for `MLflow <https://mlflow.org>`_.
You can install MLflow via ``pip install mlflow``.
See the :doc:`tutorial here </tune/examples/tune-mlflow>`.

.. autosummary::
    :nosignatures:

    tune.logger.mlflow.MLflowLoggerCallback

Wandb Integration
-----------------

Tune also provides a logger for `Weights & Biases <https://www.wandb.ai/>`_.
You can install Wandb via ``pip install wandb``.
See the :doc:`tutorial here </tune/examples/tune-wandb>`.

.. autosummary::
    :nosignatures:

    tune.logger.wandb.WandbLoggerCallback


Comet Integration
------------------------------

Tune also provides a logger for `Comet <https://www.comet.com/>`_.
You can install Comet via ``pip install comet-ml``.
See the :doc:`tutorial here </tune/examples/tune-comet>`.

.. autosummary::
    :nosignatures:

    tune.logger.comet.CometLoggerCallback

Aim Integration
---------------

Tune also provides a logger for the `Aim <https://aimstack.io/>`_ experiment tracker.
You can install Aim via ``pip install aim``.
See the :doc:`tutorial here </tune/examples/tune-aim>`

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~tune.logger.aim.AimLoggerCallback


Other Integrations
------------------

Viskit
~~~~~~

Tune automatically integrates with `Viskit <https://github.com/vitchyr/viskit>`_ via the ``CSVLoggerCallback`` outputs.
To use VisKit (you may have to install some dependencies), run:

.. code-block:: bash

    $ git clone https://github.com/rll/rllab.git
    $ python rllab/rllab/viskit/frontend.py ~/ray_results/my_experiment

The non-relevant metrics (like timing stats) can be disabled on the left to show only the
relevant ones (like accuracy, loss, etc.).

.. image:: ../images/ray-tune-viskit.png



.. _tune-search-space:

Tune Search Space API
=====================

This section covers the functions you can use to define your search spaces.

.. caution::

    Not all Search Algorithms support all distributions. In particular,
    ``tune.sample_from`` and ``tune.grid_search`` are often unsupported.
    The default :ref:`tune-basicvariant` supports all distributions.

.. tip::

    Avoid passing large objects as values in the search space, as that will incur a performance overhead.
    Use :func:`tune.with_parameters <ray.tune.with_parameters>` to pass large objects in or load them inside your trainable
    from disk (making sure that all nodes have access to the files) or cloud storage.
    See :ref:`tune-bottlenecks` for more information.

For a high-level overview, see this example:

.. TODO: test this

.. code-block :: python

    config = {
        # Sample a float uniformly between -5.0 and -1.0
        "uniform": tune.uniform(-5, -1),

        # Sample a float uniformly between 3.2 and 5.4,
        # rounding to multiples of 0.2
        "quniform": tune.quniform(3.2, 5.4, 0.2),

        # Sample a float uniformly between 0.0001 and 0.01, while
        # sampling in log space
        "loguniform": tune.loguniform(1e-4, 1e-2),

        # Sample a float uniformly between 0.0001 and 0.1, while
        # sampling in log space and rounding to multiples of 0.00005
        "qloguniform": tune.qloguniform(1e-4, 1e-1, 5e-5),

        # Sample a random float from a normal distribution with
        # mean=10 and sd=2
        "randn": tune.randn(10, 2),

        # Sample a random float from a normal distribution with
        # mean=10 and sd=2, rounding to multiples of 0.2
        "qrandn": tune.qrandn(10, 2, 0.2),

        # Sample a integer uniformly between -9 (inclusive) and 15 (exclusive)
        "randint": tune.randint(-9, 15),

        # Sample a random uniformly between -21 (inclusive) and 12 (inclusive (!))
        # rounding to multiples of 3 (includes 12)
        # if q is 1, then randint is called instead with the upper bound exclusive
        "qrandint": tune.qrandint(-21, 12, 3),

        # Sample a integer uniformly between 1 (inclusive) and 10 (exclusive),
        # while sampling in log space
        "lograndint": tune.lograndint(1, 10),

        # Sample a integer uniformly between 1 (inclusive) and 10 (inclusive (!)),
        # while sampling in log space and rounding to multiples of 2
        # if q is 1, then lograndint is called instead with the upper bound exclusive
        "qlograndint": tune.qlograndint(1, 10, 2),

        # Sample an option uniformly from the specified choices
        "choice": tune.choice(["a", "b", "c"]),

        # Sample from a random function, in this case one that
        # depends on another value from the search space
        "func": tune.sample_from(lambda spec: spec.config.uniform * 0.01),

        # Do a grid search over these values. Every value will be sampled
        # ``num_samples`` times (``num_samples`` is the parameter you pass to ``tune.TuneConfig``,
        # which is taken in by ``Tuner``)
        "grid": tune.grid_search([32, 64, 128])
    }

.. currentmodule:: ray

Random Distributions API
------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    tune.uniform
    tune.quniform
    tune.loguniform
    tune.qloguniform
    tune.randn
    tune.qrandn
    tune.randint
    tune.qrandint
    tune.lograndint
    tune.qlograndint
    tune.choice


Grid Search and Custom Function APIs
------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    tune.grid_search
    tune.sample_from

References
----------

See also :ref:`tune-basicvariant`.


Tune CLI (Experimental)
=======================

``tune`` has an easy-to-use command line interface (CLI) to manage and monitor your experiments on Ray.

Here is an example command line call:

``tune list-trials``: List tabular information about trials within an experiment.
Empty columns will be dropped by default. Add the ``--sort`` flag to sort the output by specific columns.
Add the ``--filter`` flag to filter the output in the format ``"<column> <operator> <value>"``.
Add the ``--output`` flag to write the trial information to a specific file (CSV or Pickle).
Add the ``--columns`` and ``--result-columns`` flags to select specific columns to display.

.. code-block:: bash

    $ tune list-trials [EXPERIMENT_DIR] --output note.csv

    +------------------+-----------------------+------------+
    | trainable_name   | experiment_tag        | trial_id   |
    |------------------+-----------------------+------------|
    | MyTrainableClass | 0_height=40,width=37  | 87b54a1d   |
    | MyTrainableClass | 1_height=21,width=70  | 23b89036   |
    | MyTrainableClass | 2_height=99,width=90  | 518dbe95   |
    | MyTrainableClass | 3_height=54,width=21  | 7b99a28a   |
    | MyTrainableClass | 4_height=90,width=69  | ae4e02fb   |
    +------------------+-----------------------+------------+
    Dropped columns: ['status', 'last_update_time']
    Please increase your terminal size to view remaining columns.
    Output saved at: note.csv

    $ tune list-trials [EXPERIMENT_DIR] --filter "trial_id == 7b99a28a"

    +------------------+-----------------------+------------+
    | trainable_name   | experiment_tag        | trial_id   |
    |------------------+-----------------------+------------|
    | MyTrainableClass | 3_height=54,width=21  | 7b99a28a   |
    +------------------+-----------------------+------------+
    Dropped columns: ['status', 'last_update_time']
    Please increase your terminal size to view remaining columns.


.. _tune-integration:

External library integrations for Ray Tune
===========================================

.. currentmodule:: ray

.. _tune-integration-pytorch-lightning:

PyTorch Lightning (tune.integration.pytorch_lightning)
------------------------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~tune.integration.pytorch_lightning.TuneReportCheckpointCallback

.. _tune-integration-xgboost:

XGBoost (tune.integration.xgboost)
----------------------------------

.. autosummary::
    :nosignatures:
    :template: autosummary/class_without_autosummary.rst
    :toctree: doc/

    ~tune.integration.xgboost.TuneReportCheckpointCallback


.. _tune-integration-lightgbm:

LightGBM (tune.integration.lightgbm)
------------------------------------

.. autosummary::
    :nosignatures:
    :template: autosummary/class_without_autosummary.rst
    :toctree: doc/

    ~tune.integration.lightgbm.TuneReportCheckpointCallback


.. _tune-schedulers:

Tune Trial Schedulers (tune.schedulers)
=======================================

In Tune, some hyperparameter optimization algorithms are written as "scheduling algorithms".
These Trial Schedulers can early terminate bad trials, pause trials, clone trials,
and alter hyperparameters of a running trial.

All Trial Schedulers take in a ``metric``, which is a value returned in the result dict of your
Trainable and is maximized or minimized according to ``mode``.

.. code-block:: python

    from ray import train, tune
    from tune.schedulers import ASHAScheduler

    def train_fn(config):
        # This objective function is just for demonstration purposes
        train.report({"loss": config["param"]})

    tuner = tune.Tuner(
        train_fn,
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(),
            metric="loss",
            mode="min",
            num_samples=10,
        ),
        param_space={"param": tune.uniform(0, 1)},
    )
    results = tuner.fit()

.. currentmodule:: ray.tune.schedulers

.. _tune-scheduler-hyperband:

ASHA (tune.schedulers.ASHAScheduler)
------------------------------------

The `ASHA <https://openreview.net/forum?id=S1Y7OOlRZ>`__ scheduler can be used by
setting the ``scheduler`` parameter of ``tune.TuneConfig``, which is taken in by ``Tuner``, e.g.

.. code-block:: python

    from ray import tune
    from tune.schedulers import ASHAScheduler

    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='loss',
        mode='min',
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1,
    )
    tuner = tune.Tuner(
        train_fn,
        tune_config=tune.TuneConfig(scheduler=asha_scheduler),
    )
    results = tuner.fit()

Compared to the original version of HyperBand, this implementation provides better
parallelism and avoids straggler issues during eliminations.
**We recommend using this over the standard HyperBand scheduler.**
An example of this can be found here: :doc:`/tune/examples/includes/async_hyperband_example`.

Even though the original paper mentions a bracket count of 3, discussions with the authors concluded
that the value should be left to 1 bracket.
This is the default used if no value is provided for the ``brackets`` argument.

.. autosummary::
    :nosignatures:
    :toctree: doc/
    :template: autosummary/class_without_autosummary.rst

    AsyncHyperBandScheduler
    ASHAScheduler

.. _tune-original-hyperband:

HyperBand (tune.schedulers.HyperBandScheduler)
----------------------------------------------

Tune implements the `standard version of HyperBand <https://arxiv.org/abs/1603.06560>`__.
**We recommend using the ASHA Scheduler over the standard HyperBand scheduler.**

.. autosummary::
    :nosignatures:
    :toctree: doc/

    HyperBandScheduler


HyperBand Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation details may deviate slightly from theory but are focused on increasing usability.
Note: ``R``, ``s_max``, and ``eta`` are parameters of HyperBand given by the paper.
See `this post <https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/>`_ for context.

1. Both ``s_max`` (representing the ``number of brackets - 1``) and ``eta``, representing the downsampling rate, are fixed.
    In many practical settings, ``R``, which represents some resource unit and often the number of training iterations,
    can be set reasonably large, like ``R >= 200``.
    For simplicity, assume ``eta = 3``. Varying ``R`` between ``R = 200`` and ``R = 1000``
    creates a huge range of the number of trials needed to fill up all brackets.

.. image:: /images/hyperband_bracket.png

On the other hand, holding ``R`` constant at ``R = 300`` and varying ``eta`` also leads to
HyperBand configurations that are not very intuitive:

.. image:: /images/hyperband_eta.png

The implementation takes the same configuration as the example given in the paper
and exposes ``max_t``, which is not a parameter in the paper.

2. The example in the `post <https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/>`_ to calculate ``n_0``
    is actually a little different than the algorithm given in the paper.
    In this implementation, we implement ``n_0`` according to the paper (which is `n` in the below example):

.. image:: /images/hyperband_allocation.png


3. There are also implementation specific details like how trials are placed into brackets which are not covered in the paper.
    This implementation places trials within brackets according to smaller bracket first - meaning
    that with low number of trials, there will be less early stopping.

.. _tune-scheduler-msr:

Median Stopping Rule (tune.schedulers.MedianStoppingRule)
---------------------------------------------------------

The Median Stopping Rule implements the simple strategy of stopping a trial if its performance falls
below the median of other trials at similar points in time.

.. autosummary::
    :nosignatures:
    :toctree: doc/

    MedianStoppingRule

.. _tune-scheduler-pbt:

Population Based Training (tune.schedulers.PopulationBasedTraining)
-------------------------------------------------------------------

Tune includes a distributed implementation of `Population Based Training (PBT) <https://www.deepmind.com/blog/population-based-training-of-neural-networks>`__.
This can be enabled by setting the ``scheduler`` parameter of ``tune.TuneConfig``, which is taken in by ``Tuner``, e.g.

.. code-block:: python

    from ray import tune
    from ray.tune.schedulers import PopulationBasedTraining

    pbt_scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        metric='loss',
        mode='min',
        perturbation_interval=1,
        hyperparam_mutations={
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "alpha": tune.uniform(0.0, 1.0),
        }
    )
    tuner = tune.Tuner(
        train_fn,
        tune_config=tune.TuneConfig(
            num_samples=4,
            scheduler=pbt_scheduler,
        ),
    )
    tuner.fit()

When the PBT scheduler is enabled, each trial variant is treated as a member of the population.
Periodically, **top-performing trials are checkpointed**
(this requires your Trainable to support :ref:`save and restore <tune-trial-checkpoint>`).
**Low-performing trials clone the hyperparameter configurations of top performers and
perturb them** slightly in the hopes of discovering even better hyperparameter settings.
**Low-performing trials also resume from the checkpoints of the top performers**, allowing
the trials to explore the new hyperparameter configuration starting from a partially
trained model (e.g. by copying model weights from one of the top-performing trials).

Take a look at :doc:`/tune/examples/pbt_visualization/pbt_visualization` to get an idea
of how PBT operates. :doc:`/tune/examples/pbt_guide` gives more examples
of PBT usage.

.. autosummary::
    :nosignatures:
    :toctree: doc/

    PopulationBasedTraining


.. _tune-scheduler-pbt-replay:

Population Based Training Replay (tune.schedulers.PopulationBasedTrainingReplay)
--------------------------------------------------------------------------------

Tune includes a utility to replay hyperparameter schedules of Population Based Training runs.
You just specify an existing experiment directory and the ID of the trial you would
like to replay. The scheduler accepts only one trial, and it will update its
config according to the obtained schedule.

.. code-block:: python

    from ray import tune
    from ray.tune.schedulers import PopulationBasedTrainingReplay

    replay = PopulationBasedTrainingReplay(
        experiment_dir="~/ray_results/pbt_experiment/",
        trial_id="XXXXX_00001"
    )
    tuner = tune.Tuner(
        train_fn,
        tune_config=tune.TuneConfig(scheduler=replay)
    )
    results = tuner.fit()

See :ref:`here for an example <tune-advanced-tutorial-pbt-replay>` on how to use the
replay utility in practice.

.. autosummary::
    :nosignatures:
    :toctree: doc/

    PopulationBasedTrainingReplay

.. _tune-scheduler-pb2:

Population Based Bandits (PB2) (tune.schedulers.pb2.PB2)
--------------------------------------------------------

Tune includes a distributed implementation of `Population Based Bandits (PB2) <https://arxiv.org/abs/2002.02518>`__.
This algorithm builds upon PBT, with the main difference being that instead of using random perturbations,
PB2 selects new hyperparameter configurations using a Gaussian Process model.

The Tune implementation of PB2 requires GPy and sklearn to be installed:

.. code-block:: bash

    pip install GPy scikit-learn


PB2 can be enabled by setting the ``scheduler`` parameter of ``tune.TuneConfig`` which is taken in by ``Tuner``, e.g.:

.. code-block:: python

    from ray.tune.schedulers.pb2 import PB2

    pb2_scheduler = PB2(
        time_attr='time_total_s',
        metric='mean_accuracy',
        mode='max',
        perturbation_interval=600.0,
        hyperparam_bounds={
            "lr": [1e-3, 1e-5],
            "alpha": [0.0, 1.0],
        ...
        }
    )
    tuner = tune.Tuner( ... , tune_config=tune.TuneConfig(scheduler=pb2_scheduler))
    results = tuner.fit()


When the PB2 scheduler is enabled, each trial variant is treated as a member of the population.
Periodically, top-performing trials are checkpointed (this requires your Trainable to
support :ref:`save and restore <tune-trial-checkpoint>`).
Low-performing trials clone the checkpoints of top performers and perturb the configurations
in the hope of discovering an even better variation.

The primary motivation for PB2 is the ability to find promising hyperparamters with only a small population size.
With that in mind, you can run this :doc:`PB2 PPO example </tune/examples/includes/pb2_ppo_example>` to compare PB2 vs. PBT,
with a population size of ``4`` (as in the paper).
The example uses the ``BipedalWalker`` environment so does not require any additional licenses.

.. autosummary::
    :nosignatures:
    :toctree: doc/

    pb2.PB2


.. _tune-scheduler-bohb:

BOHB (tune.schedulers.HyperBandForBOHB)
---------------------------------------

This class is a variant of HyperBand that enables the `BOHB Algorithm <https://arxiv.org/abs/1807.01774>`_.
This implementation is true to the original HyperBand implementation and does not implement pipelining nor
straggler mitigation.

This is to be used in conjunction with the Tune BOHB search algorithm.
See :ref:`TuneBOHB <suggest-TuneBOHB>` for package requirements, examples, and details.

An example of this in use can be found here: :doc:`/tune/examples/includes/bohb_example`.


.. autosummary::
    :nosignatures:
    :toctree: doc/

    HyperBandForBOHB

.. _tune-resource-changing-scheduler:

ResourceChangingScheduler
-------------------------

This class is a utility scheduler, allowing for trial resource requirements to be changed during tuning.
It wraps around another scheduler and uses its decisions.

* If you are using the Trainable (class) API for tuning, your Trainable must implement ``Trainable.update_resources``,
    which will let your model know about the new resources assigned. You can also obtain the current trial resources
    by calling ``Trainable.trial_resources``.

* If you are using the functional API for tuning, get the current trial resources obtained by calling
    `tune.get_trial_resources()` inside the training function.
    The function should be able to :ref:`load and save checkpoints <tune-function-trainable-checkpointing>`
    (the latter preferably every iteration).

An example of this in use can be found here: :doc:`/tune/examples/includes/xgboost_dynamic_resources_example`.

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ResourceChangingScheduler
    resource_changing_scheduler.DistributeResources
    resource_changing_scheduler.DistributeResourcesToTopJob

FIFOScheduler (Default Scheduler)
---------------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    FIFOScheduler

TrialScheduler Interface
------------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    TrialScheduler

.. autosummary::
    :nosignatures:
    :toctree: doc/

    TrialScheduler.choose_trial_to_run
    TrialScheduler.on_trial_result
    TrialScheduler.on_trial_complete


Shim Instantiation (tune.create_scheduler)
------------------------------------------

There is also a shim function that constructs the scheduler based on the provided string.
This can be useful if the scheduler you want to use changes often (e.g., specifying the scheduler
via a CLI option or config file).

.. autosummary::
    :nosignatures:
    :toctree: doc/

    create_scheduler


.. _trainable-docs:

.. TODO: these "basic" sections before the actual API docs start don't really belong here. Then again, the function
    API does not really have a signature to just describe.
.. TODO: Reusing actors and advanced resources allocation seem ill-placed.

Training in Tune (tune.Trainable, train.report)
=================================================

Training can be done with either a **Function API** (:func:`train.report() <ray.train.report>`) or
**Class API** (:ref:`tune.Trainable <tune-trainable-docstring>`).

For the sake of example, let's maximize this objective function:

.. literalinclude:: /tune/doc_code/trainable.py
    :language: python
    :start-after: __example_objective_start__
    :end-before: __example_objective_end__

.. _tune-function-api:

Function Trainable API
----------------------

Use the Function API to define a custom training function that Tune runs in Ray actor processes. Each trial is placed
into a Ray actor process and runs in parallel.

The ``config`` argument in the function is a dictionary populated automatically by Ray Tune and corresponding to
the hyperparameters selected for the trial from the :ref:`search space <tune-key-concepts-search-spaces>`.

With the Function API, you can report intermediate metrics by simply calling :func:`train.report() <ray.train.report>` within the function.

.. literalinclude:: /tune/doc_code/trainable.py
    :language: python
    :start-after: __function_api_report_intermediate_metrics_start__
    :end-before: __function_api_report_intermediate_metrics_end__

.. tip:: Do not use :func:`train.report() <ray.train.report>` within a ``Trainable`` class.

In the previous example, we reported on every step, but this metric reporting frequency
is configurable. For example, we could also report only a single time at the end with the final score:

.. literalinclude:: /tune/doc_code/trainable.py
    :language: python
    :start-after: __function_api_report_final_metrics_start__
    :end-before: __function_api_report_final_metrics_end__

It's also possible to return a final set of metrics to Tune by returning them from your function:

.. literalinclude:: /tune/doc_code/trainable.py
    :language: python
    :start-after: __function_api_return_final_metrics_start__
    :end-before: __function_api_return_final_metrics_end__

Note that Ray Tune outputs extra values in addition to the user reported metrics,
such as ``iterations_since_restore``. See :ref:`tune-autofilled-metrics` for an explanation of these values.

See how to configure checkpointing for a function trainable :ref:`here <tune-function-trainable-checkpointing>`.

.. _tune-class-api:

Class Trainable API
--------------------------

.. caution:: Do not use :func:`train.report() <ray.train.report>` within a ``Trainable`` class.

The Trainable **class API** will require users to subclass ``ray.tune.Trainable``. Here's a naive example of this API:

.. literalinclude:: /tune/doc_code/trainable.py
    :language: python
    :start-after: __class_api_example_start__
    :end-before: __class_api_example_end__

As a subclass of ``tune.Trainable``, Tune will create a ``Trainable`` object on a
separate process (using the :ref:`Ray Actor API <actor-guide>`).

  1. ``setup`` function is invoked once training starts.
  2. ``step`` is invoked **multiple times**.
     Each time, the Trainable object executes one logical iteration of training in the tuning process,
     which may include one or more iterations of actual training.
  3. ``cleanup`` is invoked when training is finished.

The ``config`` argument in the ``setup`` method is a dictionary populated automatically by Tune and corresponding to
the hyperparameters selected for the trial from the :ref:`search space <tune-key-concepts-search-spaces>`.

.. tip:: As a rule of thumb, the execution time of ``step`` should be large enough to avoid overheads
    (i.e. more than a few seconds), but short enough to report progress periodically (i.e. at most a few minutes).

You'll notice that Ray Tune will output extra values in addition to the user reported metrics,
such as ``iterations_since_restore``.
See :ref:`tune-autofilled-metrics` for an explanation/glossary of these values.

See how to configure checkpoint for class trainable :ref:`here <tune-class-trainable-checkpointing>`.


Advanced: Reusing Actors in Tune
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: This feature is only for the Trainable Class API.

Your Trainable can often take a long time to start.
To avoid this, you can do ``tune.TuneConfig(reuse_actors=True)`` (which is taken in by ``Tuner``) to reuse the same Trainable Python process and
object for multiple hyperparameters.

This requires you to implement ``Trainable.reset_config``, which provides a new set of hyperparameters.
It is up to the user to correctly update the hyperparameters of your trainable.

.. code-block:: python

    class PytorchTrainable(tune.Trainable):
        """Train a Pytorch ConvNet."""

        def setup(self, config):
            self.train_loader, self.test_loader = get_data_loaders()
            self.model = ConvNet()
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.get("lr", 0.01),
                momentum=config.get("momentum", 0.9))

        def reset_config(self, new_config):
            for param_group in self.optimizer.param_groups:
                if "lr" in new_config:
                    param_group["lr"] = new_config["lr"]
                if "momentum" in new_config:
                    param_group["momentum"] = new_config["momentum"]

            self.model = ConvNet()
            self.config = new_config
            return True


Comparing Tune's Function API and Class API
-------------------------------------------

Here are a few key concepts and what they look like for the Function and Class API's.

======================= =============================================== ==============================================
Concept                 Function API                                    Class API
======================= =============================================== ==============================================
Training Iteration      Increments on each `train.report` call          Increments on each `Trainable.step` call
Report  metrics         `train.report(metrics)`                         Return metrics from `Trainable.step`
Saving a checkpoint     `train.report(..., checkpoint=checkpoint)`      `Trainable.save_checkpoint`
Loading a checkpoint    `train.get_checkpoint()`                        `Trainable.load_checkpoint`
Accessing config        Passed as an argument `def train_func(config):` Passed through `Trainable.setup`
======================= =============================================== ==============================================


Advanced Resource Allocation
----------------------------

Trainables can themselves be distributed. If your trainable function / class creates further Ray actors or tasks
that also consume CPU / GPU resources, you will want to add more bundles to the :class:`PlacementGroupFactory`
to reserve extra resource slots.
For example, if a trainable class requires 1 GPU itself, but also launches 4 actors, each using another GPU,
then you should use :func:`tune.with_resources <ray.tune.with_resources>` like this:

.. code-block:: python
   :emphasize-lines: 4-10

    tuner = tune.Tuner(
        tune.with_resources(my_trainable, tune.PlacementGroupFactory([
            {"CPU": 1, "GPU": 1},
            {"GPU": 1},
            {"GPU": 1},
            {"GPU": 1},
            {"GPU": 1}
        ])),
        run_config=RunConfig(name="my_trainable")
    )

The ``Trainable`` also provides the ``default_resource_requests`` interface to automatically
declare the resources per trial based on the given configuration.

It is also possible to specify memory (``"memory"``, in bytes) and custom resource requirements.

.. currentmodule:: ray

Function API
------------
For reporting results and checkpoints with the function API,
see the :ref:`Ray Train utilities <train-loop-api>` documentation.

.. _tune-trainable-docstring:

Trainable (Class API)
---------------------

Constructor
~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~tune.Trainable


Trainable Methods to Implement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ~tune.Trainable.setup
    ~tune.Trainable.save_checkpoint
    ~tune.Trainable.load_checkpoint
    ~tune.Trainable.step
    ~tune.Trainable.reset_config
    ~tune.Trainable.cleanup
    ~tune.Trainable.default_resource_request


.. _tune-util-ref:

Tune Trainable Utilities
-------------------------

Tune Data Ingestion Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    tune.with_parameters


Tune Resource Assignment Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    tune.with_resources
    ~tune.execution.placement_groups.PlacementGroupFactory
    tune.utils.wait_for_gpu


Tune Trainable Debugging Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: doc/

    tune.utils.diagnose_serialization
    tune.utils.validate_save_restore




.. _tune-env-vars:

Environment variables used by Ray Tune
--------------------------------------

Some of Ray Tune's behavior can be configured using environment variables.
These are the environment variables Ray Tune currently considers:

* **TUNE_DISABLE_AUTO_CALLBACK_LOGGERS**: Ray Tune automatically adds a CSV and
  JSON logger callback if they haven't been passed. Setting this variable to
  `1` disables this automatic creation. Please note that this will most likely
  affect analyzing your results after the tuning run.
* **TUNE_DISABLE_AUTO_INIT**: Disable automatically calling ``ray.init()`` if
  not attached to a Ray session.
* **TUNE_DISABLE_DATED_SUBDIR**: Ray Tune automatically adds a date string to experiment
  directories when the name is not specified explicitly or the trainable isn't passed
  as a string. Setting this environment variable to ``1`` disables adding these date strings.
* **TUNE_DISABLE_STRICT_METRIC_CHECKING**: When you report metrics to Tune via
  ``session.report()`` and passed a ``metric`` parameter to ``Tuner()``, a scheduler,
  or a search algorithm, Tune will error
  if the metric was not reported in the result. Setting this environment variable
  to ``1`` will disable this check.
* **TUNE_DISABLE_SIGINT_HANDLER**: Ray Tune catches SIGINT signals (e.g. sent by
  Ctrl+C) to gracefully shutdown and do a final checkpoint. Setting this variable
  to ``1`` will disable signal handling and stop execution right away. Defaults to
  ``0``.
* **TUNE_FORCE_TRIAL_CLEANUP_S**: By default, Ray Tune will gracefully terminate trials,
  letting them finish the current training step and any user-defined cleanup.
  Setting this variable to a non-zero, positive integer will cause trials to be forcefully
  terminated after a grace period of that many seconds. Defaults to ``600`` (seconds).
* **TUNE_FUNCTION_THREAD_TIMEOUT_S**: Time in seconds the function API waits
  for threads to finish after instructing them to complete. Defaults to ``2``.
* **TUNE_GLOBAL_CHECKPOINT_S**: Time in seconds that limits how often
  experiment state is checkpointed. If not, set this will default to ``'auto'``.
  ``'auto'`` measures the time it takes to snapshot the experiment state
  and adjusts the period so that ~5% of the driver's time is spent on snapshotting.
  You should set this to a fixed value (ex: ``TUNE_GLOBAL_CHECKPOINT_S=60``)
  to snapshot your experiment state every X seconds.
* **TUNE_MAX_LEN_IDENTIFIER**: Maximum length of trial subdirectory names (those
  with the parameter values in them)
* **TUNE_MAX_PENDING_TRIALS_PG**: Maximum number of pending trials when placement groups are used. Defaults
  to ``auto``, which will be updated to ``max(200, cluster_cpus * 1.1)`` for random/grid search and ``1``
  for any other search algorithms.
* **TUNE_PLACEMENT_GROUP_PREFIX**: Prefix for placement groups created by Ray Tune. This prefix is used
  e.g. to identify placement groups that should be cleaned up on start/stop of the tuning run. This is
  initialized to a unique name at the start of the first run.
* **TUNE_PLACEMENT_GROUP_RECON_INTERVAL**: How often to reconcile placement groups. Reconcilation is
  used to make sure that the number of requested placement groups and pending/running trials are in sync.
  In normal circumstances these shouldn't differ anyway, but reconcilation makes sure to capture cases when
  placement groups are manually destroyed. Reconcilation doesn't take much time, but it can add up when
  running a large number of short trials. Defaults to every ``5`` (seconds).
* **TUNE_PRINT_ALL_TRIAL_ERRORS**: If ``1``, will print all trial errors as they come up. Otherwise, errors
  will only be saved as text files to the trial directory and not printed. Defaults to ``1``.
* **TUNE_RESULT_BUFFER_LENGTH**: Ray Tune can buffer results from trainables before they are passed
  to the driver. Enabling this might delay scheduling decisions, as trainables are speculatively
  continued. Setting this to ``1`` disables result buffering. Cannot be used with ``checkpoint_at_end``.
  Defaults to disabled.
* **TUNE_RESULT_DELIM**: Delimiter used for nested entries in
  :class:`ExperimentAnalysis <ray.tune.ExperimentAnalysis>` dataframes. Defaults to ``.`` (but will be
  changed to ``/`` in future versions of Ray).
* **TUNE_RESULT_BUFFER_MAX_TIME_S**: Similarly, Ray Tune buffers results up to ``number_of_trial/10`` seconds,
  but never longer than this value. Defaults to 100 (seconds).
* **TUNE_RESULT_BUFFER_MIN_TIME_S**: Additionally, you can specify a minimum time to buffer results. Defaults to 0.
* **TUNE_WARN_THRESHOLD_S**: Threshold for logging if an Tune event loop operation takes too long. Defaults to 0.5 (seconds).
* **TUNE_WARN_INSUFFICENT_RESOURCE_THRESHOLD_S**: Threshold for throwing a warning if no active trials are in ``RUNNING`` state
  for this amount of seconds. If the Ray Tune job is stuck in this state (most likely due to insufficient resources),
  the warning message is printed repeatedly every this amount of seconds. Defaults to 60 (seconds).
* **TUNE_WARN_INSUFFICENT_RESOURCE_THRESHOLD_S_AUTOSCALER**: Threshold for throwing a warning when the autoscaler is enabled and
  if no active trials are in ``RUNNING`` state for this amount of seconds.
  If the Ray Tune job is stuck in this state (most likely due to insufficient resources), the warning message is printed
  repeatedly every this amount of seconds. Defaults to 60 (seconds).
* **TUNE_WARN_SLOW_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S**: Threshold for logging a warning if the experiment state syncing
  takes longer than this time in seconds. The experiment state files should be very lightweight, so this should not take longer than ~5 seconds.
  Defaults to 5 (seconds).
* **TUNE_STATE_REFRESH_PERIOD**: Frequency of updating the resource tracking from Ray. Defaults to 10 (seconds).
* **TUNE_RESTORE_RETRY_NUM**: The number of retries that are done before a particular trial's restore is determined
  unsuccessful. After that, the trial is not restored to its previous checkpoint but rather from scratch.
  Default is ``0``. While this retry counter is taking effect, per trial failure number will not be incremented, which
  is compared against ``max_failures``.
* **RAY_AIR_FULL_TRACEBACKS**: If set to 1, will print full tracebacks for training functions,
  including internal code paths. Otherwise, abbreviated tracebacks that only show user code
  are printed. Defaults to 0 (disabled).
* **RAY_AIR_NEW_OUTPUT**: If set to 0, this disables
  the `experimental new console output <https://github.com/ray-project/ray/issues/36949>`_.



There are some environment variables that are mostly relevant for integrated libraries:

* **WANDB_API_KEY**: Weights and Biases API key. You can also use ``wandb login``
  instead.


.. _tune-stoppers:

Tune Stopping Mechanisms (tune.stopper)
=======================================

In addition to Trial Schedulers like :ref:`ASHA <tune-scheduler-hyperband>`, where a number of
trials are stopped if they perform subpar, Ray Tune also supports custom stopping mechanisms to stop trials early. They can also stop the entire experiment after a condition is met.
For instance, stopping mechanisms can specify to stop trials when they reached a plateau and the metric
doesn't change anymore.

Ray Tune comes with several stopping mechanisms out of the box. For custom stopping behavior, you can
inherit from the :class:`Stopper <ray.tune.Stopper>` class.

Other stopping behaviors are described :ref:`in the user guide <tune-stopping-ref>`.


.. _tune-stop-ref:

Stopper Interface (tune.Stopper)
--------------------------------

.. currentmodule:: ray.tune.stopper

.. autosummary::
    :nosignatures:
    :toctree: doc/

    Stopper

.. autosummary::
    :nosignatures:
    :toctree: doc/

    Stopper.__call__
    Stopper.stop_all

Tune Built-in Stoppers
----------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    MaximumIterationStopper
    ExperimentPlateauStopper
    TrialPlateauStopper
    TimeoutStopper
    CombinedStopper


Tune Hyperparameter Optimization Framework Examples
---------------------------------------------------

.. toctree::
    :hidden:

    Ax Example <ax_example>
    HyperOpt Example <hyperopt_example>
    Bayesopt Example <bayesopt_example>
    BOHB Example <bohb_example>
    Nevergrad Example <nevergrad_example>
    Optuna Example <optuna_example>


Tune integrates with a wide variety of hyperparameter optimization frameworks
and their respective search algorithms. Here you can find detailed examples
on each of our integrations:

.. grid:: 1 2 3 4
    :gutter: 1
    :class-container: container pb-3

    .. grid-item-card::
        :img-top: ../images/ax.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: ax_example

            How To Use Tune With Ax

    .. grid-item-card::
        :img-top: ../images/hyperopt.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: hyperopt_example

            How To Use Tune With HyperOpt

    .. grid-item-card::
        :img-top: ../images/bayesopt.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: bayesopt_example

            How To Use Tune With BayesOpt

    .. grid-item-card::
        :img-top: ../images/bohb.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: bohb_example

            How To Use Tune With TuneBOHB

    .. grid-item-card::
        :img-top: ../images/nevergrad.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: nevergrad_example

            How To Use Tune With Nevergrad

    .. grid-item-card::
        :img-top: ../images/optuna.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: optuna_example

            How To Use Tune With Optuna


Other Examples
--------------

- :doc:`/tune/examples/includes/tune_basic_example`: Simple example for doing a basic random and grid search.
- :doc:`/tune/examples/includes/async_hyperband_example`: Example of using a simple tuning function with
  AsyncHyperBandScheduler.
- :doc:`/tune/examples/includes/hyperband_function_example`:
  Example of using a Trainable function with HyperBandScheduler.
  Also uses the AsyncHyperBandScheduler.
- :doc:`/tune/examples/pbt_visualization/pbt_visualization`:
  Configuring and running (synchronous) PBT and understanding the underlying algorithm behavior with a simple example.
- :doc:`/tune/examples/includes/pbt_function`:
  Example of using the function API with a PopulationBasedTraining scheduler.
- :doc:`/tune/examples/includes/pb2_example`: Example of using the Population-based Bandits (PB2) scheduler.
- :doc:`/tune/examples/includes/logging_example`: Example of custom loggers and custom trial directory naming.


Tune Exercises
--------------

Learn how to use Tune in your browser with the following Colab-based exercises.

.. raw:: html

    <table>
      <tr>
        <th class="tune-colab">Exercise Description</th>
        <th class="tune-colab">Library</th>
        <th class="tune-colab">Colab Link</th>
      </tr>
      <tr>
        <td class="tune-colab">Basics of using Tune.</td>
        <td class="tune-colab">TF/Keras</td>
        <td class="tune-colab">
          <a href="https://colab.research.google.com/github/ray-project/tutorial/blob/master/tune_exercises/exercise_1_basics.ipynb" target="_parent">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Tune Tutorial"/>
          </a>
        </td>
      </tr>

      <tr>
        <td class="tune-colab">Using Search algorithms and Trial Schedulers to optimize your model.</td>
        <td class="tune-colab">Pytorch</td>
        <td class="tune-colab">
          <a href="https://colab.research.google.com/github/ray-project/tutorial/blob/master/tune_exercises/exercise_2_optimize.ipynb" target="_parent">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Tune Tutorial"/>
          </a>
        </td>
      </tr>

      <tr>
        <td class="tune-colab">Using Population-Based Training (PBT).</td>
        <td class="tune-colab">Pytorch</td>
        <td class="tune-colab">
          <a href="https://colab.research.google.com/github/ray-project/tutorial/blob/master/tune_exercises/exercise_3_pbt.ipynb" target="_parent">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Tune Tutorial"/>
          </a>
        </td>
      </tr>

      <tr>
        <td class="tune-colab">Fine-tuning Huggingface Transformers with PBT.</td>
        <td class="tune-colab">Huggingface Transformers/Pytorch</td>
        <td class="tune-colab">
          <a href="https://colab.research.google.com/drive/1tQgAKgcKQzheoh503OzhS4N9NtfFgmjF?usp=sharing" target="_parent">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Tune Tutorial"/>
          </a>
        </td>
      </tr>

      <tr>
        <td class="tune-colab">Logging Tune Runs to Comet ML.</td>
        <td class="tune-colab">Comet</td>
        <td class="tune-colab">
          <a href="https://colab.research.google.com/drive/1dp3VwVoAH1acn_kG7RuT62mICnOqxU1z?usp=sharing" target="_parent">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Tune Tutorial"/>
          </a>
        </td>
      </tr>
    </table>


Tutorial source files `can be found here <https://github.com/ray-project/tutorial>`_.


.. _tune-examples-ref:
.. _tune-recipes:

=================
Ray Tune Examples
=================

.. toctree::
    :hidden:

    ml-frameworks
    experiment-tracking
    hpo-frameworks
    Other Examples <other-examples>
    Exercises <exercises>


.. tip:: Check out :ref:`the Tune User Guides <tune-guides>` To learn more about Tune's features in depth.


Tune Experiment Tracking Examples
---------------------------------

.. toctree::
    :hidden:

    Weights & Biases Example <tune-wandb>
    MLflow Example <tune-mlflow>
    Aim Example <tune-aim>
    Comet Example <tune-comet>


Ray Tune integrates with some popular Experiment tracking and management tools,
such as CometML, or Weights & Biases. If you're interested in learning how
to use Ray Tune with Tensorboard, you can find more information in our
:ref:`Guide to logging and outputs <tune-logging>`.

.. grid:: 1 2 3 4
    :gutter: 1
    :class-container: container pb-3


    .. grid-item-card::
        :img-top:  /images/aim_logo.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-aim-ref

            Using Aim with Ray Tune For Experiment Management

    .. grid-item-card::
        :img-top: /images/comet_logo_full.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-comet-ref

            Using Comet with Ray Tune For Experiment Management

    .. grid-item-card::
        :img-top: /images/wandb_logo.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-wandb-ref

            Tracking Your Experiment Process Weights & Biases

    .. grid-item-card::
        :img-top: /images/mlflow.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-mlflow-ref

            Using MLflow Tracking & AutoLogging with Tune


Examples using Ray Tune with ML Frameworks
------------------------------------------

.. toctree::
    :hidden:

    Keras Example <tune_mnist_keras>
    PyTorch Example <tune-pytorch-cifar>
    PyTorch Lightning Example <tune-pytorch-lightning>
    Ray RLlib Example <pbt_ppo_example>
    XGBoost Example <tune-xgboost>
    LightGBM Example <lightgbm_example>
    Horovod Example <horovod_simple>
    Hugging Face Transformers Example <pbt_transformers>


Ray Tune integrates with many popular machine learning frameworks.
Here you find a few practical examples showing you how to tune your models.
At the end of these guides you will often find links to even more examples.

.. grid:: 1 2 3 4
    :gutter: 1
    :class-container: container pb-3

    .. grid-item-card::
        :img-top: /images/keras.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-mnist-keras

            How To Use Tune With Keras & TF Models

    .. grid-item-card::
        :img-top: /images/pytorch_logo.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-pytorch-cifar-ref

            How To Use Tune With PyTorch Models

    .. grid-item-card::
        :img-top: /images/pytorch_lightning_small.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-pytorch-lightning-ref

            How To Tune PyTorch Lightning Models

    .. grid-item-card::
        :img-top: /rllib/images/rllib-logo.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-rllib-example

            Tuning RL Experiments With Ray Tune & Ray Serve

    .. grid-item-card::
        :img-top: /images/xgboost_logo.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-xgboost-ref

            A Guide To Tuning XGBoost Parameters With Tune

    .. grid-item-card::
        :img-top: /images/lightgbm_logo.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-lightgbm-example

            A Guide To Tuning LightGBM Parameters With Tune

    .. grid-item-card::
        :img-top: /images/horovod.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-horovod-example

            A Guide To Tuning Horovod Parameters With Tune

    .. grid-item-card::
        :img-top: /images/hugging.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune-huggingface-example

            A Guide To Tuning Huggingface Transformers With Tune

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune_train_tf_example

            End-to-end Example for Tuning a TensorFlow Model

    .. grid-item-card::
        :img-top: /images/tune.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: tune_train_torch_example

            End-to-end Example for Tuning a PyTorch Model with PBT



:orphan:

PBT Visualization Helper File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used in :doc:`/tune/examples/pbt_visualization/pbt_visualization`.

.. literalinclude:: ./pbt_visualization_utils.py


:orphan:

Nevergrad Example
~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/nevergrad_example.py

:orphan:

MNIST PyTorch Example
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/mnist_pytorch.py

If you consider switching to PyTorch Lightning to get rid of some of your boilerplate
training code, please know that we also have a walkthrough on :doc:`how to use Tune with
PyTorch Lightning models </tune/examples/tune-pytorch-lightning>`.

:orphan:

MNIST PyTorch Lightning Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/mnist_ptl_mini.py


:orphan:

HyperBand Example
=================

.. literalinclude:: /../../python/ray/tune/examples/hyperband_example.py

:orphan:

MNIST PyTorch Trainable Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/mnist_pytorch_trainable.py


:orphan:

HyperBand Function Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/hyperband_function_example.py


:orphan:

PB2 PPO Example
~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/pb2_ppo_example.py


:orphan:

Hyperopt Conditional Search Space Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/hyperopt_conditional_search_space_example.py


:orphan:

PBT ConvNet Example
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/pbt_convnet_function_example.py


:orphan:

tune_basic_example
~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/tune_basic_example.py


:orphan:

BayesOpt Example
~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/bayesopt_example.py


:orphan:

PB2 Example
~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/pb2_example.py

:orphan:

Keras Cifar10 Example
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/pbt_tune_cifar10_with_keras.py

:orphan:

Custom Checkpointing Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/custom_func_checkpointing.py


:orphan:

PBT Example
~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/pbt_example.py


:orphan:

Memory NN Example
~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/pbt_memnn_example.py


:orphan:

TensorFlow MNIST Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/tf_mnist_example.py


:orphan:

BOHB Example
~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/bohb_example.py


:orphan:

XGBoost Dynamic Resources Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/xgboost_dynamic_resources_example.py

:orphan:

AX Example
~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/ax_example.py


:orphan:

Asynchronous HyperBand Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/async_hyperband_example.py

:orphan:

Logging Example
~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/logging_example.py

:orphan:

PBT Function Example
~~~~~~~~~~~~~~~~~~~~

The following script produces the following results. For a population of 8 trials,
the PBT learning rate schedule roughly matches the optimal learning rate schedule.

.. image:: images/pbt_function_results.png

.. literalinclude:: /../../python/ray/tune/examples/pbt_function.py


:orphan:

MLflow PyTorch Lightning Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /../../python/ray/tune/examples/mlflow_ptl.py


