import time

import numpy as np
from numpy.random import RandomState
import torch as th

from braindecode.experiments.monitors import (
    LossMonitor,
    MisclassMonitor,
    RuntimeMonitor,
    CroppedTrialMisclassMonitor,
    compute_trial_labels_from_crop_preds,
    compute_pred_labels_from_trial_preds,
    compute_preds_per_trial_from_crops,
)
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import (
    BalancedBatchSizeIterator,
    CropsFromTrialsIterator,
)
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.schedulers import CosineAnnealing, ScheduledOptimizer
from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.experiments.loggers import Printer, TensorboardWriter
import logging

log = logging.getLogger(__name__)


def find_optimizer(optimizer_name):
    optim_found = False
    for name in th.optim.__dict__.keys():
        if name.lower() == optimizer_name.lower():
            optimizer = th.optim.__dict__[name]
            optim_found = True
            break
    if not optim_found:
        raise ValueError("Unknown optimizer {:s}".format(optimizer))
    return optimizer


class BaseModel(object):
    def cuda(self, n_cuda=4):
        """Move underlying model to GPU."""
        self._ensure_network_exists()
        assert (
            not self.compiled
        ), "Call cuda before compiling model, otherwise optimization will not work"
        if n_cuda > 0:
            self.network = self.network.cuda()
            self.cuda = True
        else:
            self.cuda = False
        self.n_cuda = n_cuda
        return self

    def parameters(self):
        """
        Return parameters of underlying torch model.
    
        Returns
        -------
        parameters: list of torch tensors
        """
        self._ensure_network_exists()
        return self.network.parameters()

    def _ensure_network_exists(self):
        if not hasattr(self, "network"):
            self.network = self.create_network()
            self.cuda = False
            self.compiled = False

    def compile(
            self,
            loss,
            optimizer,
            logdir,
            extra_monitors=None,
            cropped=False,
            iterator_seed=0,
            pre_load_dict_path=None,
            input_time_length=None,
    ):
        """
        Setup training for this model.
        
        Parameters
        ----------
        loss: function (predictions, targets) -> torch scalar
        optimizer: `torch.optim.Optimizer` or string
            Either supply an optimizer or the name of the class (e.g. 'adam')
        extra_monitors: List of Braindecode monitors, optional
            In case you want to monitor additional values except for loss, misclass and runtime.
        cropped: bool
            Whether to perform cropped decoding, see cropped decoding tutorial.
        iterator_seed: int
            Seed to seed the iterator random generator.
        Returns
        -------

        """

        self._ensure_network_exists()
        if cropped:
            model_already_dense = np.any(
                [
                    hasattr(m, "dilation")
                    and (m.dilation != 1)
                    and (m.dilation) != (1, 1)
                    for m in self.network.modules()
                ]
            )
            if not model_already_dense:
                to_dense_prediction_model(self.network)
            else:
                log.info("Seems model was already converted to dense model...")
        if not hasattr(optimizer, "step"):
            optimizer_class = find_optimizer(optimizer)
            optimizer = optimizer_class(self.network.parameters())
        self.loss = loss
        self.optimizer = optimizer
        self.logdir = logdir
        self.extra_monitors = extra_monitors
        # Already setting it here, so multiple calls to fit
        # will lead to different batches being drawn
        self.seed_rng = RandomState(iterator_seed)
        self.cropped = cropped
        self.compiled = True
        self.pre_load = False
        if pre_load_dict_path is not None:
            self.pre_load = True
            import torch
            checkpoint = torch.load(pre_load_dict_path)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.monitors = [LossMonitor(), ]
            self.input_time_length = input_time_length

    def fit(
            self,
            train_X,
            train_y,
            epochs,
            no_decrease_epochs_to_stop,
            batch_size,
            input_time_length=None,
            validation_data=None,
            model_constraint=None,
            remember_best_column=None,
            scheduler=None,
            log_0_epoch=True,
            model_name=None,
    ):
        """
        Fit the model using the given training data.
        
        Will set `epochs_df` variable with a pandas dataframe to the history
        of the training process.
        
        Parameters
        ----------
        train_X: ndarray
            Training input data
        train_y: 1darray
            Training labels
        epochs: int
            Number of epochs to train
        batch_size: int
        input_time_length: int, optional
            Super crop size, what temporal size is pushed forward through 
            the network, see cropped decoding tuturial.
        validation_data: (ndarray, 1darray), optional
            X and y for validation set if wanted
        model_constraint: object, optional
            You can supply :class:`.MaxNormDefaultConstraint` if wanted.
        remember_best_column: string, optional
            In case you want to do an early stopping/reset parameters to some
            "best" epoch, define here the monitored value whose minimum
            determines the best epoch.
        scheduler: 'cosine' or None, optional
            Whether to use cosine annealing (:class:`.CosineAnnealing`).
        log_0_epoch: bool
            Whether to compute the metrics once before training as well.

        Returns
        -------
        exp: 
            Underlying braindecode :class:`.Experiment`
        """
        if (not hasattr(self, "compiled")) or (not self.compiled):
            raise ValueError(
                "Compile the model first by calling model.compile(loss, optimizer, metrics)"
            )

        if self.cropped and input_time_length is None:
            raise ValueError(
                "In cropped mode, need to specify input_time_length,"
                "which is the number of timesteps that will be pushed through"
                "the network in a single pass."
            )

        train_X = _ensure_float32(train_X)
        if self.cropped:
            self.network.eval()
            test_input = np_to_var(
                np.ones(
                    (1, train_X[0].shape[0], input_time_length)
                    + train_X[0].shape[2:],
                    dtype=np.float32,
                )
            )
            while len(test_input.size()) < 4:
                test_input = test_input.unsqueeze(-1)
            if self.cuda:
                test_input = test_input.cuda()
            out = self.network(test_input)
            n_preds_per_input = out.cpu().data.numpy().shape[2]
            self.iterator = CropsFromTrialsIterator(
                batch_size=batch_size,
                input_time_length=input_time_length,
                n_preds_per_input=n_preds_per_input,
                seed=self.seed_rng.randint(0, np.iinfo(np.int32).max - 1),
            )
        else:
            self.iterator = BalancedBatchSizeIterator(
                batch_size=batch_size,
                seed=self.seed_rng.randint(0, np.iinfo(np.int32).max - 1),
            )
        if log_0_epoch:
            max_epoch = MaxEpochs(epochs)
        else:
            max_epoch = MaxEpochs(epochs - 1)

        early_stop = NoDecrease('valid_misclass', no_decrease_epochs_to_stop, min_decrease=1e-06)
        stop_criterion = Or([max_epoch, early_stop])
        train_set = SignalAndTarget(train_X, train_y)
        optimizer = self.optimizer
        if scheduler is not None:
            assert (
                    scheduler == "cosine"
            ), "Supply either 'cosine' or None as scheduler."
            n_updates_per_epoch = sum(
                [1 for _ in self.iterator.get_batches(train_set, shuffle=True)]
            )
            n_updates_per_period = n_updates_per_epoch * epochs
            if scheduler == "cosine":
                scheduler = CosineAnnealing(n_updates_per_period)
            schedule_weight_decay = False
            if optimizer.__class__.__name__ == "AdamW":
                schedule_weight_decay = True
            optimizer = ScheduledOptimizer(
                scheduler,
                self.optimizer,
                schedule_weight_decay=schedule_weight_decay,
            )
        loss_function = self.loss
        if self.cropped:
            loss_function = lambda outputs, targets: self.loss(
                th.mean(outputs, dim=2), targets
            )
        if validation_data is not None:
            valid_X = _ensure_float32(validation_data[0])
            valid_y = validation_data[1]
            valid_set = SignalAndTarget(valid_X, valid_y)
        else:
            valid_set = None
        test_set = None
        self.monitors = [LossMonitor(), ]
        if self.cropped:
            self.monitors.append(CroppedTrialMisclassMonitor(input_time_length))
        else:
            self.monitors.append(MisclassMonitor())
        if self.extra_monitors is not None:
            self.monitors.extend(self.extra_monitors)
        self.monitors.append(RuntimeMonitor())
        mkdir(self.logdir)
        loggers = [Printer(), TensorboardWriter(self.logdir)]
        exp = Experiment(
            self.network,
            train_set,
            valid_set,
            test_set,
            iterator=self.iterator,
            loss_function=loss_function,
            optimizer=optimizer,
            model_constraint=model_constraint,
            monitors=self.monitors,
            stop_criterion=stop_criterion,
            remember_best_column=remember_best_column,
            run_after_early_stop=False,
            n_cuda=self.n_cuda,
            log_0_epoch=log_0_epoch,
            do_early_stop=(remember_best_column is not None),
            loggers=loggers,
        )
        best_v, best_model, best_optimizer = exp.run()
        self.epochs_df = exp.epochs_df
        self.before_stop_df = exp.before_stop_df
        import torch
        import os
        current_path = os.getcwd()
        mkdir(current_path + "/model_dict")
        if model_name is None:
            path = current_path + "/model_dict" + "/{}_{}_model.pth".format(best_v, remember_best_column, )
        else:
            path = current_path + "/model_dict" + "/{}_{}_{}_model.pth".format(best_v, remember_best_column, model_name)

        torch.save({
            'model_state_dict': best_model.module.state_dict(),
            'optimizer_state_dict': best_optimizer.state_dict(),
        }, path)
        # return exp
        return best_v, best_model, best_optimizer

    def evaluate(self, X, y, batch_size=32):
        """
        Evaluate, i.e., compute metrics on given inputs and targets.
        
        Parameters
        ----------
        X: ndarray
            Input data.
        y: 1darray
            Targets.

        Returns
        -------
        result: dict
            Dictionary with result metrics.

        """

        X = _ensure_float32(X)
        stop_criterion = MaxEpochs(0)
        train_set = SignalAndTarget(X, y)
        model_constraint = None
        valid_set = None
        test_set = None
        loss_function = self.loss
        if self.cropped:
            loss_function = lambda outputs, targets: self.loss(
                th.mean(outputs, dim=2), targets
            )

        if self.pre_load:
            # if we preload model dict, we need setting monitors, because usually it setted in self.fit() function
            if self.cropped:
                if self.input_time_length == None:
                    raise ValueError(
                        "preload cropped model need [input_time_length] to setting CroppedTrialMisclassMonitor "
                    )
                self.monitors.append(CroppedTrialMisclassMonitor(self.input_time_length))
            else:
                self.monitors.append(MisclassMonitor())
            if self.extra_monitors is not None:
                self.monitors.extend(self.extra_monitors)
            self.monitors.append(RuntimeMonitor())
            # setting iterator
            if self.cropped:
                self.network.eval()
                test_input = np_to_var(
                    np.ones(
                        (1, train_set.X[0].shape[0], self.input_time_length)
                        + train_set.X[0].shape[2:],
                        dtype=np.float32,
                    )
                )
                while len(test_input.size()) < 4:
                    test_input = test_input.unsqueeze(-1)
                if self.cuda:
                    test_input = test_input.cuda()
                out = self.network(test_input)
                n_preds_per_input = out.cpu().data.numpy().shape[2]
                self.iterator = CropsFromTrialsIterator(
                    batch_size=batch_size,
                    input_time_length=self.input_time_length,
                    n_preds_per_input=n_preds_per_input,
                    seed=self.seed_rng.randint(0, np.iinfo(np.int32).max - 1),
                )
            else:
                self.iterator = BalancedBatchSizeIterator(
                    batch_size=batch_size,
                    seed=self.seed_rng.randint(0, np.iinfo(np.int32).max - 1),
                )

        # reset runtime monitor if exists...
        for monitor in self.monitors:
            if hasattr(monitor, "last_call_time"):
                monitor.last_call_time = time.time()
        exp = Experiment(
            self.network,
            train_set,
            valid_set,
            test_set,
            iterator=self.iterator,
            loss_function=loss_function,
            optimizer=self.optimizer,
            model_constraint=model_constraint,
            monitors=self.monitors,
            stop_criterion=stop_criterion,
            remember_best_column=None,
            run_after_early_stop=False,
            n_cuda=self.n_cuda,
            log_0_epoch=True,
            do_early_stop=False,
        )

        exp.monitor_epoch({"train": train_set})

        result_dict = dict(
            [
                (key.replace("train_", ""), val)
                for key, val in dict(exp.epochs_df.iloc[0]).items()
            ]
        )
        return result_dict

    def predict_classes(
            self, X, threshold_for_binary_case=None, individual_crops=False
    ):
        """
        Predict the labels for given input data.
        
        Parameters
        ----------
        X: ndarray
            Input data.
        threshold_for_binary_case: float, optional
            In case of a model with single output, the threshold for assigning,
            label 0 or 1, e.g. 0.5.

        Returns
        -------
        pred_labels: 1darray or list of 1darrays
            Predicted labels per trial, optionally for each crop within trial.
        """
        if individual_crops:
            assert self.cropped, "Cropped labels only for cropped decoding"
        outs_per_trial = self.predict_outs(
            X=X, individual_crops=individual_crops
        )

        pred_labels = [np.argmax(o, axis=0) for o in outs_per_trial]
        if not individual_crops:
            pred_labels = np.array(pred_labels)
        return pred_labels

    def predict_outs(self, X, individual_crops=False):
        """
        Predict raw outputs of the network for given input.

        Parameters
        ----------
        X: ndarray
            Input data.
        threshold_for_binary_case: float, optional
            In case of a model with single output, the threshold for assigning,
            label 0 or 1, e.g. 0.5.
        individual_crops: bool

        Returns
        -------
            outs_per_trial: 2darray or list of 2darrays
                Network outputs for each trial, optionally for each crop within trial.
        """
        if individual_crops:
            assert self.cropped, "Cropped labels only for cropped decoding"
        X = _ensure_float32(X)
        all_preds = []
        with th.no_grad():
            dummy_y = np.ones(len(X), dtype=np.int64)
            for b_X, _ in self.iterator.get_batches(
                    SignalAndTarget(X, dummy_y), False
            ):
                b_X_var = np_to_var(b_X)
                if self.cuda:
                    b_X_var = b_X_var.cuda()
                all_preds.append(var_to_np(self.network(b_X_var)))
        if self.cropped:
            outs_per_trial = compute_preds_per_trial_from_crops(
                all_preds, self.iterator.input_time_length, X
            )
            if not individual_crops:
                outs_per_trial = np.array(
                    [np.mean(o, axis=1) for o in outs_per_trial]
                )
        else:
            outs_per_trial = np.concatenate(all_preds)
        return outs_per_trial

    def load_model_dict(self, PATH):
        import torch
        checkpoint = torch.load(PATH)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def _ensure_float32(X):
    if hasattr(X, "astype"):
        X = X.astype(np.float32, copy=False)
    else:
        X = [x.astype(np.float32, copy=False) for x in X]
    return X


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print
        path + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print
        path + ' 目录已存在'
        return False
