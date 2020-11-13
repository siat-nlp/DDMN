#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: source/utils/engine.py
"""

import os
import time
import shutil
import numpy as np
import torch

from collections import defaultdict
from source.inputter.batcher import create_turn_batch, create_kb_batch


class MetricsManager(object):
    """
    MetricsManager
    """
    def __init__(self):
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0

    def update(self, metrics_list):
        """
        update
        """
        for i, metrics in enumerate(metrics_list):
            num_samples = metrics.pop("num_samples", 1)
            self.num_samples += num_samples
            for key, val in metrics.items():
                if val is not None:
                    key_turn = str(key) + "-turn-{}".format(str(i+1))
                    self.metrics_val[key_turn] = val

                    if isinstance(val, torch.Tensor):
                        val = val.item()
                        self.metrics_cum[key] += val * num_samples
                    elif isinstance(val, tuple):
                        assert len(val) == 2
                        val, num_words = val[0].item(), val[1]
                        self.metrics_cum[key] += np.array([val * num_samples, num_words])
                    else:
                        self.metrics_cum[key] += val * num_samples

    def clear(self):
        """
        clear
        """
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0

    def get(self, name):
        """
        get
        """
        val = self.metrics_cum.get(name)
        if not isinstance(val, float):
            val = val[0]
        return val / self.num_samples

    def report_val(self):
        """
        report_val
        """
        metric_strs = []
        for key, val in self.metrics_val.items():
            metric_str = "{}={:.3f}".format(key.upper(), val)
            metric_strs.append(metric_str)
        metric_strs = "   ".join(metric_strs)
        return metric_strs

    def report_cum(self):
        """
        report_cum
        """
        metric_strs = []
        for key, val in self.metrics_cum.items():
            if isinstance(val, float):
                val, num_words = val, None
            else:
                val, num_words = val

            metric_str = "{}={:.3f}".format(key.upper(), val / self.num_samples)
            metric_strs.append(metric_str)

            if num_words is not None:
                ppl = np.exp(min(val / num_words, 100))
                metric_str = "{}-PPL={:.3f}".format(key.upper(), ppl)
                metric_strs.append(metric_str)

        metric_strs = "   ".join(metric_strs)
        return metric_strs


class Trainer(object):
    """
    Trainer
    """
    def __init__(self,
                 model,
                 optimizer,
                 train_iter,
                 valid_iter,
                 logger,
                 valid_metric_name="-loss",
                 num_epochs=1,
                 pre_epochs=1,
                 save_dir=None,
                 log_steps=None,
                 valid_steps=None,
                 grad_clip=None,
                 lr_scheduler=None,
                 entity_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.logger = logger

        self.is_decreased_valid_metric = valid_metric_name[0] == "-"
        self.valid_metric_name = valid_metric_name[1:]
        self.num_epochs = num_epochs
        self.pre_epochs = pre_epochs
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.valid_steps = valid_steps
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.entity_dir = entity_dir

        self.best_valid_metric = float(
            "inf") if self.is_decreased_valid_metric else -float("inf")
        self.epoch = 0
        self.batch_num = 0
        self.use_rl = False

        self.train_start_message = "\n".join(["",
                                              "=" * 85,
                                              "=" * 34 + " Model Training " + "=" * 35,
                                              "=" * 85,
                                              ""])
        self.valid_start_message = "\n" + "-" * 33 + " Model Evaulation " + "-" * 33

    def train(self):
        """
        train
        """
        for epoch in range(self.epoch, self.pre_epochs):
            self.train_epoch()

        self.use_rl = True
        self.valid_metric_name = "bleu_score"      # "bleu_score" or "f1_score"
        self.best_valid_metric = -float("inf")
        self.is_decreased_valid_metric = False
        optimizer = self.optimizer
        #optimizer = getattr(torch.optim, "Adam")(self.model.parameters(), lr=6.25e-5)
        lr_decay = self.lr_scheduler.factor
        patience = self.lr_scheduler.patience
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='max', factor=lr_decay,
            patience=patience, verbose=True, min_lr=1e-6)

        self.logger.info("RL-training starts ...")
        for epoch in range(self.epoch, self.num_epochs):
            self.train_epoch()

    def train_epoch(self):
        """
        train_epoch
        """
        self.epoch += 1
        train_mm = MetricsManager()
        num_batches = self.train_iter.n_batch
        self.train_iter.prepare_epoch()
        self.logger.info(self.train_start_message)

        for batch_idx in range(num_batches):
            self.model.train()
            start_time = time.time()

            local_data = self.train_iter.get_batch(batch_idx)
            turn_inputs = create_turn_batch(local_data['inputs'])
            kb_inputs = create_kb_batch(local_data['kbs'])
            assert len(turn_inputs) == local_data['max_turn']
            
            metrics_list = self.model.iterate(turn_inputs, kb_inputs,
                                              optimizer=self.optimizer,
                                              grad_clip=self.grad_clip,
                                              use_rl=self.use_rl,
                                              entity_dir=self.entity_dir,
                                              is_training=True)

            elapsed = time.time() - start_time
            train_mm.update(metrics_list)
            self.batch_num += 1

            if (batch_idx + 1) % self.log_steps == 0:
                message_prefix = "[Train][{:2d}][{}/{}]".format(self.epoch, batch_idx + 1, num_batches)
                metrics_message = train_mm.report_val()
                message_posfix = "TIME={:.2f}s".format(elapsed)
                self.logger.info("   ".join(
                    [message_prefix, metrics_message, message_posfix]))

            if (batch_idx + 1) % self.valid_steps == 0:
                self.logger.info(self.valid_start_message)
                valid_mm = self.evaluate(self.model, self.valid_iter, use_rl=self.use_rl, entity_dir=self.entity_dir)

                message_prefix = "[Valid][{:2d}][{}/{}]".format(self.epoch, batch_idx + 1, num_batches)
                metrics_message = valid_mm.report_cum()
                self.logger.info("   ".join([message_prefix, metrics_message]))

                cur_valid_metric = valid_mm.get(self.valid_metric_name)
                if self.is_decreased_valid_metric:
                    is_best = cur_valid_metric < self.best_valid_metric
                else:
                    is_best = cur_valid_metric > self.best_valid_metric
                if is_best:
                    self.best_valid_metric = cur_valid_metric
                self.save(is_best, is_rl=self.use_rl)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(cur_valid_metric)
                self.logger.info("-" * 85 + "\n")

        self.save()
        self.logger.info('')

    @staticmethod
    def evaluate(model, data_iter, use_rl=False, entity_dir=None):
        """
        evaluate
        """
        model.eval()
        mm = MetricsManager()
        num_batches = data_iter.n_batch
        with torch.no_grad():
            for batch_idx in range(num_batches):
                local_data = data_iter.get_batch(batch_idx)
                turn_inputs = create_turn_batch(local_data['inputs'])
                kb_inputs = create_kb_batch(local_data['kbs'])
                assert len(turn_inputs) == local_data['max_turn']

                metrics_list = model.iterate(turn_inputs, kb_inputs,
                                             use_rl=use_rl, entity_dir=entity_dir, is_training=False)
                mm.update(metrics_list)
        return mm

    def save(self, is_best=False, is_rl=False):
        """
        save
        """
        model_file = os.path.join(self.save_dir, "state_epoch_{}.model".format(self.epoch))
        torch.save(self.model.state_dict(), model_file)
        self.logger.info("Saved model state to '{}'".format(model_file))

        train_file = os.path.join(self.save_dir, "state_epoch_{}.train".format(self.epoch))
        train_state = {"epoch": self.epoch,
                       "batch_num": self.batch_num,
                       "best_valid_metric": self.best_valid_metric,
                       "optimizer": self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            train_state["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(train_state, train_file)
        self.logger.info("Saved train state to '{}'".format(train_file))

        if is_best:
            if is_rl:
                best_model_file = os.path.join(self.save_dir, "best_rl.model")
                best_train_file = os.path.join(self.save_dir, "best_rl.train")
            else:
                best_model_file = os.path.join(self.save_dir, "best.model")
                best_train_file = os.path.join(self.save_dir, "best.train")
            shutil.copy(model_file, best_model_file)
            shutil.copy(train_file, best_train_file)
            self.logger.info(
                "Saved best model state to '{}' with new best valid metric {}={:.3f}".format(
                    best_model_file, self.valid_metric_name.upper(), self.best_valid_metric))

    def load(self, file_ckpt):
        """
        load
        """
        if os.path.isfile(os.path.join(self.save_dir, file_ckpt)):
            file_prefix = file_ckpt.split('.')[0]
            model_file = "{}/{}.model".format(self.save_dir, file_prefix)
            train_file = "{}/{}.train".format(self.save_dir, file_prefix)

            model_state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(model_state_dict)
            self.logger.info("Loaded model state from '{}'".format(model_file))

            train_state_dict = torch.load(train_file, map_location=lambda storage, loc: storage)
            self.epoch = train_state_dict["epoch"]
            self.best_valid_metric = train_state_dict["best_valid_metric"]
            self.batch_num = train_state_dict["batch_num"]
            self.optimizer.load_state_dict(train_state_dict["optimizer"])
            if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
                self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
            self.logger.info(
                "Loaded train state from '{}' with (epoch-{} best_valid_metric={:.3f})".format(
                    train_file, self.epoch, self.best_valid_metric))
