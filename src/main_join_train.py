#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime
import sys
import uuid
import glob
from pathlib import Path
import fire
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Engine, Events)
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, RunningAverage, Precision, Recall
from ignite.utils import convert_tensor
import random
import dataset
import models
import utils
import metrics
import losses
import torch.backends.cudnn as cudnn
from sklearn.metrics import average_precision_score
from torch.utils.tensorboard import SummaryWriter
import adabound


DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(DEVICE)
seed = 2021
if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

class Runner(object):
    """Main class to run experiments with e.g., train and evaluate"""
    def __init__(self, seed=42):
        """__init__
        :param config: YAML config file
        :param **kwargs: Overwrite of yaml config
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _forward(model, batch):
        inputs, frame_level_targets, time, embedding,embed_label, filenames,events = batch
        #print('inputs ',inputs.shape)
        # print('frame_level_targets ',frame_level_targets)
        inputs = convert_tensor(inputs, device=DEVICE, non_blocking=True)
        frame_level_targets = convert_tensor(frame_level_targets.float(), device=DEVICE, non_blocking=True)
        embedding = convert_tensor(embedding, device=DEVICE, non_blocking=True)
        embed_label = convert_tensor(embed_label,device=DEVICE,non_blocking=True)
        decision,decision_up,logit = model(inputs,embedding)
        return decision,decision_up, frame_level_targets, time,embed_label,logit
    
    @staticmethod
    def _negative_loss(engine):
        return -engine.state.metrics['Loss']

    def train(self, config, **kwargs):
        """Trains a given model specified in the config file or passed as the --model parameter.
        All options in the config file can be overwritten as needed by passing --PARAM
        Options with variable lengths ( e.g., kwargs can be passed by --PARAM '{"PARAM1":VAR1, "PARAM2":VAR2}'
        :param config: yaml config file
        :param **kwargs: parameters to overwrite yaml config
        """
        config_parameters = utils.parse_config_or_kwargs(config, **kwargs) # get parameters dict according to yaml file
        outputdir = os.path.join(config_parameters['outputpath'], config_parameters['model'],
            "{}_{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'), uuid.uuid1().hex)) #  according time and uuid, we can get one file path,all of our experiment results will store in it
        
        # Create base dir
        Path(outputdir).mkdir(exist_ok=True, parents=True) # make dir
        logger = utils.getfile_outlogger(os.path.join(outputdir, 'train.log')) # record train process. logger obeject can help us record our process
        logger.info("Storing files in {}".format(outputdir))
        # utils.pprint_dict
        utils.pprint_dict(config_parameters, logger.info) # print yaml file content
        logger.info("Running on device {}".format(DEVICE))
        transform = utils.parse_transforms(config_parameters['transforms']) # applying three data augment methods [freqmask, timemask, timeshift]
        torch.save(config_parameters, os.path.join(outputdir,'run_config.pth')) # save config_parameters
        logger.info("Transforms:")
        utils.pprint_dict(transform, logger.info, formatter='pretty') # print the details of transform
        # For Unbalanced Audioset, this is true
        sampling_kwargs = {"shuffle": True}
        logger.info("Using Sampler {}".format(sampling_kwargs))
        writter = SummaryWriter(os.path.join(outputdir, 'log'))
        trainloader = dataset.getdataloader_join(
            config_parameters['train_data'], # feature path
            config_parameters['spk_emb_file_path'],
            transform=transform,
            batch_size=config_parameters['batch_size'],
            num_workers=config_parameters['num_workers'],**sampling_kwargs)

        cvdataloader = dataset.getdataloader_join(
            config_parameters['cv_data'],
            config_parameters['spk_emb_file_path'],
            transform=None,
            shuffle=False,
            batch_size=config_parameters['batch_size'],
            num_workers=config_parameters['num_workers'])
        
        model = getattr(models, config_parameters['model'])(model_config=config_parameters,inputdim=224,  # inputdim = mel dimension of UrbanSED data, set accordingly as what dimesnion you have used to compute mel specgram in extract_features.py
                                                           outputdim=2, load_pretrained = True,**config_parameters['model_args'])
        if 'pretrained' in config_parameters and config_parameters['pretrained'] is not None:
            models.load_pretrained(model, config_parameters['pretrained'], outputdim=2)
            logger.info("Loading pretrained model {}".format(config_parameters['pretrained']))

        model = model.to(DEVICE)
        if config_parameters['optimizer'] == 'AdaBound':
            try:
                
                optimizer = adabound.AdaBound(model.parameters(), **config_parameters['optimizer_args'])
            except ImportError:
                config_parameters['optimizer'] = 'Adam'
                config_parameters['optimizer_args'] = {}
        else:
            optimizer = getattr(torch.optim,config_parameters['optimizer'],)(model.parameters(), **config_parameters['optimizer_args'])  
            
        utils.pprint_dict(optimizer, logger.info, formatter='pretty')
        utils.pprint_dict(model, logger.info, formatter='pretty')
        if DEVICE.type != 'cpu' and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs!".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model) 

        criterion_sed = getattr(losses, config_parameters['loss_sed'])().to(DEVICE)
        criterion_ce = getattr(losses, config_parameters['loss_ce'])().to(DEVICE)
        
        def _train_batch(_, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                decision,decision_up, frame_level_target, time,embed_label,logit = self._forward(model, batch)  # output is tuple (clip, frame, target)
                 
                loss_sed = criterion_sed(decision,frame_level_target)
                loss_ce = criterion_ce(logit,embed_label)
                loss = loss_sed + loss_ce
                loss.backward()
                # Single loss
                optimizer.step()
                writter.add_scalar('loss_sed', loss_sed.cpu().item(), train_engine.state.epoch)
                writter.add_scalar('loss_ce', loss_ce.cpu().item(), train_engine.state.epoch)
                writter.add_scalar('loss', loss.cpu().item(), train_engine.state.epoch)
                return loss.item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                return self._forward(model, batch)

        def thresholded_output_transform(output):
            decision,decision_up, frame_level_target, time,embed_label,logit = output
            y_pred = torch.round(decision)  
            return y_pred, frame_level_target

        precision = Precision(thresholded_output_transform, average=False)
        recall = Recall(thresholded_output_transform, average=False)
        f1_score = (precision * recall * 2 / (precision + recall)).mean()
        metrics = {
            'Loss': losses.Loss_join(criterion_sed),  #reimplementation of Loss, supports 3 way loss 
            'Precision': Precision(thresholded_output_transform),
            'Recall': Recall(thresholded_output_transform),
            'Accuracy': Accuracy(thresholded_output_transform),
            'F1': f1_score}
        train_engine = Engine(_train_batch)
        inference_engine = Engine(_inference)
        for name, metric in metrics.items():
            metric.attach(inference_engine, name)

        def compute_metrics(engine):
            inference_engine.run(cvdataloader) # run validate set
            results = inference_engine.state.metrics # 
            output_str_list = ["Validation Results - Epoch : {:<5}".format(engine.state.epoch)]
            for metric in metrics:
                output_str_list.append("{} {:<5.2f}".format(metric, results[metric])) # get all metric obout this validation
            logger.info(" ".join(output_str_list))
            
        pbar = ProgressBar(persist=False)
        pbar.attach(train_engine)
        if 'itercv' in config_parameters and config_parameters['itercv'] is not None:
            train_engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=config_parameters['itercv']), compute_metrics)
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, compute_metrics) # add validate process on train engine
        # Default scheduler is using patience=3, factor=0.1
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config_parameters['scheduler_args']) # using scheduler with learning rate

        @inference_engine.on(Events.EPOCH_COMPLETED)
        def update_reduce_on_plateau(engine):
            logger.info(f"Scheduling epoch {engine.state.epoch}")
            val_loss = engine.state.metrics['Loss']
            if 'ReduceLROnPlateau' == scheduler.__class__.__name__:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        early_stop_handler = EarlyStopping(
            patience=config_parameters['early_stop'],
            score_function=self._negative_loss, trainer=train_engine)

        inference_engine.add_event_handler(Events.EPOCH_COMPLETED, early_stop_handler) # add early stop to inference engine
        if config_parameters['save'] == 'everyepoch':
            checkpoint_handler = ModelCheckpoint(outputdir, 'run', n_saved=5, require_empty=False)
            train_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': model})
            train_engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=config_parameters['itercv']),
                checkpoint_handler, {'model': model})
        else:
            checkpoint_handler = ModelCheckpoint(
                outputdir,
                'run',
                n_saved=1,
                require_empty=False,
                score_function=self._negative_loss,
                global_step_transform=global_step_from_engine(train_engine), score_name='loss')
            inference_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': model})
        train_engine.run(trainloader, max_epochs=config_parameters['epochs'])
        return outputdir
    
    def evaluate(
            self,
            experiment_path: str,
            pred_file='hard_predictions_{}.txt',
            time_ratio=10. / 500,
            postprocessing='median',
            threshold=None,
            window_size=None,
            **kwargs):
        
        config = torch.load(list(Path(f'{experiment_path}').glob("run_config*"))[0], map_location='cpu')
        # Use previous config, but update data such as kwargs
        config_parameters = dict(config, **kwargs)

        model_parameters = torch.load(glob.glob("{}/run_model*".format(experiment_path))[0],
                                     map_location=lambda storage, loc: storage) # load parameter    
        strong_labels_df = pd.read_csv(config_parameters['label'], sep='\t') # get 
        if not np.issubdtype(strong_labels_df['filename'].dtype, np.number):
            strong_labels_df['filename'] = strong_labels_df['filename'].apply(os.path.basename)
        if 'audiofilepath' in strong_labels_df.columns:  # In case of ave dataset, the audiofilepath column is the main column
            strong_labels_df['audiofilepath'] = strong_labels_df['audiofilepath'].apply(os.path.basename)
            colname = 'audiofilepath'  # AVE
        else:
            colname = 'filename'  # Dcase etc.
        if "event_labels" in strong_labels_df.columns:
            assert False, "Data with the column event_labels are used to train not to evaluate"
        dataloader = dataset.getdataloader_join(
            config_parameters['test_data'],
            config_parameters['spk_emb_file_path'],
            batch_size=1, shuffle=False)
        model = getattr(models, config_parameters['model'])(
            model_config=config_parameters,inputdim=64, outputdim=2, **config_parameters['model_args'])
        model.load_state_dict(model_parameters)
        model = model.to(DEVICE).eval()
        time_predictions, clip_predictions = [], []
        
        mAP_pred, mAP_tar = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, unit='file', leave=False): # dataloard  
                inputs, frame_level_target, time, embedding,embed_label, filenames,events = batch
                #inputs, frame_level_target, time, embedding, filenames,events = batch
                decision,decision_up, frame_level_target, time,embed_label,logit = self._forward(model, batch) # 
                mAP_tar.append(frame_level_target.detach().cpu().numpy().squeeze(0))
                mAP_pred.append(decision.detach().cpu().numpy().squeeze(0))
                 
                pred = decision_up.detach().cpu().numpy()
                pred = pred[:,:,0] # shape(B, 501)
                # print(pred.shape)
                if postprocessing == 'median':
                    if threshold is None:
                        thres = 0.37
                    else:
                        thres = threshold
                    if window_size is None:
                        window_size = 1
                    # print(pred.shape)
                    filtered_pred = utils.median_filter(pred, window_size=window_size, threshold=thres)
                   
                    decoded_pred = []
                    decoded_pred_ = utils.decode_with_timestamps(events[0],filtered_pred[0,:])
                    if len(decoded_pred_) == 0: # neg deal
                        decoded_pred_.append((events[0],0,0))
                    decoded_pred.append(decoded_pred_)
                    # print('decoded_pred ',decoded_pred)
                    # assert 1==2
                else:
                    # Double thresholding as described in
                    # https://arxiv.org/abs/1904.03841
                    if threshold is None:
                        hi_thres, low_thres = (0.75, 0.2) # i change 0.75 to 0.7
                    else:
                        hi_thres, low_thres = threshold
                    filtered_pred = utils.double_threshold(pred, high_thres=hi_thres, low_thres=low_thres)
                    decoded_pred = utils.decode_with_timestamps(events[0],filtered_pred)

                for num_batch in range(len(decoded_pred)): # when we test our model,the batch_size is 1
                    #print('len(decoded_pred) ',len(decoded_pred))
                    filename = filenames[num_batch]
                    #print('filename ',filenames[num_batch])
                    cur_pred = pred[num_batch]
                    # Save each frame output, for later visualization
                    label_prediction = decoded_pred[num_batch] # frame predict
                    # print(label_prediction)
                    for event_label, onset, offset in label_prediction:
                        time_predictions.append({
                            'filename': filename,
                            'onset': onset,
                            'offset': offset,
                            'event_label': event_label}) # get real predict results,including event_label,onset,offset

        assert len(time_predictions) > 0, "No outputs, lower threshold?"
        pred_df = pd.DataFrame(time_predictions, columns=['filename', 'onset', 'offset','event_label']) # it store the happen event and its time information
        pred_df = utils.predictions_to_time(pred_df, ratio=time_ratio) # transform the number of frame to real time
        test_data_filename = os.path.splitext(os.path.basename(config_parameters['label']))[0] + "_threshold_0.37" 
        if pred_file: # it name is hard_predictions...
            pred_df.to_csv(os.path.join(experiment_path, pred_file.format(test_data_filename)),
                                        index=False, sep="\t")
        
        print("Prediction file saved in {}".format(os.path.join(experiment_path, pred_file.format(test_data_filename))))
        

    def train_evaluate(self, config, **kwargs):
        experiment_path = self.train(config, **kwargs) # return some directory path as "experiment/Join_fusion/2024..."
        from h5py import File
        # Get the output time-ratio factor from the model
        model_parameters = torch.load(
            glob.glob("{}/run_model*".format(experiment_path))[0],  # load the pytorch model present in that experiment_path 
            map_location=lambda storage, loc: storage)
        config_param = torch.load(glob.glob(
            "{}/run_config*".format(experiment_path))[0],           # load the config file present in that experiment_path  
                                  map_location=lambda storage, loc: storage)
        # Dummy to calculate the pooling factor a bit dynamic
        config_parameters = utils.parse_config_or_kwargs(config, **kwargs) # parse the config file and store it in config_parameters dictionary
        model = getattr(models, config_param['model'])(model_config=config_parameters,inputdim=224,
                        outputdim=2, load_pretrained = False, **config_param['model_args'])   # it instantiates the model, getattr(models, config_param['model']) will return the model class 
        model.load_state_dict(model_parameters)
        model.to(DEVICE)
        #time_ratio = 10.0/500
        time_ratio = 10.0/1001 # fro convnext
        threshold = config_parameters.get('threshold', None)
        postprocessing = config_parameters.get('postprocessing', 'double')  
        window_size = config_parameters.get('window_size', None)
        self.evaluate(experiment_path,
                      time_ratio=time_ratio,
                      postprocessing=postprocessing,
                      threshold=threshold,
                      window_size=window_size)

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    fire.Fire(Runner)
    
