import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
from math import *
import time
from PIL import Image
import numpy as np
import torch.nn.functional as F
import scipy.io as sio

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

def evaluate(diffusion, loader):
    origin_dice, regist_dice, mse_list, jac_list, jmin_list, jmax_list, time_list = [], [], [], [], [], [], []
    for data in loader:
        time1 = time.time()
        diffusion.feed_data(data)
        diffusion.test_registration(continuous=True)
        time2 = time.time()

        visuals = diffusion.get_current_registration()
        defm_frames = visuals['contD'].squeeze().numpy().transpose(0, 2, 3, 1)
        flow_frames = visuals['contF'].squeeze().numpy().transpose(0, 3, 4, 2, 1)
        flow_ES = flow_frames[-1]

        sflow = torch.from_numpy(flow_ES.transpose(3, 2, 0, 1).copy()).unsqueeze(0)
        sflow = Metrics.transform_grid(sflow[:, 0], sflow[:, 1], sflow[:, 2])
        nb, nc, nd, nh, nw = sflow.shape
        segflow = torch.FloatTensor(sflow.shape).zero_()
        segflow[:, 2] = (sflow[:, 0] / (nd - 1) - 0.5) * 2.0
        segflow[:, 1] = (sflow[:, 1] / (nh - 1) - 0.5) * 2.0
        segflow[:, 0] = (sflow[:, 2] / (nw - 1) - 0.5) * 2.0
        regist_seg = F.grid_sample(data['MS'].cuda().float(), segflow.cuda().float().permute(0, 2, 3, 4, 1),
                                   mode='nearest', align_corners=True)
        regist_seg = regist_seg.squeeze().cpu().numpy().transpose(1, 2, 0)
        label_seg = data['FS'].squeeze().cpu().numpy().transpose(1, 2, 0)
        origin_seg = data['MS'].squeeze().cpu().numpy().transpose(1, 2, 0)
        regist_dice.append(np.mean(Metrics.dice_OASIS(regist_seg, label_seg)))
        origin_dice.append(np.mean(Metrics.dice_OASIS(origin_seg, label_seg)))

        data_fixed = data['F'].squeeze().cpu().numpy().transpose(1, 2, 0)
        vol_def = defm_frames[-1] * 255.0
        vol_fix = (data_fixed + 1) / 2.0 * 255.0
        mse_list.append(Metrics.mse_metric(vol_def, vol_fix))

        disp_field = flow_ES.transpose(3, 2, 0, 1)
        jac_folds, min_jac, max_jac = Metrics.jacobian_determinant_3d(disp_field)
        jac_list.append(jac_folds)
        jmin_list.append(min_jac)
        jmax_list.append(max_jac)
        time_list.append(time2 - time1)

    return {
        'Original Dice': origin_dice,
        'Deform Dice': regist_dice,
        'MSE': mse_list,
        'Jac Negatives (%)': jac_list,
        'Min J': jmin_list,
        'Max J': jmax_list,
        'Deform Time (s)': time_list
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase != opt['phase']: continue
        if opt['phase'] == 'train':
            batchSize = opt['datasets']['train']['batch_size']
            train_set = Data.create_dataset_3D(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            val_set = Data.create_dataset_3D(dataset_opt, 'val')
            val_loader = Data.create_dataloader(val_set, dataset_opt, 'val')
            training_iters = int(ceil(train_set.data_len / float(batchSize)))
        elif opt['phase'] == 'test':
            test_set = Data.create_dataset_3D(dataset_opt, phase)
            test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    if opt['phase'] == 'train':
        current_step = diffusion.begin_step
        current_epoch = diffusion.begin_epoch
        n_epoch = opt['train']['n_epoch']
        if opt['path']['resume_state']:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

        while current_epoch < n_epoch:
            current_epoch += 1
            for istep, train_data in enumerate(train_loader):
                iter_start_time = time.time()
                current_step += 1

                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if (istep + 1) % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    t = (time.time() - iter_start_time) / batchSize
                    message = '(Train - epoch: %d | iters: %d/%d | time: %.3f) ' % (current_epoch, istep + 1, training_iters, t)
                    message += ' '.join('%s: %.6f' % (k, v) for k, v in logs.items())
                    logger.info(message)

            if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)
                val_metrics = evaluate(diffusion, val_loader)
                logger.info('Validation epoch %d' % current_epoch)
                for name, values in val_metrics.items():
                    logger.info('  %-18s | mean = %.4f, std = %.4f' % (name, np.mean(values), np.std(values)))

        # save model
        logger.info('End of training.')

    else:

        registDice = []
        originDice = []
        registTime = []
        mseList = []
        jacList = []
        jminList = []  # Añadido para Min J
        jmaxList = []  # Añadido para Max J
        logger.info('Begin Model Evaluation.')
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for istep,  test_data in enumerate(test_loader):
            idx += 1
            dataName = test_data['P'][0].split('/')[-1][:-4]
            print('Test Data: %s' % dataName)

            time1 = time.time()
            diffusion.feed_data(test_data)
            diffusion.test_registration(continuous=True)
            time2 = time.time()

            visuals = diffusion.get_current_registration()
            defm_frames = visuals['contD'].squeeze().numpy().transpose(0, 2, 3, 1)
            flow_frames = visuals['contF'].squeeze().numpy().transpose(0, 3, 4, 2, 1)
            flow_frames_ES = flow_frames[-1]

            # guardar la primera del test
            if idx == 1:
                disp_field = flow_frames_ES.transpose(3, 2, 0, 1)
                _, dpt, hgt, wdt = disp_field.shape

                D_mesh, H_mesh, W_mesh = np.meshgrid(np.arange(dpt), np.arange(hgt), np.arange(wdt), indexing='ij')
                identidad = np.stack([D_mesh, H_mesh, W_mesh], axis=0)

                # deformacion = id  + desplazamientos
                phi = identidad + disp_field

                img_target = (test_data['F'].squeeze().cpu().numpy() + 1.0) / 2.0

                img_warped = visuals['contD'].squeeze().cpu().numpy()[-1]
                img_source = (test_data['M'].squeeze().cpu().numpy() + 1.0) / 2.0


                # guardar todo enun mat
                mat_path = os.path.join(result_path, f'{dataName}_completo.mat')
                sio.savemat(mat_path, {
                    'phi': phi,
                    'identidad': identidad,
                    'img_target': img_target,
                    'img_warped': img_warped,
                    'img_source': img_source
                })
                print(f'-> Saved {mat_path}.')


            sflow = torch.from_numpy(flow_frames_ES.transpose(3, 2, 0, 1).copy()).unsqueeze(0)
            sflow = Metrics.transform_grid(sflow[:, 0], sflow[:, 1], sflow[:, 2])
            nb, nc, nd, nh, nw = sflow.shape
            segflow = torch.FloatTensor(sflow.shape).zero_()
            segflow[:, 2] = (sflow[:, 0] / (nd - 1) - 0.5) * 2.0  # D[0 -> 2]
            segflow[:, 1] = (sflow[:, 1] / (nh - 1) - 0.5) * 2.0  # H[1 -> 1]
            segflow[:, 0] = (sflow[:, 2] / (nw - 1) - 0.5) * 2.0  # W[2 -> 0]
            origin_seg_tensor = test_data['MS'].cuda().float()
            regist_seg = F.grid_sample(origin_seg_tensor, (segflow.cuda().float().permute(0, 2, 3, 4, 1)),mode='nearest', align_corners=True)
            regist_seg = regist_seg.squeeze().cpu().numpy().transpose(1, 2, 0)
            label_seg = test_data['FS'].squeeze().cpu().numpy().transpose(1, 2, 0)
            origin_seg = test_data['MS'].squeeze().cpu().numpy().transpose(1, 2, 0)
            vals_regist = Metrics.dice_OASIS(regist_seg, label_seg)
            vals_origin = Metrics.dice_OASIS(origin_seg, label_seg)
            registDice.append(np.mean(vals_regist))
            originDice.append(np.mean(vals_origin))
            
            data_origin = test_data['M'].squeeze().cpu().numpy().transpose(1, 2, 0)
            data_fixed = test_data['F'].squeeze().cpu().numpy().transpose(1, 2, 0)

            vol_def = defm_frames[-1] * 255.0
            vol_fix = (data_fixed + 1) / 2.0 * 255.0
            mse_val = Metrics.mse_metric(vol_def, vol_fix)

            disp_field = flow_frames_ES.transpose(3, 2, 0, 1)
            jac_folds, min_jac, max_jac = Metrics.jacobian_determinant_3d(disp_field) 
            
            mseList.append(mse_val)
            jacList.append(jac_folds)
            jminList.append(min_jac) 
            jmaxList.append(max_jac) 

            print('---- Original Dice: %03f | Deformed Dice: %03f' % (np.mean(vals_origin), np.mean(vals_regist)))
            print('---- MSE: %.2f | Jacobian Folds (<=0): %.4f %% | Min J: %.4f | Max J: %.4f' % (mse_val, jac_folds, min_jac, max_jac))
            print('---- Deform Time: %.3f s' % (time2 - time1))

            data_origin = test_data['M'].squeeze().cpu().numpy().transpose(1, 2, 0)
            data_fixed = test_data['F'].squeeze().cpu().numpy().transpose(1, 2, 0)
            label_origin = test_data['MS'].squeeze().cpu().numpy()
            label_fixed = test_data['FS'].squeeze().cpu().numpy()

            dpt = 65

            # rescale
            img_mov = ((data_origin[:, :, dpt] + 1) / 2. * 255).astype(np.uint8)
            img_fix = ((data_fixed[:, :, dpt] + 1) / 2. * 255).astype(np.uint8)
            img_def = (defm_frames[-1][:, :, dpt] * 255).astype(np.uint8)

            diff_mf = np.clip((img_mov.astype(float) - img_fix.astype(float)) / 2.0 + 128, 0, 255).astype(np.uint8)
            diff_df = np.clip((img_def.astype(float) - img_fix.astype(float)) / 2.0 + 128, 0, 255).astype(np.uint8)

            save_image(img_mov, os.path.join(result_path, '%s_1_moving.png' % dataName))
            save_image(img_fix, os.path.join(result_path, '%s_2_fixed.png' % dataName))
            save_image(img_def, os.path.join(result_path, '%s_3_deformed.png' % dataName))
            save_image(diff_mf, os.path.join(result_path, '%s_4_diff_mov_fix.png' % dataName))
            save_image(diff_df, os.path.join(result_path, '%s_5_diff_def_fix.png' % dataName))

            # time register
            registTime.append(time2 - time1)

        omdice, osdice = np.mean(originDice), np.std(originDice)
        mdice, sdice = np.mean(registDice), np.std(registDice)
        mtime, stime = np.mean(registTime), np.std(registTime)
        mmse, smse = np.mean(mseList), np.std(mseList)
        mjac, sjac = np.mean(jacList), np.std(jacList)
        mjmin, sjmin = np.mean(jminList), np.std(jminList)
        mjmax, sjmax = np.mean(jmaxList), np.std(jmaxList)

        print()
        print('---------------------------------------------')
        print('Total Metrics for Table----------------------')
        print('---------------------------------------------')
        print('Original Dice | mean = %.3f, std= %.3f' % (omdice, osdice))
        print('Deform Dice   | mean = %.3f, std= %.3f' % (mdice, sdice))
        print('Jac Negatives | mean = %.4f %%, std = %.4f %%' % (mjac, sjac))
        print('Min J         | mean = %.4f, std = %.4f' % (mjmin, sjmin))
        print('Max J         | mean = %.4f, std = %.4f' % (mjmax, sjmax))
        print('Deform Time   | mean = %.3f, std= %.3f' % (mtime, stime))
