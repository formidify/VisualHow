import os
import argparse
import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='wikihow',
                        help='coco or f30k')
    parser.add_argument('--data_path', default='/opt/xianyu/CVPR_2022/data/wikihow')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--evaluate_cxc', action='store_true')
    opt = parser.parse_args()

    if opt.dataset == 'coco':
        weights_bases = [
            'runs/release_weights/coco_butd_region_bert',
            'runs/release_weights/coco_butd_grid_bert',
            'runs/release_weights/coco_wsl_grid_bert',
        ]
    elif opt.dataset == 'f30k':
        weights_bases = [
            'runs/release_weights/f30k_butd_region_bert',
            'runs/release_weights/f30k_butd_grid_bert',
            'runs/release_weights/f30k_wsl_grid_bert',
        ]
    elif opt.dataset == 'wikihow':
        weights_bases = [
            'runs/wikihow_wsl_grid_bert',
        ]
    else:
        raise ValueError('Invalid dataset argument {}'.format(opt.dataset))

    for base in weights_bases:
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        if opt.save_results:  # Save the final results for computing ensemble results
            save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        result_path = os.path.join(opt.data_path, 'retrieving_results')
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if opt.dataset == 'coco':
            if not opt.evaluate_cxc:
                # Evaluate COCO 5-fold 1K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
                # Evaluate COCO 5K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
            else:
                # Evaluate COCO-trained models on CxC
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True, cxc=True)
        elif opt.dataset == 'f30k':
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)

        elif opt.dataset == 'wikihow':
            # Evaluate wikihow
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path,
                                result_path=result_path)




if __name__ == '__main__':
    main()
