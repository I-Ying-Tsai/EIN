import yaml
import argparse
from utils.dataloader import *
from utils.tools import *


if __name__ == '__main__':

    dataset = 'DRWeibo'
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_filename', default='configs/EIN/' + dataset +'.yaml', 
                    type=str, help='the configuration to use')
    args = parser.parse_args()

    print(f'Starting experiment with configurations in {args.config_filename}...')
    
    configs = yaml.load(
        open(args.config_filename), 
        Loader=yaml.FullLoader
    )
    
    args = argparse.Namespace(**configs)

    for i in range(5):
        args.seed = i
        exec('from supervisor import EIN_' + args.base_model + '_supervisor')
        exec('EIN_' + args.base_model + '_supervisor(args)')
