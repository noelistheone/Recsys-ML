from SELFRec import SELFRec
from util.conf import ModelConf
import time
import argparse
import os

def print_models(title, models):
    print(f"{'=' * 80}\n{title}\n{'-' * 80}")
    for category, model_list in models.items():
        print(f"{category}:\n   {'   '.join(model_list)}\n{'-' * 80}")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SELFRec: A library for self-supervised recommendation')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to configuration file (e.g., ./conf/PT4Rec.yaml)')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Model name (alternative to --config, for backward compatibility)')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Dataset name (optional, overrides config file)')
    args = parser.parse_args()
    
    models = {
        'Graph-Based Baseline Models': ['LightGCN', 'DirectAU', 'MF', 'UserKNN', 'ItemKNN'],
        'Self-Supervised Graph-Based Models': ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL', 'MixGCF', 'CPTPP', 'PT4Rec', 'PT4Rec_Unlearn', 'PT4Rec_Pop', 'PT4Rec_Adv', 'PT4Rec_Enhanced'],
        'Sequential Baseline Models': ['SASRec'],
        'Self-Supervised Sequential Models': ['CL4SRec', 'BERT4Rec']
    }

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    
    config_path = None
    model = None
    
    # Mode 1: Specify config file directly
    if args.config:
        config_path = args.config
        if not os.path.exists(config_path):
            print(f"Error: Configuration file '{config_path}' not found!")
            exit(-1)
        
        # Load config to get model name
        conf = ModelConf(config_path)
        model = conf['model']['name']
        print(f"Using configuration: {config_path}")
        print(f"Model: {model}")
        print(f"Dataset: {conf['training.set']}")
        print('=' * 80)
    
    # Mode 2: Specify model name (backward compatible)
    elif args.model:
        model = args.model
        all_models = sum(models.values(), [])
        if model not in all_models:
            print(f"Error: Unknown model '{model}'!")
            print_models("Available Models", models)
            exit(-1)
        config_path = f'./conf/{model}.yaml'
        if not os.path.exists(config_path):
            print(f"Error: Configuration file '{config_path}' not found!")
            exit(-1)
        conf = ModelConf(config_path)
        print(f"Model: {model}")
        print(f"Using configuration: {config_path}")
        print('=' * 80)
    
    # Mode 3: Interactive mode (original behavior)
    else:
        print_models("Available Models", models)
        model = input('Please enter the model you want to run:')
        all_models = sum(models.values(), [])
        if model not in all_models:
            print('Wrong model name!')
            exit(-1)
        config_path = f'./conf/{model}.yaml'
        conf = ModelConf(config_path)

    # Override dataset if specified
    if args.dataset and args.dataset != conf.get('training.set', ''):
        print(f"Note: Dataset override not implemented yet. Using dataset from config file.")

    # Execute
    s = time.time()
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print(f"\n{'=' * 80}")
    print(f"Running time: {e - s:.2f} s")
    print('=' * 80)
