from pathlib import Path

def Get_Config():
    return{
        "batch_size": 16,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "es",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "output_path": "/output/models/",
        "preload": "latest",
        "tokenizer_filename": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def Get_Weights_File_Path(config, epoch: str):
    output_path = config['output_path']
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / output_path / model_folder / model_filename)

def Get_Tokenizer_File_Path(config, lang: str):
    output_path = config['output_path']
    tokenizer_filename = Path(config['tokenizer_filename'].format(lang))
    return str(Path('.') / output_path / tokenizer_filename)

def Get_Model_File_Path(config):
    output_path = config['output_path']
    model_folder = config['model_folder']
    return str(Path('.') / output_path / model_folder)

# Find the latest weights file in the weights folder
def Latest_Weights_File_Path(config):
    output_path = config['output_path']
    model_folder = config['model_folder']
    model_path = str(Path('.') / output_path / model_folder)
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_path).glob(model_filename))
    if len(weights_files) == 0:
        print(f"Number of files in preload file: {weights_files}")
        return None
    weights_files.sort()
    return str(weights_files[-1])