def Get_All_Sentances(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def Get_or_Build_Tokenizer(config, ds, lang):
    tokenizer_path = Get_Tokenizer_File_Path(config, lang)
    if not Path(tokenizer_path).exists():
        print('Tokenizer not found, building...')
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(Get_All_Sentances(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        print('Tokenizer found, loading...')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def Get_Dataset(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # Built tokenizer
    tokenizer_src = Get_or_Build_Tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = Get_or_Build_Tokenizer(config, ds_raw, config['lang_tgt'])

    # Split train and validation sets
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) -  train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])


    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentance: {max_len_src}')
    print(f'Max length of target sentance: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def Get_Transformer(config, src_vocab_size, tgt_vocab_size):
    model = BuildTransformer(src_vocab_size, tgt_vocab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def Get_DeepHead(config, src_vocab_size, tgt_vocab_size):
    model = BuildDeepHead(src_vocab_size, tgt_vocab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def Train_model(config):
    # setting device to train on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Create file for parameters to be saved to
    model_folder = Get_Model_File_Path(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    # Initialize dataloaders and tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = Get_Dataset(config)

    # Initialize model
    model = Get_DeepHead(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Create a tensorboard for loss visualization
    writer = SummaryWriter(config['experiment_name'])

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Initial epoch start and global stop of this training cycle
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    # preload previous model parameters if they exist
    if preload == "latest":
      model_filename = Latest_Weights_File_Path(config)
      print(f'Preloading model {model_filename}')
      state = torch.load(model_filename)  # loading previous model state
      initial_epoch = state['epoch'] + 1
      model.load_state_dict(state['model_state_dict'])
      optimizer.load_state_dict(state['optimizer_state_dict'])
      global_step = state['global_step']
      print(f'Starting training at epoch {initial_epoch}')
    else:
      model_filename = Get_Model_File_Path(config)
      config['preload'] = 'latest'
      print(f'No model found to preload')
      print(f'Starting training at epoch {initial_epoch}')

    # Initializing loss function
    # Ignores any padded tokens in loss computation
    # Smoothes result by distributing .1 of argmax value to all other labels
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Training loop
    print('Beginning training...')
    for epoch in range(initial_epoch, initial_epoch + config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        # Initialize batch iterator using tqdm for a progression bar visualization
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            if batch == None:
                continue
            encoder_input = batch['encoder_input'].to(device)   # input: (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)   # input: (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)     # (batch, 1, 1, seq_len) to mask padded tokens
            decoder_mask = batch['decoder_mask'].to(device)     # (batch, 1, seq_len, seq_len) to mask padded tokens and future words

            # run tensors through model
            encoder_output = model.encode(encoder_input, encoder_mask)  # (batch, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)    # (batch, seq_len, d_model)
            projection_output = model.projection(decoder_output)    # (batch, seq_len, tgt_vocab_size)
            # Get label from batch
            label = batch['label'].to(device)

            # input (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            # update the progress bar with the loss
            batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'})

            # log the loss in tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropogate the loss
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save the model at the end of every epoch
        model_filename = Get_Weights_File_Path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

        # Run validation at the end of each epoch
        Run_Validation(model, device, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], lambda msg: batch_iterator.write(msg), global_step, writer)
    print('\nTraining complete...')

if __name__=="__main__":
    config = Get_Config()
    Train_model(config)
