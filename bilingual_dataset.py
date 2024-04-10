import torch

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_txt = src_target_pair['translation'][self.src_lang]
        tgt_txt = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_txt).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_txt).ids

        enc_num_pad_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_pad_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_pad_tokens < 0:
            enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
            enc_num_pad_tokens = 0
        if dec_num_pad_tokens < 0:
            dec_input_tokens = dec_input_tokens[:self.seq_len - 1]
            dec_num_pad_tokens = 0

        # Add SOS and EOS to source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Add SOS to decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Add SOS to the label (expected decoder output)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Check to ensure the sizes of encoder, decoder, and label texts are of sequence length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # data set
        return {
            "encoder_input": encoder_input, # size = seq_len
            "decoder_input": decoder_input, # size = seq_len
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & Causual_Mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "source_text": src_txt,
            "target_text": tgt_txt
        }

# returns a masked matrix with all values above diagonal masked out
def Causual_Mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
