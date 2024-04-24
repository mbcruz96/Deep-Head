import model
from bilingual_dataset import Causual_Mask
import torch

def Greedy_Decode(model, device, encoder_input, encoder_mask, src_tokenizer, tgt_tokenizer, max_len):
    sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    eos_idx = tgt_tokenizer.token_to_id('[EOS]')

    # Precompute encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(encoder_input, encoder_mask)
    # Initialize the decoder output with the SOS token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(encoder_input).to(device)

    while True:
        # Break if decoder is max sequence length
        if decoder_input.size(1) == max_len:
            break

        # build mask for the current decoder input
        decoder_mask = Causual_Mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

        # calculate the decoder output
        output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)

        # get the next token
        prob = model.projection(output[:, -1]) # project the output for the last token in the sequence
        # select the token with the next highest probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1)

        # break if next word is eos token
        if next_word == eos_idx:
            break

    # return generated sequence without batch dimension
    return decoder_input.squeeze(0)

def beam_search_decode(model, device, beam_size, source, source_mask, tokenizer_src, tokenizer_tgt, max_len):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # Create a candidate list
    candidates = [(decoder_initial_input, 1)]

    while True:

        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break

        # Create a new list of candidates
        new_candidates = []

        for candidate, score in candidates:

            # Do not expand candidates that have reached the eos token
            if candidate[0][-1].item() == eos_idx:
                continue

            # Build the candidate's mask
            candidate_mask = Causual_Mask(candidate.size(1)).type_as(source_mask).to(device)
            # calculate output
            out = model.decode(encoder_output, source_mask, candidate, candidate_mask)
            # get next token probabilities
            prob = model.project(out[:, -1])
            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                # create a new candidate by appending the token to the current candidate
                new_candidate = torch.cat([candidate, token], dim=1)
                # We sum the log probabilities because the probabilities are in log space
                new_candidates.append((new_candidate, score + token_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]

        # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze()

def Run_Validation(model, device, val_ds, src_tokenizer, tgt_tokenizer, max_len, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    console_width = 80 # size of control window

    with torch.no_grad():
        for batch in val_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            # ensure validation batch is 1
            assert encoder_input.size(0) == 1, 'Batch size must be 1 for validation'

            # generate the output tokens
            beam_size = 4
            model_out = beam_search_decode(model, device, beam_size, encoder_input, encoder_mask, src_tokenizer, tgt_tokenizer, max_len)

            # get actual model output
            source_text = batch['source_text'][0]
            target_text = batch['target_text'][0]
            model_out_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy())

            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

            if writer:
                # Evaluate the character error rate
                # Compute the char error rate
                metric = torchmetrics.CharErrorRate()
                cer = metric(model_out_text, target_text)
                writer.add_scalar('validation cer', cer, global_step)
                writer.flush()

                # Compute the word error rate
                metric = torchmetrics.WordErrorRate()
                wer = metric(model_out_text, target_text)
                writer.add_scalar('validation wer', wer, global_step)
                writer.flush()

                # Compute the BLEU metric
                metric = torchmetrics.BLEUScore()
                bleu = metric(model_out_text, target_text)
                writer.add_scalar('validation BLEU', bleu, global_step)
                writer.flush()
