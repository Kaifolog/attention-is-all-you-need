import argparse

import torch
from torch.utils.data import DataLoader
import tqdm
import evaluate

from utils import load_dataset, BatchProcessor
from inference import model_inference

device = 'cpu'

def test(saves_dir_name,
              dataset_name,
              e_ls,
              batch_size,
              max_seq_len,
              **model_params):
    model = torch.load(model_params['model_path'],
                       map_location=device, )

    dataset, src_tokenizer, trg_tokenizer, src_lambda, trg_lambda = load_dataset(dataset_name, saves_dir_name)

    trg_init_idx = trg_tokenizer.token_to_id("<s>")
    trg_eos_idx = trg_tokenizer.token_to_id("</s>")

    src_padding_idx = src_tokenizer.token_to_id("<pad>")
    trg_padding_idx = src_tokenizer.token_to_id("<pad>")

    src_vocab_size = src_tokenizer.get_vocab_size()
    trg_vocab_size = trg_tokenizer.get_vocab_size()

    criterion = torch.nn.CrossEntropyLoss(  ignore_index=trg_padding_idx, 
                                            label_smoothing=e_ls, 
                                            reduction="sum"
                                          )

    valid_dl = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BatchProcessor(src_tokenizer, trg_tokenizer,
                                  src_lambda, trg_lambda),
    )

    bleu = evaluate.load("bleu")
    bleu_max_order = 2
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')

    eval_bleu = eval_rouge = eval_meteor = eval_loss = 0
    n = batches_num = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(valid_dl):
            oo = model(batch["src"], batch["trg"][:, :-1])
            loss = criterion(oo.view(-1, trg_vocab_size), batch["trg"][:, 1:].flatten())

            encoded_output = [[i for i in replica.tolist() if i != 0] for replica in batch["trg"][:, 1:]]
            decoded_output = [[trg_tokenizer.decode(replica).replace("Ġ", "")] for replica in encoded_output]
            decoded_result = [model_inference(model, trg_tokenizer, trg_init_idx, trg_eos_idx, replica.unsqueeze(-2), max_seq_len=max_seq_len) for replica in batch["src"]]

            eval_loss += loss.item()
            n += batch["trg"].ne(trg_padding_idx).sum().item()

            eval_bleu += bleu.compute(predictions=decoded_result, references=decoded_output, max_order=bleu_max_order)['bleu']
            eval_rouge += rouge.compute(predictions=decoded_result, references=decoded_output)['rougeL']
            eval_meteor += meteor.compute(predictions=decoded_result, references=decoded_output)['meteor']
            batches_num += 1

            

    print(f"Total Evaluation Loss : {eval_loss / n}")
    print(f"Total Evaluation BLEU-{bleu_max_order} : {eval_bleu / batches_num}")
    print(f"Total Evaluation ROUGE-L : {eval_rouge / batches_num}")
    print(f"Total Evaluation METEOR : {eval_meteor / batches_num}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', nargs='?', type=str)
    parser.add_argument("--saves-dir-name", type=str, default="saves")
    parser.add_argument("--dataset-name", type=str, nargs="?")
    parser.add_argument("--batch-size", type=int, nargs="?")
    parser.add_argument("--e-ls", type=float, nargs="?", help='Label smoothing value')

    parser.add_argument('--max-seq-len', nargs='?', type=int, default=400)

    args = vars(parser.parse_args())

    test(**args)
