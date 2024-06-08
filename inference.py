import argparse

import torch

from utils import load_dataset

device = 'cpu'

def model_inference(model, trg_tokenizer, trg_init_idx, trg_eos_idx, input, max_seq_len=500):
    model.eval()
    initial_output = torch.LongTensor([[trg_init_idx]])
    output = initial_output
    
    for _ in range(max_seq_len):
        output = torch.log_softmax(model(input, output), dim=-1)
        output = output.argmax(dim=-1)
        output = torch.cat((initial_output, output), dim=-1)
        if output[0, -1].item() == trg_eos_idx:
            break
    return trg_tokenizer.decode(output[0].tolist()).replace("Ġ", "")

def inference(saves_dir_name,
              dataset_name,
              max_seq_len,
              **model_params):
    model = torch.load(model_params['model_path'],
                       map_location=device, )

    dataset, src_tokenizer, trg_tokenizer, src_lambda, trg_lambda = load_dataset(dataset_name, saves_dir_name)

    trg_init_idx = trg_tokenizer.token_to_id("<s>")
    trg_eos_idx = trg_tokenizer.token_to_id("</s>")

    while True:

        input_seq = input("Input the phrase: ")
        
        initial_source = torch.as_tensor(
                    [
                        src_tokenizer.encode('<s>'+input_seq+'</s>').ids
                    ][: max_seq_len] ,
                        device=device
        )

        output = model_inference(model, trg_tokenizer, trg_init_idx, trg_eos_idx, initial_source, max_seq_len=max_seq_len)

        print("Source : ", src_tokenizer.decode(initial_source[0].tolist()).replace("Ġ", ""))
        print(f"Output (of length {len(output)}) : ", output)
        print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', nargs='?', type=str)
    parser.add_argument("--saves-dir-name", type=str, default="saves")
    parser.add_argument("--dataset-name", type=str, nargs="?")

    parser.add_argument('--max-seq-len', nargs='?', type=int, default=400)

    args = vars(parser.parse_args())

    inference(**args)
