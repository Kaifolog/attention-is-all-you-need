import argparse
import os
import time

import torch
import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import VanillaTransformer
from utils import load_dataset, BatchProcessor, print_model_info

torch.manual_seed(42)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(device)
print(f"Using {device} device")


def train(
    saves_dir_name,
    epoch,
    batch_size,
    dataset_name,
    max_seq_len,
    e_ls,
    **model_params,
):

    if not os.path.exists(saves_dir_name):
        os.mkdir(saves_dir_name)

    print(f"Train : {epoch=}, {batch_size=}")

    dataset, src_tokenizer, trg_tokenizer, src_lambda, trg_lambda = load_dataset(dataset_name, saves_dir_name)

    src_padding_idx = src_tokenizer.token_to_id("<pad>")
    trg_padding_idx = src_tokenizer.token_to_id("<pad>")

    src_vocab_size = src_tokenizer.get_vocab_size()
    trg_vocab_size = trg_tokenizer.get_vocab_size()

    if 'load_from' in model_params.keys():
        print(f"Loading {model_params['load_from']} model from save")
        model = torch.load( model_params['load_from'],
                            map_location=device, )
    else:
        model = VanillaTransformer(
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
            src_padding_idx=src_padding_idx,
            trg_padding_idx=trg_padding_idx,
            max_seq_len=max_seq_len,
            **model_params,
        )

        print_model_info(model)

        # init weights
        for param in model.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    
    opt = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.98), eps=1e-9,
    )
    lr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=model_params['warmup_steps'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=trg_padding_idx, 
                                          label_smoothing=e_ls, 
                                          reduction="sum"
                                          )

    if 'subset_size' in model_params.keys():
        subset_size = model_params['subset_size']
        subset_indices = torch.randperm(len(dataset["train"]))[:subset_size]
        subset_dataset = torch.utils.data.Subset(dataset["train"], subset_indices)
        train_dl = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=BatchProcessor(src_tokenizer, trg_tokenizer,
                                    src_lambda, trg_lambda),
            generator=torch.Generator(device=device),
        )
    else:
        train_dl = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=BatchProcessor(src_tokenizer, trg_tokenizer,
                                    src_lambda, trg_lambda),
            generator=torch.Generator(device=device),
        )


    losses_y = losses_y_epoc = []
    model.train()
    for e in range(epoch):
        epoch_loss = n = 0
        for batch in (pbar := tqdm.tqdm(train_dl)):
            opt.zero_grad()

            output = model(batch["src"], batch["trg"][:, :-1])
            loss = criterion(
                output.view(-1, trg_vocab_size), batch["trg"][:, 1:].flatten()
            )

            loss.backward()
            opt.step()
            lr.step()

            epoch_loss += loss.item()
            n += batch["trg"].ne(trg_padding_idx).sum().item()

            losses_y.append(float(loss.item() / batch["trg"].ne(trg_padding_idx).sum().item()))
            losses_y_epoc.append(float(loss.item() / batch["trg"].ne(trg_padding_idx).sum().item()))
            pbar.set_postfix(
                loss = float(loss.item() / batch["trg"].ne(trg_padding_idx).sum().item()), 
                warmup_lr = lr.get_last_lr()[0]
            )
        if 'save_every_n_epoch' in model_params.keys() and e % model_params['save_every_n_epoch']==0 and e!=0:
            torch.save(model, f"{saves_dir_name}/model_{str(int(time.time()))}_{e-model_params['save_every_n_epoch']}_{e}.pb")
            plt.plot(list(range(len(losses_y_epoc))), losses_y_epoc)
            plt.title(f"n={model_params['n']}, d_model={model_params['d_model']}, d_ff={model_params['d_ff']}, \nh={model_params['h']}, {batch_size=}")
            plt.savefig(f"{saves_dir_name}/model_{str(int(time.time()))}_{e-model_params['save_every_n_epoch']}_{e}.jpg")
            plt.clf()
            losses_y_epoc = []

        epoch_loss = epoch_loss / n
        print(f"Epoch {e + 1}/{epoch} Train Loss : {epoch_loss}")
    torch.save(model, saves_dir_name+'/model_'+str(int(time.time()))+'.pb')

    plt.plot(list(range(len(losses_y))), losses_y)
    plt.title(f"n={model_params['n']}, d_model={model_params['d_model']}, d_ff={model_params['d_ff']}, \nh={model_params['h']}, {batch_size=}")
    plt.savefig(f"./{saves_dir_name}/model_{str(int(time.time()))}.jpg")
    plt.show()

    model.eval()
    eval_loss = n = 0
    valid_dl = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BatchProcessor(src_tokenizer, trg_tokenizer,
                                  src_lambda, trg_lambda),
    )
    with torch.no_grad():
        for batch in tqdm.tqdm(valid_dl):
            oo = model(batch["src"], batch["trg"][:, :-1])
            loss = criterion(oo.view(-1, trg_vocab_size), batch["trg"][:, 1:].flatten())

            eval_loss += loss.item()
            n += batch["trg"].ne(trg_padding_idx).sum().item()
    eval_loss /= n
    print(f"Total Evaluation Loss : {eval_loss}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    ## model params
    parser.add_argument("--n", type=int, nargs="?", help='Number of layers in decoder and encoder')
    parser.add_argument("--d-model", type=int, nargs="?")
    parser.add_argument("--d-ff", type=int, nargs="?")
    parser.add_argument("--h", type=int, nargs="?")
    parser.add_argument("--p-drop", type=float, nargs="?", default=0.1)
    parser.add_argument('--max-seq-len', nargs='?', type=int, default=400)

    ## train hyperparams
    parser.add_argument("--dataset-name", type=str, nargs="?", choices=['wmt/wmt14', 
                                                                        'benjaminbeilharz/daily_dialog_w_turn_templates',
                                                                        'goendalf666/sales-conversations',
                                                                        'local-merge/sales-dailydialog'])
    parser.add_argument("--subset-size", type=int, nargs="?", help='Use if you want to try the model on a subset of the dataset.')
    parser.add_argument("--epoch", type=int, nargs="?", default=1)    
    parser.add_argument("--batch-size", type=int, nargs="?")
    parser.add_argument("--e-ls", type=float, nargs="?", help='Label smoothing value', default=0.1)

    parser.add_argument("--save-every-n-epoch", type=int, nargs="?")
    parser.add_argument("--load-from", type=str, nargs="?")

    ## warmup params
    # parser.add_argument("--warmup", type=int, nargs="?", default='cosine', choices=['cosine',])
    parser.add_argument("--warmup-steps", type=int, nargs="?")

    ## env params
    parser.add_argument("--saves-dir-name", type=str, default="saves")

    args = vars(parser.parse_args())
    print(args)

    train(**args)
