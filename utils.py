import torch

import datasets
import tokenizers
import tokenizers.normalizers

def construct_tokenizer(data, name, vocab_size, lowercase=False):
    try:
        tokenizer = tokenizers.Tokenizer.from_file(name)
        print(f'Loading {name} tokenizer from save')
    except:
        tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
        if lowercase:
            tokenizer.normalizer = tokenizers.normalizers.Lowercase()
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()
        tokenizer.train_from_iterator(
            data,
            trainer=tokenizers.trainers.BpeTrainer(
                special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
                vocab_size=vocab_size,
                min_frequency=2,
            ),
        )
        tokenizer.enable_padding()
        tokenizer.save(name)
    finally:
        return tokenizer

def rebuild_sales_conversations(dataset):
    # Function flattens columns '1' ... '18' --to-> 'first', 'second'
    customer_replicas = []
    salesman_replicas = []
    def rebuild_sales_conversation(example):
        customer_start = 'Customer: '   #
        salesman_start = 'Salesman: '   # both have length 10
        customer_replica = []
        salesman_replica = []
        for relica in example.values():
            if relica == None:
                break
            if customer_start in relica:
                customer_replica.append(relica[10:])
            else:
                salesman_replica.append(relica[10:])
        # trimmering
        if len(salesman_replica) > len(customer_replica):
            salesman_replica = salesman_replica[:len(customer_replica)]
        customer_replicas.extend(customer_replica)
        salesman_replicas.extend(salesman_replica)
        return example
    dataset.map(rebuild_sales_conversation, keep_in_memory=True)
    dataset = datasets.Dataset.from_dict({"first": customer_replicas, 
                                        "second": salesman_replicas}).train_test_split(test_size=0.1)
    return dataset

def load_dataset(dataset_name, saves_dir_name, ):
    dataset_params = {}
    if dataset_name == "wmt/wmt14":
        dataset_params = {
                'path': 'wmt/wmt14',
                'data_files': {    # You can use the full dataset if you have a fancy RTX or Tesla card.
                            'train': 'de-en/train-00000-of-00003.parquet',
                            'test': 'de-en/test-00000-of-00001.parquet', 
                            'validation': 'de-en/validation-00000-of-00001.parquet'
                            },
                'verification_mode': 'no_checks',
            }
    if dataset_name == "benjaminbeilharz/daily_dialog_w_turn_templates":
        dataset_params = {
                'path': 'benjaminbeilharz/daily_dialog_w_turn_templates',
        }
    if dataset_name == 'goendalf666/sales-conversations':
        dataset_params = {
                'path': 'goendalf666/sales-conversations',
        }
    if dataset_name == 'local-merge/sales-dailydialog':
        dataset_params = {
                'path': 'benjaminbeilharz/daily_dialog_w_turn_templates',
        }
    dataset = datasets.load_dataset(**dataset_params)
    
    if dataset_name == "wmt/wmt14":
        src_tokenizer = construct_tokenizer(
            map(lambda x: x["translation"]["en"], dataset["train"]),
            f"{saves_dir_name}/tokenizer_{dataset_name.replace('/','')}_de-en_1.json",
            37000
        )
        trg_tokenizer = construct_tokenizer(
            map(lambda x: x["translation"]["de"], dataset["train"]),
            f"{saves_dir_name}/tokenizer_{dataset_name.replace('/','')}_de-en_2.json",
            37000
        )
        src_lambda = lambda x: '<s>' + x["translation"]["en"] + '</s>'
        trg_lambda = lambda x: '<s>' + x["translation"]["de"] + '</s>'

    if dataset_name == "benjaminbeilharz/daily_dialog_w_turn_templates":
        src_tokenizer = construct_tokenizer(
            map(lambda x: x["first"], dataset["train"]),
            f"{saves_dir_name}/tokenizer_{dataset_name.replace('/','')}_1.json",
            17500,
            lowercase=True
        )
        trg_tokenizer = construct_tokenizer(
            map(lambda x: x["second"], dataset["train"]),
            f"{saves_dir_name}/tokenizer_{dataset_name.replace('/','')}_2.json",
            17500,
            lowercase=True
        )
        src_lambda = lambda x: '<s>' + x["first"] + '</s>'
        trg_lambda = lambda x: '<s>' + x["second"] + '</s>'

    if dataset_name == 'goendalf666/sales-conversations':
        dataset = rebuild_sales_conversations(dataset)
        src_tokenizer = construct_tokenizer(
            map(lambda x: x["first"], dataset["train"]),
            f"{saves_dir_name}/tokenizer_{dataset_name.replace('/','')}_1.json",
            17500,
            lowercase=True
        )
        trg_tokenizer = construct_tokenizer(
            map(lambda x: x["second"], dataset["train"]),
            f"{saves_dir_name}/tokenizer_{dataset_name.replace('/','')}_2.json",
            17500,
            lowercase=True
        )
        src_lambda = lambda x: '<s>' + x["first"] + '</s>'
        trg_lambda = lambda x: '<s>' + x["second"] + '</s>'

    if dataset_name == 'local-merge/sales-dailydialog':
        dataset1 = datasets.load_dataset('goendalf666/sales-conversations')
        dataset1 = rebuild_sales_conversations(dataset1)
        dataset = datasets.DatasetDict({
            'train': datasets.concatenate_datasets([dataset["train"], dataset1["train"]]),
            'test': datasets.concatenate_datasets([dataset["test"], dataset1["test"]]),
        })
        src_tokenizer = construct_tokenizer(
            map(lambda x: x["first"], dataset["train"]),
            f"{saves_dir_name}/tokenizer_{dataset_name.replace('/','')}_1.json",
            20000,
            lowercase=True
        )
        trg_tokenizer = construct_tokenizer(
            map(lambda x: x["second"], dataset["train"]),
            f"{saves_dir_name}/tokenizer_{dataset_name.replace('/','')}_2.json",
            20000,
            lowercase=True
        )
        src_lambda = lambda x: '<s>' + x["first"] + '</s>'
        trg_lambda = lambda x: '<s>' + x["second"] + '</s>'
    
    return dataset, src_tokenizer, trg_tokenizer, src_lambda, trg_lambda

def print_model_info(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total params:', pytorch_total_params)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))
    
class BatchProcessor:
    def __init__(
        self,
        src_tokenizer,
        trg_tokenizer,
        src_lambda,
        trg_lambda,
    ):
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.src_lambda = src_lambda
        self.trg_lambda = trg_lambda

    def __call__(self, items: list[dict]) -> dict:
        return {
            "src": torch.as_tensor(
                [
                    enc.ids
                    for enc in self.src_tokenizer.encode_batch(
                        [self.src_lambda(item) for item in items]
                    )
                ]
            ),
            "trg": torch.as_tensor(
                [
                    enc.ids
                    for enc in self.trg_tokenizer.encode_batch(
                        [self.trg_lambda(item) for item in items]
                    )
                ]
            ),
        }