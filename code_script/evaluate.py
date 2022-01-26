import torch
from collections import Counter
import numpy as np
from tqdm import tqdm
from modeling import get_model, get_tokenizer
from data_prompt import REPromptDataset
from arguments import get_args_parser
from templating import get_temps
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def evaluate(model, dataset, dataloader):
    model.eval()
    scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            logits = model(**batch)
            res = []
            for i in dataset.prompt_id_2_label:
                _res = 0.0
                for j in range(len(i)):
                    _res += logits[j][:, i[j]]
                _res = _res.detach().cpu()
                res.append(_res)
            logits = torch.stack(res, 0).transpose(1, 0)
            labels = batch['labels'].detach().cpu().tolist()
            all_labels += labels
            scores.append(logits.cpu().detach())
        scores = torch.cat(scores, 0)
        scores = scores.detach().cpu().numpy()
        all_labels = np.array(all_labels)
        np.save("scores.npy", scores)
        np.save("all_labels.npy", all_labels)

        pred = np.argmax(scores, axis=-1)
        # acc, f1, conf_mat = f1_score(pred, all_labels, dataset.num_class,
        #                              dataset.NA_NUM)
        acc = accuracy_score(y_true=all_labels, y_pred=pred)
        mi_f1 = f1_score(y_true=all_labels, y_pred=pred, average='micro')
        ma_f1 = f1_score(y_true=all_labels, y_pred=pred, average='macro')
        conf_mat = confusion_matrix(y_true=all_labels, y_pred=pred)
    return acc, mi_f1, ma_f1, conf_mat


args = get_args_parser()
tokenizer = get_tokenizer(special=[])
temps = get_temps(tokenizer)

train_dataset = REPromptDataset.load(
    path=args.output_dir,
    name="train",
    temps=temps,
    tokenizer=tokenizer,
    rel2id=args.data_dir + "/" + "rel2id.json")

test_dataset = REPromptDataset.load(
    path=args.output_dir,
    name="test",
    temps=temps,
    tokenizer=tokenizer,
    rel2id=args.data_dir + "/" + "rel2id.json")
test_dataset.cuda()
test_sampler = SequentialSampler(test_dataset)
train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=train_batch_size)

model = get_model(tokenizer, train_dataset.prompt_label_idx)

model.load_state_dict(torch.load(args.output_dir + "/" + 'parameter4.pkl'))
acc, mi_f1, ma_f1, conf_mat = evaluate(model, test_dataset, test_dataloader)

print(f"acc: {acc}, micro-f1: {mi_f1}, macro-f1: {ma_f1}")
print(conf_mat)
