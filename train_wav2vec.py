import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from datasets.load import load_dataset, load_metric
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

# disable HF thousand warnings
warnings.simplefilter("ignore")
# set os environ variable for multiprocesses
os.environ["PYTHONWARNINGS"] = "ignore"


def join_phones(batch):
    phones_list = batch["phonetic_detail"]["utterance"]
    phones_list.pop(0)
    phones_list.pop(-1)
    phones = " ".join(phones_list)

    return {"phones": phones}


def extract_all_chars(batch):
    all_text = " ".join(batch["phones"])
    vocab = list(set(all_text.split(" ") + ["|"]))
    return {"vocab": [vocab], "all_text": [all_text]}


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["phones"]).input_ids
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or
        :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`,
        defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's
            padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch
              (or no padding if only a single equence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument
              :obj:`max_length` or to the maximum acceptable input length for the model
              if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can
             output a batch with sequences of different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally
            padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding
            length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA
            hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(  # type: ignore
            labels_batch.attention_mask.ne(1), -100  # type: ignore
        )

        batch["labels"] = labels  # type: ignore

        return batch  # type: ignore


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[
        pred.label_ids == -100
    ] = processor.tokenizer.pad_token_id  # type: ignore

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == "__main__":
    from typing import cast

    from datasets.dataset_dict import DatasetDict

    timit: DatasetDict = cast(
        DatasetDict, load_dataset("timit_asr", data_dir="../data/timit")
    )
    timit = timit.remove_columns(
        ["dialect_region", "sentence_type", "speaker_id", "id", "word_detail", "text"]
    )
    timit = timit.map(join_phones, remove_columns="phonetic_detail")

    vocabs = timit.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=timit.column_names["train"],
    )

    vocab_list = list(
        set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0])
    )

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open("vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer(
        "./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    timit_prepared = timit.map(
        prepare_dataset, remove_columns=timit.column_names["train"], num_proc=4
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load_metric("wer")

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,  # type: ignore
        vocab_size=processor.tokenizer.vocab_size,  # type: ignore
    )

    model.freeze_feature_encoder()  # type: ignore

    training_args = TrainingArguments(
        output_dir="wav2vec2-base-timit-psm",
        group_by_length=True,
        per_device_train_batch_size=64,
        evaluation_strategy="steps",
        num_train_epochs=220,
        fp16=True,
        save_steps=250,
        eval_steps=250,
        logging_steps=250,
        learning_rate=6e-4,
        weight_decay=0.005,
        warmup_steps=500,
        save_total_limit=2,
        dataloader_num_workers=16,
    )

    trainer = Trainer(
        model=model,  # type: ignore
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=timit_prepared["train"],  # type: ignore
        eval_dataset=timit_prepared["test"],  # type: ignore
        tokenizer=processor.feature_extractor,  # type: ignore
    )

    trainer.train()
