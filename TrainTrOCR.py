import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from datasets import load_metric
import jiwer

file_path = "SyntheticData/labels.json"
processor_name = "microsoft/trocr-base-handwritten"
model_name = "microsoft/trocr-base-stage1"
image_dir = "SyntheticData/images/"

processor = TrOCRProcessor.from_pretrained(processor_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)


def get_image_data(json_path):
    df = pd.read_json(json_path)

    train_df, test_df = train_test_split(df, test_size=0.2)
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    return train_df, test_df


class CalculusDataset(Dataset):
    def __init__(self, image_dir, df, processor, max_target_length=1280):
        self.image_dir = image_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['filename'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.image_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


def create_datasets(train_df, test_df):
    train = CalculusDataset(image_dir=image_dir,
                            df=train_df,
                            processor=processor)
    eval = CalculusDataset(image_dir=image_dir,
                           df=test_df,
                           processor=processor)
    print("Number of training examples:", len(train))
    print("Number of validation examples:", len(eval))
    return train, eval


def train_model(train_dataset, eval_dataset):
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=False,  # Can only be done on CUDA
        output_dir="./",
        logging_steps=2,
        save_steps=1000,
        eval_steps=200,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()


def compute_metrics(pred):
    cer_metric = load_metric("cer")
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


if __name__ == '__main__':
    train_df, test_df = get_image_data(file_path)
    train_dataset, eval_dataset = create_datasets(train_df, test_df)
    train_model(train_dataset, eval_dataset)
