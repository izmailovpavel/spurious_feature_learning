from transformers import AlbertForSequenceClassification
from transformers import BertForSequenceClassification
from transformers import DebertaV2ForSequenceClassification
import types
import torch


def _bert_replace_fc(model):
    model.fc = model.classifier
    delattr(model, "classifier")

    def classifier(self, x):
        return self.fc(x)
    
    model.classifier = types.MethodType(classifier, model)

    model.base_forward = model.forward

    def forward(self, x):
        return self.base_forward(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2]).logits

    model.forward = types.MethodType(forward, model)
    return model


def bert_pretrained(output_dim):
	return _bert_replace_fc(BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=output_dim))


def bert_pretrained_multilingual(output_dim):
    return _bert_replace_fc(BertForSequenceClassification.from_pretrained(
            'bert-base-multilingual-uncased', num_labels=output_dim))


def bert(output_dim):
    config_class = BertForSequenceClassification.config_class
    config = config_class.from_pretrained(
            'bert-base-uncased', num_labels=output_dim)
    return _bert_replace_fc(BertForSequenceClassification(config))


def bert_large_pretrained(output_dim):
    return _bert_replace_fc(BertForSequenceClassification.from_pretrained(
            'bert-large-uncased', num_labels=output_dim))


def deberta_pretrained(output_dim):
    return _bert_replace_fc(DebertaV2ForSequenceClassification.from_pretrained(
            'microsoft/deberta-v3-base', num_labels=output_dim))


def deberta_large_pretrained(output_dim):
    return _bert_replace_fc(DebertaV2ForSequenceClassification.from_pretrained(
            'microsoft/deberta-v3-large', num_labels=output_dim))


def albert_pretrained(output_dim):
    return _bert_replace_fc(AlbertForSequenceClassification.from_pretrained(
            'albert-base-v2', num_labels=3))
