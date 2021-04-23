from transformers import BertPreTrainedModel, BertModel
import torch


class BertForMD(BertPreTrainedModel): # Metaphor Detection, modified from BertForTokenClassification
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.config.num_labels)
        # self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3], dtype=torch.float32))
        self.loss = torch.nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        word_posi=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)
        batch_size = input_ids.shape[0]

        word_state = torch.empty((0, last_hidden_state.shape[2]), dtype=torch.float32).cuda()
        for i in range(batch_size):
            word_state = torch.cat((word_state, last_hidden_state[i][word_posi[i]].unsqueeze(0)))

        logits = self.classifier(word_state)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss(logits.view(-1), labels.to(torch.float32))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
