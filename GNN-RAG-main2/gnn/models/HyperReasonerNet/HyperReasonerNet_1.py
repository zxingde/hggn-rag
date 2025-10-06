

from models.base_model import BaseModel
from modules.question_encoding.bert_encoder import BERTInstruction

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class HyperReasonerNet(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        Init HyperReasonerNet model.
        """
        super(HyperReasonerNet, self).__init__(args, num_entity, num_relation, num_word)
        self.private_module_def(args, num_entity, num_relation)

    def private_module_def(self, args, num_entity, num_relation):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])


    def forward(self, batch, training=False):
        print(batch)
        return 0



