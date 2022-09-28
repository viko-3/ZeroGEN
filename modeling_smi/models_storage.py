import os


from modeling_smi.transformer import Transformer, TransformerTrainer, transformer_parser


class ModelsStorage():

    def __init__(self):
        self._models = {}
        self.add_model('transformer', Transformer, TransformerTrainer, transformer_parser)


    def add_model(self, name, class_, trainer_, parser_):
        self._models[name] = {'class': class_,
                              'trainer': trainer_,
                              'parser': parser_}

    def get_model_names(self):
        return list(self._models.keys())

    def get_model_trainer(self, name):
        return self._models[name]['trainer']

    def get_model_class(self, name):
        return self._models[name]['class']

    def get_model_train_parser(self, name):
        return self._models[name]['parser']
