from textattack.models.wrappers import ModelWrapper


class ICLModelWrapper(ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, text_input_list):
        pass
      # x_transform = []
      # for i, review in enumerate(text_input_list):
      #   tokens = [x.strip(",") for x in review.split()]
      #   BoW_array = np.zeros((NUM_WORDS,))
      #   for word in tokens:
      #     if word in vocabulary:
      #       if vocabulary[word] < len(BoW_array):
      #         BoW_array[vocabulary[word]] += 1
      #   x_transform.append(BoW_array)
      # x_transform = np.array(x_transform)
      # prediction = self.model.predict(x_transform)
      #
      # return prediction


def textbugger_attack_setup(llm):
    model_wrapper = ICLModelWrapper(llm)

    attack = TextBuggerLi2018.build(model_wrapper)