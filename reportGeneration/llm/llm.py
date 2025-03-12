import ollama

class ChatModel():
    """
    """
    def __init__(self, lang, model, data):
        self.lang = lang
        self.model = model
        self.data = data

    def __cratePromt(self):
        self.instruction_base ="""
        You are an assistant who creates a report on the training of a machine learning model. You receive the entire pipeline and the metrics of the model as input. You must write the most detailed report on them.
            1) describe the initial pipeline and (specify the model equations)
            2) describe the metrics and give them an estimate 
            3) write a learning conclusion 
            4) Write at least 3-5 options for improving the pipeline
        """
        self.fullinstruction = self.instruction_base + f"And you also return an answer in the language (2 letters of which were given to you here {self.lang}. For example , if ru is Russian , if en is English , and so on"

    def qa(self):
        self.__cratePromt()
        response = ollama.chat(
            model=f"{self.model}",
            messages=[
                {"role": "system", "content": self.fullinstruction},
                {"role": "user", "content": self.data}
            ]
        )
        return response['message']['content']
